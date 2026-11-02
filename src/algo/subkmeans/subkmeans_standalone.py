from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import importlib.metadata
import io
import itertools
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time
import warnings

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'src'))
from data import DATASETS, deduplicate, load_dataset, read_fasta

COMMIT = '6de210e1bc4395176a4299aab96e4156541295b5'
SOURCES = {
    'partition/subkmeans.py': '7c8538b076f433c1c7316076790815d24cebf5dfd7471616011f704c8e620077',
    'alternative/nrkmeans.py': 'f97143dba2cf6ca25a25a554567192c97a668176f1ae8e552531bea86763701e',
    'utils/_information_theory.py': '4267c915fe4887bd1b300ab1d56e519d383f3d41635346b1bfb98a0ff06ab6b3',
}

EXPECTED_FULL_N = 1_587
EXPECTED_DEDUP_N = 1_480
EXPECTED_DIMENSIONS = 1_024
EXPECTED_FULL_SUBSET_HASH = '1906375a02bc'

FIXED_SETTINGS = dict(V=None, m=None, cluster_centers=None, mdl_for_noisespace=False,
                      outliers=False, max_iter=300, n_init=1, cost_type='default',
                      threshold_negative_eigenvalue=-1e-7, max_distance=None, precision=None,
                      debug=True)

GEOMETRIES = ('raw1024', 'l2_raw1024')
RECORD_SETS = ('full', 'deduplicated')

TIME_LIMIT = 600
MEMORY_LIMIT = 4 * 1024**3
THREADS = 4


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def verify_source():
    import clustpy
    from clustpy.partition import SubKmeans
    import inspect
    package = Path(clustpy.__file__).parent
    if importlib.metadata.version('clustpy') != '0.0.2':
        raise ValueError('This run requires ClustPy 0.0.2')
    for relative, expected in SOURCES.items():
        if digest(package / relative) != expected:
            raise ValueError(f'ClustPy source mismatch: {relative}')
    if Path(inspect.getfile(SubKmeans)).resolve() != (package / 'partition/subkmeans.py').resolve():
        raise ValueError('Imported estimator differs from checked source')
    return SubKmeans


def effective_start(seed):
    from sklearn.utils import check_random_state
    return int(check_random_state(seed).choice(10_000, 1, replace=False)[0])


def canonical_input(ids, X):
    ids, X = np.asarray(ids), np.asarray(X)
    if ids.ndim != 1 or ids.dtype.kind not in 'US' or len(set(ids.tolist())) != len(ids):
        raise ValueError('Protein IDs must be unique strings')
    if X.ndim != 2 or X.shape[0] != len(ids) or not np.isfinite(X).all():
        raise ValueError('Coordinates must be finite and aligned with IDs')
    order = np.argsort(ids, kind='stable')
    return ids[order], np.ascontiguousarray(X[order])


def build_spaces(full, deduplicated):
    from sklearn.preprocessing import normalize

    raw = np.ascontiguousarray(full.X, dtype=np.float64)
    if raw.shape != (EXPECTED_FULL_N, EXPECTED_DIMENSIONS) or not np.isfinite(raw).all():
        raise ValueError(f'Expected finite full coordinates shaped '
                         f'{(EXPECTED_FULL_N, EXPECTED_DIMENSIONS)}, got {raw.shape}')
    norms = np.linalg.norm(raw, axis=1)
    if not np.isfinite(norms).all() or (norms == 0).any():
        raise ValueError('A record has a zero or non-finite norm, so L2 normalisation is undefined')
    full_spaces = {'raw1024': raw, 'l2_raw1024': normalize(raw, norm='l2', axis=1, copy=True)}

    position = {identifier: index for index, identifier in enumerate(full.ids)}
    missing = [i for i in deduplicated.ids if i not in position]
    if missing:
        raise ValueError(f'{len(missing)} deduplicated proteins are absent from the full set')
    positions = np.array([position[i] for i in deduplicated.ids], dtype=int)
    if len(deduplicated) != EXPECTED_DEDUP_N:
        raise ValueError(f'Expected {EXPECTED_DEDUP_N} deduplicated records, got {len(deduplicated)}')
    if not np.array_equal(full.X[positions], deduplicated.X):
        raise ValueError('Deduplicated coordinates do not match the full matrix')

    spaces = {'full': full_spaces,
              'deduplicated': {name: np.ascontiguousarray(coordinates[positions])
                               for name, coordinates in full_spaces.items()}}
    identifiers = {'full': list(full.ids), 'deduplicated': list(deduplicated.ids)}
    expected_rows = {'full': EXPECTED_FULL_N, 'deduplicated': EXPECTED_DEDUP_N}
    for records, geometries in spaces.items():
        if tuple(geometries) != GEOMETRIES:
            raise ValueError(f'{records} geometries do not match the fixed pair')
        for name, coordinates in geometries.items():
            if coordinates.shape != (expected_rows[records], EXPECTED_DIMENSIONS):
                raise ValueError(f'Unexpected shape for {records}/{name}: {coordinates.shape}')
            if coordinates.dtype != np.float64 or not np.isfinite(coordinates).all():
                raise ValueError(f'Non-finite or non-float64 coordinates in {records}/{name}')
        if not np.allclose(np.linalg.norm(geometries['l2_raw1024'], axis=1), 1.0):
            raise ValueError(f'{records} L2 coordinates are not unit norm')
    return spaces, identifiers


def termination(log, max_iter):
    values = re.findall(r'\[NrKmeans\] Converged in iteration (\d+)', log)
    iteration = int(values[0]) if len(values) == 1 else None
    status = 'converged' if iteration is not None and 0 < iteration < max_iter else 'unknown'
    return status, iteration


def validate_arrays(a, n, d, requested_k):
    needed = {'protein_id', 'label', 'V', 'm', 'centres', 'scatter'}
    if set(a) != needed:
        raise ValueError('Unexpected or missing archive arrays')
    ids, labels = a['protein_id'], a['label']
    if ids.shape != (n,) or ids.dtype.kind not in 'US' or len(np.unique(ids)) != n:
        raise ValueError('Invalid archive IDs')
    if not np.array_equal(ids, np.sort(ids)):
        raise ValueError('Archive IDs are not canonical')
    if labels.shape != (n,) or labels.dtype.kind not in 'iu':
        raise ValueError('Invalid native labels')
    centres = a['centres']
    k = centres.shape[0] if centres.ndim == 2 else 0
    if not 1 <= k <= requested_k or centres.shape != (k, d):
        raise ValueError('Invalid returned centres')
    if not np.array_equal(np.unique(labels), np.arange(k)):
        raise ValueError('Labels disagree with realised centres or disabled outlier setting')
    if a['m'].shape != () or a['m'].dtype.kind not in 'iu' or not 0 <= int(a['m']) <= d:
        raise ValueError('Invalid subspace dimension')
    for key, shape in [('V', (d, d)), ('scatter', (d, d)), ('centres', (k, d))]:
        if a[key].shape != shape or not np.isfinite(a[key]).all():
            raise ValueError(f'Invalid {key}')
    if not np.allclose(a['V'].T @ a['V'], np.eye(d), atol=1e-6):
        raise ValueError('Saved rotation is not orthogonal')
    return k


def derived_diagnostics(X64, arrays):
    labels = np.asarray(arrays['label'])
    centres = np.asarray(arrays['centres'], dtype=np.float64)
    V = np.asarray(arrays['V'], dtype=np.float64)
    k = centres.shape[0]

    means = np.stack([X64[labels == c].mean(axis=0) for c in range(k)])
    wcss_label_means = float(((X64 - means[labels]) ** 2).sum())
    wcss_saved_centres = float(((X64 - centres[labels]) ** 2).sum())

    offsets = centres - centres.mean(axis=0)
    rank = int(np.linalg.matrix_rank(offsets))
    total_energy = float((offsets ** 2).sum())
    captured = float(((offsets @ V[:, :rank]) ** 2).sum())
    captured_energy = captured / total_energy if total_energy > 0 else float('nan')

    squared = (centres ** 2).sum(axis=1) - 2.0 * (X64 @ centres.T)
    nearest = np.asarray(squared.argmin(axis=1), dtype=labels.dtype)
    moved = int((nearest != labels).sum())

    return dict(wcss_label_means=wcss_label_means, wcss_saved_centres=wcss_saved_centres,
                centre_offset_rank=rank, captured_energy=captured_energy,
                moved_under_full_reassignment=moved, native_m=int(arrays['m']))


def fit_restarts(ids, X, k, seeds, dtype):
    estimator_type = verify_source()
    ids, X64 = canonical_input(ids, X)
    if X64.dtype != np.float64:
        raise ValueError('Coordinates must reach this function in float64')
    starts = [effective_start(seed) for seed in seeds]
    if len(set(starts)) != len(starts):
        raise ValueError(f'Seeds {list(seeds)} do not give distinct effective starts {starts}')
    fitted = X64.astype(np.float32) if dtype == 'float32' else X64

    restarts, best_arrays, best_evidence = [], None, None
    for seed, start in zip(seeds, starts):
        settings = dict(FIXED_SETTINGS, n_clusters=k, random_state=seed)
        log = io.StringIO()
        started = time.perf_counter()
        with warnings.catch_warnings(record=True) as caught, contextlib.redirect_stdout(log):
            warnings.simplefilter('always')
            try:
                estimator = estimator_type(**settings).fit(fitted)
            except BaseException:
                sys.__stdout__.write(log.getvalue())
                sys.__stdout__.flush()
                raise
        seconds = time.perf_counter() - started
        arrays = dict(protein_id=ids, label=np.asarray(estimator.labels_),
                      V=np.asarray(estimator.V), m=np.array(estimator.m),
                      centres=np.asarray(estimator.cluster_centers),
                      scatter=np.asarray(estimator.scatter_matrix_))
        returned_k = validate_arrays(arrays, *fitted.shape, k)
        if returned_k != estimator.n_clusters:
            raise ValueError('Estimator count differs from labels')
        projected = fitted @ arrays['V'][:, :int(arrays['m'])]
        np.testing.assert_allclose(projected, estimator.transform_clustered_space(fitted),
                                   rtol=1e-12, atol=1e-12)
        status, iteration = termination(log.getvalue(), settings['max_iter'])
        evidence = dict(seed=int(seed), effective_start=start, returned_k=returned_k,
                        m=int(arrays['m']), objective=float(estimator.calculate_cost_function(fitted)),
                        fit_seconds=seconds, optimisation_status=status, iterations=iteration,
                        warnings=[f'{w.category.__name__}: {w.message}' for w in caught],
                        native_log=log.getvalue())
        restarts.append(evidence)
        if best_evidence is None or evidence['objective'] < best_evidence['objective']:
            best_arrays, best_evidence = arrays, evidence

    diagnostics = derived_diagnostics(X64, best_arrays)
    return best_arrays, dict(selected=best_evidence, restarts=restarts, diagnostics=diagnostics)


def reserve_run(root, run_id):
    if not re.fullmatch(r'[A-Za-z0-9_-]+', run_id):
        raise ValueError('Unsafe run ID')
    root = Path(root).resolve()
    paths = {name: root / name / run_id for name in ['assignments', 'metrics', 'logs']}
    if any(p.exists() or p.parent.is_symlink() for p in paths.values()):
        raise FileExistsError('Run exists or parent is a link; no files will be reset')
    for path in paths.values():
        if not path.resolve().is_relative_to(root):
            raise ValueError('Output escapes root')
        path.mkdir(parents=True, exist_ok=False)
    return paths


RUN_FIELDS = ['run_id', 'purpose', 'geometry', 'records', 'n', 'd', 'subset_hash', 'requested_k',
              'returned_k', 'm', 'fit_dtype', 'restart_count', 'selected_seed',
              'selected_effective_start', 'objective', 'fit_seconds', 'optimisation_status',
              'iterations', 'wcss_label_means', 'wcss_saved_centres', 'centre_offset_rank',
              'captured_energy', 'moved_under_full_reassignment']

RESTART_FIELDS = ['run_id', 'seed', 'effective_start', 'returned_k', 'm', 'objective',
                  'fit_seconds', 'optimisation_status', 'iterations', 'selected']


def restart_value(metadata, restart, field):
    if field == 'run_id':
        return metadata['run_id']
    if field == 'selected':
        return int(restart['seed'] == metadata['selected_seed'])
    return restart[field]


def save_result(root, run_id, arrays, metadata):
    root = Path(root)
    archive = root / 'assignments' / run_id / 'partition.npz'
    with archive.open('xb') as stream:
        np.savez_compressed(stream, **arrays)
    with np.load(archive, allow_pickle=False) as reopened:
        for key in arrays:
            np.testing.assert_array_equal(reopened[key], arrays[key])
    metadata['archive_sha256'] = digest(archive)

    metrics = root / 'metrics' / run_id
    write_json(metrics / 'run.json', metadata)
    with (metrics / 'run.csv').open('x', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=RUN_FIELDS)
        writer.writeheader()
        writer.writerow({key: metadata[key] for key in RUN_FIELDS})
    with (metrics / 'restarts.csv').open('x', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=RESTART_FIELDS)
        writer.writeheader()
        for restart in metadata['restarts']:
            writer.writerow({field: restart_value(metadata, restart, field)
                             for field in RESTART_FIELDS})
    with (metrics / 'cluster_sizes.csv').open('x', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['run_id', 'label', 'n'])
        values, counts = np.unique(arrays['label'], return_counts=True)
        writer.writerows((run_id, int(v), int(c)) for v, c in zip(values, counts))


def validate_saved(root, run_id, require_success=True):
    root = Path(root)
    metadata = json.loads((root / 'metrics' / run_id / 'run.json').read_text())
    archive = root / 'assignments' / run_id / 'partition.npz'
    if metadata['run_id'] != run_id or metadata['purpose'] not in ('diagnostic', 'scientific'):
        raise ValueError('Wrong run identity')
    if metadata['source_commit'] != COMMIT or metadata['source_hashes'] != SOURCES:
        raise ValueError('Wrong library provenance')
    if digest(archive) != metadata['archive_sha256']:
        raise ValueError('Archive hash changed')
    with np.load(archive, allow_pickle=False) as a:
        k = validate_arrays(a, metadata['n'], metadata['d'], metadata['requested_k'])
        if k != metadata['returned_k'] or int(a['m']) != metadata['m']:
            raise ValueError('Metadata differs from arrays')
        subset = hashlib.sha1('\n'.join(a['protein_id'].tolist()).encode()).hexdigest()[:12]
        if subset != metadata['subset_hash']:
            raise ValueError('Subset identity changed')
        values, counts = np.unique(a['label'], return_counts=True)
    with (root / 'metrics' / run_id / 'cluster_sizes.csv').open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    if [(r['run_id'], int(r['label']), int(r['n'])) for r in rows] != [
            (run_id, int(v), int(c)) for v, c in zip(values, counts)]:
        raise ValueError('Cluster-size table differs from native labels')

    selected = [r for r in metadata['restarts'] if r['seed'] == metadata['selected_seed']]
    if len(selected) != 1 or any(r['objective'] < selected[0]['objective']
                                 for r in metadata['restarts']):
        raise ValueError('The selected restart is not the cheapest fit')
    starts = [r['effective_start'] for r in metadata['restarts']]
    if len(set(starts)) != len(starts):
        raise ValueError('Two restarts share an effective start, so they are one fit')
    status, iteration = termination(selected[0]['native_log'], metadata['settings']['max_iter'])
    if (status, iteration) != (metadata['optimisation_status'], metadata['iterations']):
        raise ValueError('Stopping classification differs from evidence')

    for name, fields, sources in [('run.csv', RUN_FIELDS, [None]),
                                  ('restarts.csv', RESTART_FIELDS, metadata['restarts'])]:
        with (root / 'metrics' / run_id / name).open(newline='') as stream:
            table = list(csv.DictReader(stream))
        if len(table) != len(sources) or list(table[0]) != fields:
            raise ValueError(f'{name} does not match the recorded run')
        for row, source in zip(table, sources):
            for field in fields:
                value = metadata[field] if source is None else restart_value(metadata, source, field)
                if row[field] != ('' if value is None else str(value)):
                    raise ValueError(f'{name} differs from metadata at {field}')
    if require_success:
        success = json.loads((root / 'metrics' / run_id / 'success.json').read_text())
        resource = json.loads((root / 'logs' / run_id / 'resource.json').read_text())
        if success['archive_sha256'] != digest(archive) or resource['status'] != 'completed':
            raise ValueError('No validated successful process outcome')
    return metadata


def bounded(command, log_path, seconds, memory_bytes, threads=THREADS):
    import win32api
    import win32con
    import win32job
    job = win32job.CreateJobObject(None, '')
    info = win32job.QueryInformationJobObject(job, win32job.JobObjectExtendedLimitInformation)
    info['BasicLimitInformation']['LimitFlags'] = (win32job.JOB_OBJECT_LIMIT_JOB_MEMORY |
                                                   win32job.JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE)
    info['JobMemoryLimit'] = memory_bytes
    win32job.SetInformationJobObject(job, win32job.JobObjectExtendedLimitInformation, info)
    applied = win32job.QueryInformationJobObject(job, win32job.JobObjectExtendedLimitInformation)
    if applied['JobMemoryLimit'] != memory_bytes:
        job.Close()
        raise RuntimeError('Windows did not apply the memory limit')
    env = os.environ.copy()
    env.update({key: str(threads) for key in
                ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                 'NUMEXPR_NUM_THREADS', 'LOKY_MAX_CPU_COUNT']})
    env['MPLBACKEND'] = 'Agg'
    env['MPLCONFIGDIR'] = str(Path(tempfile.gettempdir()) / 'subkmeans_matplotlib')
    env['SUBKMEANS_BOUNDED'] = '1'
    bootstrap = ("import subprocess,sys; assert sys.stdin.readline().strip()=='GO';"
                 " sys.exit(subprocess.call(sys.argv[1:]))")
    started = time.perf_counter()
    process = None
    record = dict(command=command, seconds_limit=seconds, memory_limit_bytes=memory_bytes,
                  memory_measure='Windows-reported peak job commitment; may include denied '
                                 'allocations; supervisor excluded',
                  applied_memory_limit_bytes=applied['JobMemoryLimit'],
                  status='launch_error', exit_code=None)
    try:
        with Path(log_path).open('x', encoding='utf-8') as log:
            process = subprocess.Popen([sys.executable, '-c', bootstrap, *command], cwd=ROOT,
                                       env=env, stdin=subprocess.PIPE, stdout=log, stderr=log,
                                       text=True, creationflags=subprocess.CREATE_NO_WINDOW)
            handle = win32api.OpenProcess(win32con.PROCESS_SET_QUOTA | win32con.PROCESS_TERMINATE,
                                          False, process.pid)
            try:
                win32job.AssignProcessToJobObject(job, handle)
            finally:
                handle.Close()
            process.stdin.write('GO\n')
            process.stdin.close()
            try:
                record['exit_code'] = process.wait(timeout=seconds)
                record['status'] = 'completed' if process.returncode == 0 else 'failed'
            except subprocess.TimeoutExpired:
                record['status'] = 'timeout'
    finally:
        win32job.TerminateJobObject(job, 1)
        if process is not None:
            if process.poll() is None:
                process.kill()
            process.wait(timeout=10)
        for _ in range(100):
            accounting = win32job.QueryInformationJobObject(
                job, win32job.JobObjectBasicAccountingInformation)
            if accounting['ActiveProcesses'] == 0:
                break
            time.sleep(.05)
        record['active_processes_after_cleanup'] = accounting['ActiveProcesses']
        record['peak_job_committed_bytes'] = win32job.QueryInformationJobObject(
            job, win32job.JobObjectExtendedLimitInformation)['PeakJobMemoryUsed']
        record['wall_seconds'] = time.perf_counter() - started
        job.Close()
        write_json(Path(log_path).with_suffix('.json'), record)
    if record['active_processes_after_cleanup']:
        raise RuntimeError('Job cleanup did not finish')
    return record


def subset_hash(ids):
    return hashlib.sha1('\n'.join(sorted(ids)).encode()).hexdigest()[:12]


def prepare_all():
    full = load_dataset('phosphatase')
    if full.subset_hash != EXPECTED_FULL_SUBSET_HASH:
        raise ValueError(f'Unexpected full record set: {full.subset_hash}')
    return build_spaces(full, deduplicate(full))


def prepare(records):
    spaces, identifiers = prepare_all()
    return spaces[records], identifiers[records]


def sequence_crosscheck(identifiers):
    sequences = read_fasta(DATASETS['phosphatase'][2])
    lowest_per_sequence = {}
    for identifier, sequence in sorted(sequences.items()):
        lowest_per_sequence.setdefault(sequence, identifier)
    fasta_ids = set(lowest_per_sequence.values())
    deduplicated = set(identifiers['deduplicated'])
    return dict(fasta_records=len(sequences), fasta_unique_sequences=len(lowest_per_sequence),
                deduplicated_records=len(deduplicated),
                only_in_vector_route=sorted(deduplicated - fasta_ids),
                only_in_sequence_route=sorted(fasta_ids - deduplicated),
                routes_agree=deduplicated == fasta_ids)


def worker(root, run_id, geometry, records, k, seeds, dtype, purpose, threads=THREADS):
    if os.environ.get('SUBKMEANS_BOUNDED') != '1':
        raise RuntimeError('Launch through --fit so the resource limits are applied')
    geometries, ids = prepare(records)
    X = geometries[geometry]
    from threadpoolctl import threadpool_info, threadpool_limits
    with threadpool_limits(limits=threads):
        arrays, evidence = fit_restarts(ids, X, k, seeds, dtype)
        pools = threadpool_info()
    selected, diagnostics = evidence['selected'], evidence['diagnostics']
    try:
        revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True,
                                           stderr=subprocess.DEVNULL).strip()
        dirty = bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True,
                                             stderr=subprocess.DEVNULL).strip())
    except (OSError, subprocess.CalledProcessError):
        revision, dirty = 'unknown', False
    subset_hash = hashlib.sha1('\n'.join(sorted(ids)).encode()).hexdigest()[:12]
    metadata = dict(
        run_id=run_id, purpose=purpose, geometry=geometry, records=records, n=len(ids),
        d=X.shape[1], subset_hash=subset_hash, requested_k=k, returned_k=selected['returned_k'],
        m=selected['m'], fit_dtype=dtype, restart_count=len(seeds),
        selected_seed=selected['seed'], selected_effective_start=selected['effective_start'],
        objective=selected['objective'], fit_seconds=selected['fit_seconds'],
        optimisation_status=selected['optimisation_status'], iterations=selected['iterations'],
        **diagnostics,
        settings=dict(FIXED_SETTINGS, n_clusters=k), requested_seeds=list(seeds),
        restarts=evidence['restarts'], source_commit=COMMIT, source_hashes=SOURCES,
        project_revision=revision + ('-dirty' if dirty else ''),
        runner_sha256=digest(__file__), loader_sha256=digest(ROOT / 'src/data.py'),
        input_sha256={str(p.relative_to(ROOT)): digest(p) for p in DATASETS['phosphatase']},
        versions={p: importlib.metadata.version(p) for p in
                  ['clustpy', 'numpy', 'scipy', 'scikit-learn', 'threadpoolctl']},
        python=sys.version, executable=sys.executable, threads=threads, thread_pools=pools)
    for restart in evidence['restarts']:
        print(restart['native_log'], flush=True)
        for message in restart['warnings']:
            print(message, flush=True)
    save_result(root, run_id, arrays, metadata)
    validate_saved(root, run_id, require_success=False)


RESTARTS_PER_FIT = 10
OUTER_SEEDS = tuple(range(10))

BASELINE = ROOT / 'results/kmeans/metrics'
BASELINE_SELECTION = BASELINE / 'kmeans_standalone_elbow_selection.csv'
BASELINE_RUNS = BASELINE / 'kmeans_standalone_runs.csv'

MANIFEST_FIELDS = ['row', 'geometry', 'records', 'requested_k', 'outer_seed', 'k_source', 'roles',
                   'restart_seeds', 'effective_starts', 'fit_dtype', 'status']


def selected_k_table():
    import pandas as pd

    selection = pd.read_csv(BASELINE_SELECTION)
    primary = selection[(selection.is_primary) & (selection.records == 'full')
                        & (selection.space.isin(GEOMETRIES))]
    table = {}
    for geometry in GEOMETRIES:
        rows = primary[primary.space == geometry].set_index('seed').sort_index()
        if list(rows.index) != list(OUTER_SEEDS):
            raise ValueError(f'{geometry} does not have one primary selection per seed 0 to 9')
        table[geometry] = {int(seed): int(k) for seed, k in rows.selected_k.items()}
    return table


def modal_k(selected):
    modes = {}
    for geometry, by_seed in selected.items():
        counts = {}
        for k in by_seed.values():
            counts[k] = counts.get(k, 0) + 1
        best = max(counts.values())
        modes[geometry] = min(k for k, count in counts.items() if count == best)
    return modes


OVERFLOW_SEED_BASE = 1_000


def restart_seed_pool(block_start, used, overflow):
    seeds = []
    for candidate in range(block_start, block_start + RESTARTS_PER_FIT):
        start = effective_start(candidate)
        if start not in used:
            used.add(start)
            seeds.append(candidate)
    while len(seeds) < RESTARTS_PER_FIT:
        candidate = next(overflow)
        start = effective_start(candidate)
        if start not in used:
            used.add(start)
            seeds.append(candidate)
    return seeds


def z14_pairings():
    import pandas as pd

    runs = pd.read_csv(BASELINE_RUNS)
    pca = runs[(runs.space.str.startswith('pca')) & (runs.role == 'procedure_selected')]
    return [dict(space=str(row.space), records=str(row.records), seed=int(row.seed),
                 realised_k=int(row.realised_k)) for row in pca.itertuples()]


def build_manifest():
    selected = selected_k_table()
    modes = modal_k(selected)
    rows, index = {}, {}

    def add(geometry, records, k, seed, role, source):
        key = (geometry, records, int(k), int(seed))
        if key in index:
            entry = index[key]
            if role not in entry['roles']:
                entry['roles'].append(role)
                entry['k_source'].append(source)
            return entry
        entry = dict(geometry=geometry, records=records, requested_k=int(k), outer_seed=int(seed),
                     roles=[role], k_source=[source])
        index[key] = entry
        rows[key] = entry
        return entry

    for geometry in GEOMETRIES:
        for seed in OUTER_SEEDS:
            for records in RECORD_SETS:
                add(geometry, records, selected[geometry][seed], seed, 'procedure_selected',
                    f'baseline elbow, {geometry} seed {seed}')

    for geometry in GEOMETRIES:
        for seed in OUTER_SEEDS:
            if selected[geometry][seed] != modes[geometry]:
                for records in RECORD_SETS:
                    add(geometry, records, modes[geometry], seed, 'fixed_k_stability',
                        f'modal k for {geometry}')

    for seed in OUTER_SEEDS:
        for records in RECORD_SETS:
            for geometry in GEOMETRIES:
                add(geometry, records, selected['raw1024'][seed], seed, 'geometry_controlled',
                    f'raw1024 anchor, seed {seed}')

    for pairing in z14_pairings():
        add('raw1024', pairing['records'], pairing['realised_k'], pairing['seed'], 'z14_pca_pair',
            f"{pairing['space']} realised k, seed {pairing['seed']}")

    manifest = []
    for number, key in enumerate(sorted(rows), start=1):
        entry = rows[key]
        manifest.append(entry)
        entry['row'] = f'SKM{number:03d}'
    used_by_group = {}
    overflow = itertools.count(OVERFLOW_SEED_BASE)
    for entry in manifest:
        group = (entry['geometry'], entry['records'], entry['requested_k'])
        used = used_by_group.setdefault(group, set())
        entry['restart_seeds'] = restart_seed_pool(entry['outer_seed'] * RESTARTS_PER_FIT, used,
                                                   overflow)
        entry['effective_starts'] = [effective_start(seed) for seed in entry['restart_seeds']]
        entry['fit_dtype'] = 'float64'
        entry['status'] = 'planned'
    return manifest, selected, modes


def write_manifest(root):
    import pandas as pd

    manifest, selected, modes = build_manifest()
    spaces, identifiers = prepare_all()
    directory = Path(root) / 'manifest'
    directory.mkdir(parents=True, exist_ok=False)

    with (directory / 'manifest.csv').open('x', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        for entry in manifest:
            writer.writerow({field: (' '.join(str(value) for value in entry[field])
                                     if isinstance(entry[field], list) else entry[field])
                             for field in MANIFEST_FIELDS})

    roles = {}
    for entry in manifest:
        for role in entry['roles']:
            roles[role] = roles.get(role, 0) + 1
    provenance = dict(
        built=time.strftime('%Y-%m-%d'), fits=len(manifest), role_counts=roles,
        fits_by_group={f'{geometry}/{records}': sum(1 for e in manifest if e['geometry'] == geometry
                                                    and e['records'] == records)
                       for geometry in GEOMETRIES for records in RECORD_SETS},
        restarts_per_fit=RESTARTS_PER_FIT,
        estimator_calls=len(manifest) * RESTARTS_PER_FIT,
        outer_seeds=list(OUTER_SEEDS), selected_k=selected, modal_k=modes,
        substituted_restart_seeds=[dict(row=entry['row'], seeds=entry['restart_seeds'])
                                   for entry in manifest
                                   if entry['restart_seeds'] != list(range(
                                       entry['outer_seed'] * RESTARTS_PER_FIT,
                                       entry['outer_seed'] * RESTARTS_PER_FIT + RESTARTS_PER_FIT))],
        settings=FIXED_SETTINGS, fit_dtype='float64',
        record_sets={records: dict(n=len(ids), subset_hash=subset_hash(ids))
                     for records, ids in identifiers.items()},
        coordinates={f'{records}/{geometry}': dict(
            shape=list(coordinates.shape), dtype=str(coordinates.dtype),
            sha256=hashlib.sha256(np.ascontiguousarray(coordinates)).hexdigest())
            for records, geometries in spaces.items() for geometry, coordinates in geometries.items()},
        baseline_tables={path.name: digest(path) for path in [BASELINE_SELECTION, BASELINE_RUNS]},
        source_commit=COMMIT, source_hashes=SOURCES, runner_sha256=digest(__file__),
        loader_sha256=digest(ROOT / 'src/data.py'),
        input_sha256={str(p.relative_to(ROOT)): digest(p) for p in DATASETS['phosphatase']},
        versions={p: importlib.metadata.version(p) for p in
                  ['clustpy', 'numpy', 'scipy', 'scikit-learn', 'pandas']})
    write_json(directory / 'manifest.json', provenance)

    table = pd.read_csv(directory / 'manifest.csv')
    if len(table) != len(manifest) or table.row.duplicated().any():
        raise ValueError('The written manifest does not match what was built')
    for entry in manifest:
        row = table[table.row == entry['row']].iloc[0]
        if (row.geometry, row.records, int(row.requested_k), int(row.outer_seed)) != (
                entry['geometry'], entry['records'], entry['requested_k'], entry['outer_seed']):
            raise ValueError(f"Manifest row {entry['row']} changed on the way to disk")
        starts = [int(value) for value in str(row.effective_starts).split()]
        if len(set(starts)) != RESTARTS_PER_FIT:
            raise ValueError(f"Manifest row {entry['row']} repeats an effective start")
    for group, used in [(g, {start for entry in manifest
                             if (entry['geometry'], entry['records'], entry['requested_k']) == g
                             for start in entry['effective_starts']}) for g in
                        {(e['geometry'], e['records'], e['requested_k']) for e in manifest}]:
        expected = RESTARTS_PER_FIT * sum(
            1 for entry in manifest
            if (entry['geometry'], entry['records'], entry['requested_k']) == group)
        if len(used) != expected:
            raise ValueError(f'Group {group} shares an effective start between fits')
    return provenance


GRID_TABLES = dict(runs='subkmeans_standalone_runs.csv',
                   restarts='subkmeans_standalone_restarts.csv',
                   clusters='subkmeans_standalone_clusters.csv',
                   failures='subkmeans_standalone_failures.csv')

CONSECUTIVE_FAILURE_LIMIT = 5


def read_manifest(root):
    import pandas as pd

    directory = Path(root) / 'manifest'
    provenance = json.loads((directory / 'manifest.json').read_text())
    table = pd.read_csv(directory / 'manifest.csv')
    if len(table) != provenance['fits'] or table.row.duplicated().any():
        raise ValueError('The manifest table and its provenance disagree')

    if provenance['source_hashes'] != SOURCES or provenance['source_commit'] != COMMIT:
        raise ValueError('The manifest was frozen against different library source')
    for name, expected in provenance['input_sha256'].items():
        if digest(ROOT / name) != expected:
            raise ValueError(f'Input changed since the manifest was frozen: {name}')
    for name, expected in provenance['baseline_tables'].items():
        if digest(BASELINE / name) != expected:
            raise ValueError(f'Baseline table changed since the manifest was frozen: {name}')

    spaces, identifiers = prepare_all()
    for key, recorded in provenance['coordinates'].items():
        records, geometry = key.split('/')
        actual = hashlib.sha256(np.ascontiguousarray(spaces[records][geometry])).hexdigest()
        if actual != recorded['sha256']:
            raise ValueError(f'Coordinates changed since the manifest was frozen: {key}')
    return table, provenance, spaces, identifiers


def grid_tables(root, table):
    import pandas as pd

    runs, restarts, clusters = [], [], []
    for row in table.itertuples():
        metrics = Path(root) / 'metrics' / row.row
        if not (metrics / 'success.json').exists():
            continue
        metadata = validate_saved(root, row.row)
        record = {field: metadata[field] for field in RUN_FIELDS}
        record['outer_seed'] = int(row.outer_seed)
        record['roles'] = row.roles
        record['k_source'] = row.k_source
        record['archive_sha256'] = metadata['archive_sha256']
        record['project_revision'] = metadata['project_revision']
        resource = json.loads((Path(root) / 'logs' / row.row / 'resource.json').read_text())
        record['wall_seconds'] = resource['wall_seconds']
        record['peak_job_committed_bytes'] = resource['peak_job_committed_bytes']
        runs.append(record)
        restarts.append(pd.read_csv(metrics / 'restarts.csv'))
        clusters.append(pd.read_csv(metrics / 'cluster_sizes.csv'))
    runs = pd.DataFrame(runs)
    if len(runs):
        runs.insert(runs.columns.get_loc('records') + 1, 'outer_seed', runs.pop('outer_seed'))
    return (runs, pd.concat(restarts, ignore_index=True) if restarts else pd.DataFrame(),
            pd.concat(clusters, ignore_index=True) if clusters else pd.DataFrame())


def run_row(root, row, threads):
    run_id = row.row
    seeds = [int(value) for value in str(row.restart_seeds).split()]
    paths = reserve_run(root, run_id)
    write_json(paths['logs'] / 'request.json', dict(
        run_id=run_id, purpose='scientific', geometry=row.geometry, records=row.records,
        requested_k=int(row.requested_k), outer_seed=int(row.outer_seed), roles=str(row.roles),
        requested_seeds=seeds, effective_starts=[effective_start(seed) for seed in seeds],
        fit_dtype=str(row.fit_dtype), settings=dict(FIXED_SETTINGS, n_clusters=int(row.requested_k)),
        source_commit=COMMIT, source_hashes=SOURCES, runner_sha256=digest(__file__),
        loader_sha256=digest(ROOT / 'src/data.py'),
        input_sha256={str(p.relative_to(ROOT)): digest(p) for p in DATASETS['phosphatase']},
        seconds_limit=TIME_LIMIT, memory_limit_bytes=MEMORY_LIMIT, threads=threads))
    result = bounded([sys.executable, str(Path(__file__).resolve()), '--worker',
                      '--run-id', run_id, '--geometry', str(row.geometry),
                      '--records', str(row.records), '--k', str(int(row.requested_k)),
                      '--seeds', ','.join(str(seed) for seed in seeds),
                      '--dtype', str(row.fit_dtype), '--purpose', 'scientific',
                      '--threads', str(threads), '--output-root', str(Path(root).resolve())],
                     paths['logs'] / 'resource.log', TIME_LIMIT, MEMORY_LIMIT, threads)
    if result['status'] != 'completed':
        raise RuntimeError(f"bounded fit {result['status']}, exit {result['exit_code']}")
    metadata = validate_saved(root, run_id, require_success=False)
    write_json(paths['metrics'] / 'success.json', {'archive_sha256': metadata['archive_sha256']})
    validate_saved(root, run_id)
    return run_id


def run_scientific(root, limit=None, jobs=1, threads=THREADS):
    import pandas as pd
    from concurrent.futures import ThreadPoolExecutor
    import threading

    cpus = os.cpu_count() or 1
    if jobs < 1 or threads < 1:
        raise ValueError('jobs and threads must both be at least 1')
    if jobs * threads > cpus:
        raise ValueError(f'{jobs} jobs x {threads} threads is more than the {cpus} logical CPUs; '
                         f'lower --threads')
    root = Path(root)
    table, provenance, _, _ = read_manifest(root)
    position = {row.row: number for number, row in enumerate(table.itertuples(), start=1)}
    pending = [row for row in table.itertuples()
               if not (root / 'metrics' / row.row / 'success.json').exists()]
    skipped = len(table) - len(pending)
    if limit is not None:
        pending = pending[:limit]

    started = time.perf_counter()
    stop = threading.Event()
    lock = threading.Lock()
    failures, consecutive = [], [0]

    def attempt(row):
        if stop.is_set():
            return 'not_started'
        print(f'[{position[row.row]}/{len(table)}] {row.row} {row.geometry}/{row.records} '
              f'k={row.requested_k} seed={row.outer_seed} '
              f'elapsed={time.perf_counter() - started:.0f}s', flush=True)
        try:
            run_row(root, row, threads)
        except Exception as error:
            with lock:
                failures.append(dict(row=row.row, geometry=row.geometry, records=row.records,
                                     requested_k=int(row.requested_k),
                                     outer_seed=int(row.outer_seed),
                                     reason=f'{type(error).__name__}: {error}'))
                consecutive[0] += 1
                print(f'  FAILED {row.row}: {type(error).__name__}: {error}', flush=True)
                if consecutive[0] >= CONSECUTIVE_FAILURE_LIMIT and not stop.is_set():
                    print(f'Stopping: {consecutive[0]} fits failed in a row', flush=True)
                    stop.set()
            return 'failed'
        with lock:
            consecutive[0] = 0
        return 'completed'

    with ThreadPoolExecutor(max_workers=jobs) as pool:
        outcomes = list(pool.map(attempt, pending))
    completed, not_started = outcomes.count('completed'), outcomes.count('not_started')

    runs, restarts, clusters = grid_tables(root, table)
    metrics = root / 'metrics'
    for frame, name in [(runs, GRID_TABLES['runs']), (restarts, GRID_TABLES['restarts']),
                        (clusters, GRID_TABLES['clusters']),
                        (pd.DataFrame(failures, columns=['row', 'geometry', 'records',
                                                         'requested_k', 'outer_seed', 'reason']),
                         GRID_TABLES['failures'])]:
        frame.to_csv(metrics / name, index=False)

    summary = dict(planned=len(table), validated=len(runs), completed_now=completed,
                   skipped_already_done=skipped, not_started=not_started, failed=len(failures),
                   jobs=jobs, threads_per_fit=threads,
                   wall_seconds=time.perf_counter() - started,
                   estimator_calls=int(runs.restart_count.sum()) if len(runs) else 0,
                   manifest_frozen=provenance['built'])
    print(json.dumps(summary, indent=2), flush=True)
    return summary


CONTROL_PERMUTATIONS = 5
SCORE_TABLES = dict(scores='subkmeans_standalone_scores.csv',
                    objective='subkmeans_standalone_objective_vs_kmeans.csv',
                    agreement='subkmeans_standalone_partition_agreement.csv',
                    agreement_summary='subkmeans_standalone_agreement_summary.csv')
BASELINE_ASSIGNMENTS = ROOT / 'results/kmeans/assignments/kmeans_standalone'


def internal_scores(X, labels):
    from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
    scores = dict(silhouette=float(silhouette_score(X, labels, metric='euclidean')),
                  calinski_harabasz=float(calinski_harabasz_score(X, labels)),
                  davies_bouldin=float(davies_bouldin_score(X, labels)))
    if not all(np.isfinite(value) for value in scores.values()):
        raise ValueError('all internal scores must be finite')
    return scores


def control_summaries(X, labels, seed):
    rng = np.random.default_rng(seed)
    controls = [internal_scores(X, rng.permutation(labels)) for _ in range(CONTROL_PERMUTATIONS)]
    summaries = {}
    for metric in ('silhouette', 'calinski_harabasz', 'davies_bouldin'):
        values = np.asarray([control[metric] for control in controls], dtype=float)
        summaries.update({f'control_{metric}_minimum': float(values.min()),
                          f'control_{metric}_median': float(np.median(values)),
                          f'control_{metric}_maximum': float(values.max()),
                          f'control_{metric}_range': float(np.ptp(values))})
    return summaries


def aligned_ari(left, right):
    import pandas as pd
    from sklearn.metrics import adjusted_rand_score
    frames = []
    for side, payload in (('left', left), ('right', right)):
        ids = np.asarray(payload['protein_id'], dtype=str)
        labels = np.asarray(payload['label'])
        if ids.ndim != 1 or labels.shape != ids.shape or len(np.unique(ids)) != len(ids):
            raise ValueError(f'{side} partition requires unique aligned protein IDs and labels')
        frames.append(pd.DataFrame({'protein_id': ids, side: labels}))
    aligned = frames[0].merge(frames[1], on='protein_id', how='inner', validate='one_to_one')
    if aligned.empty:
        raise ValueError('partitions have no protein IDs in common')
    return len(aligned), float(adjusted_rand_score(aligned['left'], aligned['right']))


def agreement_table(runs, payloads):
    import pandas as pd
    from itertools import combinations

    rows = []
    has = lambda frame, role: frame[frame['roles'].str.split().apply(lambda roles: role in roles)]

    def add(comparison, left, right, geometry, records):
        n_common, ari = aligned_ari(payloads[left['run_id']], payloads[right['run_id']])
        rows.append(dict(comparison=comparison, geometry=geometry, records=records,
                         left_run=left['run_id'], right_run=right['run_id'],
                         left_k=int(left['requested_k']), right_k=int(right['requested_k']),
                         n_common=n_common, adjusted_rand=ari))

    def one(frame, **match):
        selected = frame
        for column, value in match.items():
            selected = selected[selected[column] == value]
        if len(selected) != 1:
            raise ValueError(f'expected one partition for {match}, found {len(selected)}')
        return selected.iloc[0]

    procedure = has(runs, 'procedure_selected')
    for (geometry, records), group in procedure.groupby(['geometry', 'records'], sort=True):
        ordered = [row for _, row in group.sort_values('outer_seed').iterrows()]
        if len(ordered) != len(OUTER_SEEDS):
            raise ValueError(f'{geometry}/{records} does not have all ten procedure partitions')
        for left, right in combinations(ordered, 2):
            add('procedure_selected_stability', left, right, geometry, records)

    modes = modal_k(selected_k_table())
    for geometry in GEOMETRIES:
        for records in RECORD_SETS:
            chosen = [one(runs, geometry=geometry, records=records, requested_k=modes[geometry],
                          outer_seed=seed) for seed in OUTER_SEEDS]
            for left, right in combinations(chosen, 2):
                add('fixed_k_stability', left, right, geometry, records)

    for _, left in procedure[procedure['records'] == 'full'].iterrows():
        right = one(procedure, geometry=left['geometry'], records='deduplicated',
                    outer_seed=left['outer_seed'])
        add('full_vs_deduplicated', left, right, left['geometry'], 'full_vs_deduplicated')

    anchors = selected_k_table()['raw1024']
    for records in RECORD_SETS:
        for seed in OUTER_SEEDS:
            add('raw_vs_l2', one(procedure, geometry='raw1024', records=records, outer_seed=seed),
                one(procedure, geometry='l2_raw1024', records=records, outer_seed=seed),
                'raw_vs_l2', records)
            add('raw_vs_l2_controlled',
                one(runs, geometry='raw1024', records=records, outer_seed=seed,
                    requested_k=anchors[seed]),
                one(runs, geometry='l2_raw1024', records=records, outer_seed=seed,
                    requested_k=anchors[seed]),
                'raw_vs_l2', records)

    agreement = pd.DataFrame(rows)
    if agreement['adjusted_rand'].isna().any() or not agreement['adjusted_rand'].between(-1, 1).all():
        raise ValueError('all adjusted Rand scores must be finite and lie between -1 and 1')
    return agreement


def agreement_summary(agreement):
    summary = (agreement.groupby(['comparison', 'geometry', 'records'], sort=True)['adjusted_rand']
               .agg(['count', 'median', 'min', 'max']).reset_index()
               .rename(columns={'min': 'minimum', 'max': 'maximum'}))
    summary['range'] = summary['maximum'] - summary['minimum']
    return summary


def wcss_from_labels(X, labels):
    labels = np.asarray(labels)
    means = np.stack([X[labels == c].mean(axis=0) for c in np.unique(labels)])
    index = np.searchsorted(np.unique(labels), labels)
    return float(((X - means[index]) ** 2).sum())


def objective_against_kmeans(runs, payloads, spaces, identifiers):
    import pandas as pd
    baseline = pd.read_csv(BASELINE_RUNS)
    rows = []
    for _, run in runs.iterrows():
        match = baseline[(baseline['space'] == run['geometry']) & (baseline['records'] == run['records'])
                         & (baseline['seed'] == run['outer_seed'])
                         & (baseline['requested_k'] == run['requested_k'])]
        if match.empty:
            continue
        if len(match) != 1:
            raise ValueError(f"{run['run_id']} matches {len(match)} baseline rows")
        base = match.iloc[0]
        ids = np.asarray(identifiers[run['records']])
        order = np.argsort(ids, kind='stable')
        X = spaces[run['records']][run['geometry']][order]
        canonical = ids[order]
        with np.load(BASELINE_ASSIGNMENTS / base['archive'], allow_pickle=False) as archive:
            base_order = np.argsort(archive['protein_id'], kind='stable')
            if not np.array_equal(archive['protein_id'][base_order], canonical):
                raise ValueError(f"{base['archive']} covers different proteins")
            base_labels = archive['label'][base_order]
        payload = payloads[run['run_id']]
        if not np.array_equal(payload['protein_id'], canonical):
            raise ValueError(f"{run['run_id']} covers different proteins")
        subkmeans = wcss_from_labels(X, payload['label'])
        kmeans = wcss_from_labels(X, base_labels)
        rows.append(dict(run_id=run['run_id'], geometry=run['geometry'], records=run['records'],
                         outer_seed=int(run['outer_seed']), requested_k=int(run['requested_k']),
                         roles=run['roles'], baseline_archive=base['archive'],
                         baseline_role=base['role'], subkmeans_wcss=subkmeans,
                         kmeans_wcss_recomputed=kmeans,
                         kmeans_recorded_minus_recomputed=float(base['inertia']) - kmeans,
                         wcss_gap=subkmeans - kmeans,
                         relative_gap=(subkmeans - kmeans) / kmeans))
    return pd.DataFrame(rows)


def agreement_against_kmeans(runs, payloads):
    import pandas as pd
    baseline = pd.read_csv(BASELINE_RUNS)
    rows = []
    for _, run in runs.iterrows():
        match = baseline[(baseline['space'] == run['geometry'])
                         & (baseline['records'] == run['records'])
                         & (baseline['seed'] == run['outer_seed'])
                         & (baseline['requested_k'] == run['requested_k'])]
        if match.empty:
            continue
        if len(match) != 1:
            raise ValueError(f"{run['run_id']} matches {len(match)} baseline rows")
        base = match.iloc[0]
        with np.load(BASELINE_ASSIGNMENTS / base['archive'], allow_pickle=False) as archive:
            other = dict(protein_id=archive['protein_id'], label=archive['label'])
        n_common, ari = aligned_ari(payloads[run['run_id']], other)
        rows.append(dict(comparison='subkmeans_vs_kmeans', geometry=run['geometry'],
                         records=run['records'], left_run=run['run_id'],
                         right_run=str(base['archive']), left_k=int(run['requested_k']),
                         right_k=int(base['requested_k']), n_common=n_common,
                         adjusted_rand=ari))
    agreement = pd.DataFrame(rows)
    if agreement.empty or not agreement['adjusted_rand'].between(-1, 1).all():
        raise ValueError('cross-method agreement must be non-empty and lie between -1 and 1')
    return agreement


def score_grid(root):
    import pandas as pd
    root = Path(root)
    runs = pd.read_csv(root / 'metrics' / GRID_TABLES['runs'])
    spaces, identifiers = prepare_all()
    payloads, scores = {}, []
    for number, run in enumerate(runs.itertuples(), start=1):
        with np.load(root / 'assignments' / run.run_id / 'partition.npz', allow_pickle=False) as archive:
            payload = dict(protein_id=archive['protein_id'], label=archive['label'])
        payloads[run.run_id] = payload
        ids = np.asarray(identifiers[run.records])
        order = np.argsort(ids, kind='stable')
        if not np.array_equal(ids[order], payload['protein_id']):
            raise ValueError(f'{run.run_id} does not cover its record set in canonical order')
        X = spaces[run.records][run.geometry][order]
        row = dict(run_id=run.run_id, geometry=run.geometry, records=run.records,
                   outer_seed=int(run.outer_seed), requested_k=int(run.requested_k),
                   realised_k=int(run.returned_k), n=int(run.n), n_scored=int(run.n),
                   coverage=1.0, scoring_space=f'{run.geometry} input coordinates, Euclidean',
                   undefined_reason='')
        try:
            row.update(internal_scores(X, payload['label']))
            row.update(control_summaries(X, payload['label'], int(run.outer_seed)))
        except ValueError as error:
            row['undefined_reason'] = str(error)
        scores.append(row)
        if number % 10 == 0:
            print(f'scored {number}/{len(runs)}', flush=True)

    metrics = root / 'metrics'
    scores = pd.DataFrame(scores)
    objective = objective_against_kmeans(runs, payloads, spaces, identifiers)
    agreement = pd.concat([agreement_table(runs, payloads),
                           agreement_against_kmeans(runs, payloads)], ignore_index=True)
    summary = agreement_summary(agreement)
    for frame, name in [(scores, 'scores'), (objective, 'objective'), (agreement, 'agreement'),
                        (summary, 'agreement_summary')]:
        frame.to_csv(metrics / SCORE_TABLES[name], index=False)
    return scores, objective, agreement, summary


FIGURE_COLOURS = dict(procedure_selected='#4472a3', fixed_k_stability='#b33a3a',
                      raw1024='#4472a3', l2_raw1024='#b33a3a')
FIGURE_MARKERS = dict(procedure_selected='o', fixed_k_stability='^', full='o', deduplicated='s')

DATASET_ONLY = 'Phosphatase, 1,587 ProtT5 embeddings.'
DATASET_NOTE = ('Phosphatase, 1,587 ProtT5 embeddings; dedup is the 1,480 left after '
                'removing duplicate vectors.')


def figure_note(fig, text, height_inches):
    artist = fig.text(0.5, 0.010, text, ha='center', va='bottom', fontsize=7, color='#444444')
    fig.canvas.draw()
    drawn = artist.get_window_extent().width
    available = fig.get_figwidth() * fig.dpi
    if drawn > available - 8:
        longest = max(text.split('\n'), key=len)
        raise ValueError(f'Figure note is {drawn:.0f}px wide in a {available:.0f}px image. '
                         f'Split the longest line ({len(longest)} characters): {longest!r}')
    return (0.16 + 0.11 * text.count('\n')) / height_inches



def plot_stability(agreement, path):
    import matplotlib.pyplot as plt
    by_arm = [(('raw1024', 'full'), 'raw\nfull'), (('raw1024', 'deduplicated'), 'raw\ndedup'),
              (('l2_raw1024', 'full'), 'L2\nfull'), (('l2_raw1024', 'deduplicated'), 'L2\ndedup')]
    by_geometry = [(('raw1024', 'full_vs_deduplicated'), 'raw'),
                   (('l2_raw1024', 'full_vs_deduplicated'), 'L2')]
    by_records = [(('raw_vs_l2', 'full'), 'full'), (('raw_vs_l2', 'deduplicated'), 'dedup')]
    panels = [('procedure_selected_stability', 'seed only\n(each at its own k)', by_arm),
              ('fixed_k_stability', 'seed only\n(all at one shared k)', by_arm),
              ('full_vs_deduplicated', 'record set\n(full vs dedup)', by_geometry),
              ('raw_vs_l2', 'space\n(raw vs L2, own k)', by_records),
              ('raw_vs_l2_controlled', 'space\n(raw vs L2, k matched)', by_records)]
    fig, axes = plt.subplots(1, len(panels), figsize=(13, 4.5), sharey=True,
                             gridspec_kw=dict(width_ratios=[len(g) for _, _, g in panels]))
    for axis, (comparison, title, groups) in zip(axes, panels):
        rows = agreement[agreement.comparison == comparison]
        values = [rows[(rows.geometry == geometry) & (rows.records == records)]
                  .adjusted_rand.to_numpy() for (geometry, records), _ in groups]
        if sum(len(v) for v in values) != len(rows):
            raise ValueError(f'{comparison} rows do not split cleanly into its groups')
        axis.boxplot(values, tick_labels=[label for _, label in groups], widths=0.5,
                     showfliers=False, medianprops=dict(color='#222222'))
        for index, points in enumerate(values, start=1):
            spread = np.linspace(-0.12, 0.12, len(points)) if len(points) > 1 else [0.0]
            axis.scatter(index + np.asarray(spread), points, s=9, color='#4472a3', alpha=0.55,
                         edgecolors='none', zorder=3)
        axis.set_title(title, fontsize=8.5)
        axis.grid(axis='y', color='#e6e6e6', linewidth=0.6)
    axes[0].set_ylabel('Adjusted Rand index between partitions')
    fig.suptitle('How much a SubKMeans partition changes when one thing changes', fontsize=10)
    note = figure_note(fig, 'Adjusted Rand index: 1.0 is the same partition twice, 0.0 is chance '
                            'agreement. Each panel title names the one thing that differs within a '
                            'pair.\nBox is the median and quartiles; every pair is also drawn as a '
                            'dot, 45 per box in the two seed panels and 10 in the other three. '
                            'Nothing is pooled across panels.\n' + DATASET_NOTE, 4.5)
    fig.tight_layout(rect=(0, note, 1, 0.93))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_internal_scores(scores, path):
    import matplotlib.pyplot as plt
    metrics = [('silhouette', 'Silhouette'), ('calinski_harabasz', 'Calinski-Harabasz'),
               ('davies_bouldin', 'Davies-Bouldin')]
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.4))
    for row_index, geometry in enumerate(GEOMETRIES):
        for column, (metric, label) in enumerate(metrics):
            axis = axes[row_index][column]
            for position, records in enumerate(RECORD_SETS):
                rows = scores[(scores.geometry == geometry) & (scores.records == records)]
                spread = np.linspace(-0.2, 0.2, len(rows))
                axis.scatter(position + spread, rows[metric], s=14, color='#4472a3',
                             marker=FIGURE_MARKERS[records], edgecolors='none',
                             label={'full': 'full records',
                                    'deduplicated': 'deduplicated'}[records]
                             if (row_index, column) == (0, 0) else None, zorder=3)
            axis.set_xticks(range(len(RECORD_SETS)), ['full', 'dedup'])
            axis.set_xlim(-0.6, len(RECORD_SETS) - 0.4)
            axis.set_title(f"{label}, {'raw' if geometry == 'raw1024' else 'L2'}\n"
                           f"({'lower' if metric == 'davies_bouldin' else 'higher'} is better)",
                           fontsize=8.5)
            if column == 0:
                axis.set_xlabel('Record set', fontsize=8)
            axis.grid(axis='y', color='#e6e6e6', linewidth=0.6)
    axes[0][0].legend(fontsize=7, frameon=False, loc='best')
    fig.suptitle('The three internal measures, each in its own space', fontsize=10, y=0.985)
    note = figure_note(fig, 'One point per fit, 110 in all. Circles are the full 1,587 records and '
                            'squares the 1,480 deduplicated.\nThe y axes are not shared between '
                            'rows on purpose: raw and L2 are different spaces, so read down a '
                            'column, never across one.\n' + DATASET_NOTE, 6.4)
    fig.tight_layout(rect=(0, note, 1, 0.96))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_against_kmeans(agreement, path):
    import matplotlib.pyplot as plt
    import pandas as pd

    baseline = pd.read_csv(BASELINE / 'kmeans_standalone_partition_agreement.csv')
    floor = baseline[(baseline['comparison'] == 'procedure_selected_stability')
                     & (baseline['geometry'].isin(GEOMETRIES))]
    if len(floor) != len(GEOMETRIES) * len(RECORD_SETS) * 45:
        raise ValueError(f'expected 180 baseline seed-stability pairs, found {len(floor)}')
    groups = [('SubKMeans\nvs k-means',
               agreement[agreement['comparison'] == 'subkmeans_vs_kmeans']['adjusted_rand']),
              ('SubKMeans\nvs itself',
               agreement[agreement['comparison'] == 'procedure_selected_stability']['adjusted_rand']),
              ('k-means\nvs itself', floor['adjusted_rand'])]
    if any(len(values) == 0 for _, values in groups):
        raise ValueError('every box needs rows; one comparison returned none')

    fig, axis = plt.subplots(figsize=(6.6, 4.6))
    axis.boxplot([values.to_numpy() for _, values in groups],
                 tick_labels=[label for label, _ in groups], widths=0.5, showfliers=False,
                 medianprops=dict(color='#222222'))
    for index, (_, values) in enumerate(groups, start=1):
        points = values.to_numpy()
        spread = np.linspace(-0.13, 0.13, len(points)) if len(points) > 1 else [0.0]
        axis.scatter(index + np.asarray(spread), points, s=10, color='#4472a3', alpha=0.5,
                     edgecolors='none', zorder=3)
    axis.set_ylabel('Adjusted Rand index between partitions')
    axis.set_title('SubKMeans differs from k-means by about as much as k-means differs from itself',
                   fontsize=9)
    axis.grid(axis='y', color='#e6e6e6', linewidth=0.6)
    counts = ', '.join(f'{len(values)} pairs' for _, values in groups)
    note = figure_note(fig, 'Adjusted Rand index: 1.0 is the same partition twice, 0.0 is chance '
                            f'agreement. Left to right, {counts}.\nEvery pair is drawn as a dot. '
                            'Two runs of one method disagree because each reaches its own local '
                            'optimum,\nso the two right-hand boxes are the floor the left one is '
                            'read against. ' + DATASET_ONLY, 4.6)
    fig.tight_layout(rect=(0, note, 1, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_runtime(runs, path):
    import matplotlib.pyplot as plt
    fig, axis = plt.subplots(figsize=(7.0, 4.4))
    for geometry in GEOMETRIES:
        for records in RECORD_SETS:
            rows = runs[(runs.geometry == geometry) & (runs.records == records)]
            spread = np.linspace(-0.2, 0.2, len(rows))
            axis.scatter(rows.requested_k + spread, rows.wall_seconds, s=18,
                         marker=FIGURE_MARKERS[records], color=FIGURE_COLOURS[geometry],
                         edgecolors='white', linewidths=0.6,
                         label=f"{'raw' if geometry == 'raw1024' else 'L2'}, "
                               f"{'full' if records == 'full' else 'dedup'}")
    axis.set_xlabel('Requested number of clusters, k')
    axis.set_ylabel('Wall seconds per fit (10 restarts)')
    axis.set_title('What one SubKMeans fit cost, on one workstation', fontsize=9)
    axis.grid(color='#e6e6e6', linewidth=0.6)
    axis.legend(fontsize=7, frameon=False)
    note = figure_note(fig, 'One point per fit, 110 in all, run one at a time at four BLAS threads, '
                            'timed including saving and checks.\nPoints are spread sideways at each '
                            'k so they do not overlap; that spread carries no meaning.\nNone '
                            'failed. Not a benchmark against methods timed elsewhere. '
                            + DATASET_ONLY, 4.4)
    fig.tight_layout(rect=(0, note, 1, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_figures(root):
    import matplotlib
    matplotlib.use('Agg')
    matplotlib.rcParams['axes.axisbelow'] = True
    import pandas as pd
    root = Path(root)
    metrics, figures = root / 'metrics', root / 'figures'
    figures.mkdir(exist_ok=True)
    runs = pd.read_csv(metrics / GRID_TABLES['runs'])
    scores = pd.read_csv(metrics / SCORE_TABLES['scores'])
    agreement = pd.read_csv(metrics / SCORE_TABLES['agreement'])
    if not (len(runs) == len(scores) == len(set(runs.run_id))):
        raise ValueError('run and score tables do not describe the same fits')
    written = []
    for name, draw in [('against_kmeans', lambda p: plot_against_kmeans(agreement, p)),
                       ('stability', lambda p: plot_stability(agreement, p)),
                       ('internal_scores', lambda p: plot_internal_scores(scores, p)),
                       ('runtime', lambda p: plot_runtime(runs, p))]:
        path = figures / f'subkmeans_standalone_{name}.png'
        draw(path)
        written.append(str(path))
    return written


def parse_seeds(text):
    seeds = [int(value) for value in str(text).split(',') if value != '']
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError('Seeds must be a non-empty list of distinct integers')
    return seeds


def default_run_id(geometry, records, k, dtype, seeds):
    code = {'raw1024': 'RAW', 'l2_raw1024': 'L2R'}[geometry]
    return (f'SK-{code}-{records[0].upper()}-K{k}-{dtype[-2:]}'
            f'-S{seeds[0]:02d}' + (f'-N{len(seeds)}' if len(seeds) > 1 else ''))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--fit', action='store_true', help='one bounded fit')
    modes.add_argument('--manifest', action='store_true', help='freeze the run manifest')
    modes.add_argument('--validate-output', action='store_true', help='recheck a saved run')
    modes.add_argument('--scientific', action='store_true', help='run the frozen manifest')
    modes.add_argument('--score', action='store_true', help='score the saved grid partitions')
    modes.add_argument('--figures', action='store_true', help='draw figures from saved tables')
    modes.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--geometry', choices=GEOMETRIES, default='raw1024')
    parser.add_argument('--records', choices=RECORD_SETS, default='full')
    parser.add_argument('--k', type=int, default=15)
    parser.add_argument('--seeds', default='0', help='comma-separated restart seeds')
    parser.add_argument('--dtype', choices=['float32', 'float64'], default='float64')
    parser.add_argument('--purpose', choices=['diagnostic', 'scientific'], default='diagnostic')
    parser.add_argument('--run-id', default=None)
    parser.add_argument('--limit', type=int, default=None,
                        help='with --scientific, attempt at most this many unfinished rows')
    parser.add_argument('--jobs', type=int, default=1,
                        help='with --scientific, fits to run at once')
    parser.add_argument('--threads', type=int, default=THREADS,
                        help='BLAS threads per fit; jobs x threads may not exceed the CPU count')
    parser.add_argument('--output-root', type=Path, default=ROOT / 'results/subkmeans')
    args = parser.parse_args()
    seeds = parse_seeds(args.seeds)
    run_id = args.run_id or default_run_id(args.geometry, args.records, args.k, args.dtype, seeds)

    if args.worker:
        worker(args.output_root, run_id, args.geometry, args.records, args.k, seeds,
               args.dtype, args.purpose, args.threads)
    elif args.figures:
        print('\n'.join(write_figures(args.output_root)))
    elif args.score:
        scores, objective, agreement, summary = score_grid(args.output_root)
        print(json.dumps(dict(scored=len(scores), undefined=int((scores.undefined_reason != '').sum()),
                              objective_pairs=len(objective), agreement_pairs=len(agreement)),
                         indent=2))
    elif args.scientific:
        run_scientific(args.output_root, args.limit, args.jobs, args.threads)
    elif args.manifest:
        provenance = write_manifest(args.output_root)
        crosscheck = sequence_crosscheck(prepare_all()[1])
        write_json(Path(args.output_root) / 'manifest' / 'record_set_crosscheck.json', crosscheck)
        print(json.dumps({key: provenance[key] for key in
                          ['fits', 'role_counts', 'fits_by_group', 'estimator_calls', 'selected_k',
                           'modal_k', 'substituted_restart_seeds', 'record_sets']}, indent=2))
        print(json.dumps({key: crosscheck[key] for key in
                          ['fasta_records', 'fasta_unique_sequences', 'deduplicated_records',
                           'routes_agree']}, indent=2))
    elif args.validate_output:
        print(json.dumps(validate_saved(args.output_root, run_id), indent=2)[:2000])
    elif args.fit:
        paths = reserve_run(args.output_root, run_id)
        write_json(paths['logs'] / 'request.json', dict(
            run_id=run_id, purpose=args.purpose, geometry=args.geometry, records=args.records,
            requested_k=args.k, requested_seeds=seeds,
            effective_starts=[effective_start(seed) for seed in seeds], fit_dtype=args.dtype,
            settings=dict(FIXED_SETTINGS, n_clusters=args.k), source_commit=COMMIT,
            source_hashes=SOURCES, runner_sha256=digest(__file__),
            loader_sha256=digest(ROOT / 'src/data.py'),
            input_sha256={str(p.relative_to(ROOT)): digest(p) for p in DATASETS['phosphatase']},
            seconds_limit=TIME_LIMIT, memory_limit_bytes=MEMORY_LIMIT, threads=args.threads))
        result = bounded([sys.executable, str(Path(__file__).resolve()), '--worker',
                          '--run-id', run_id, '--geometry', args.geometry,
                          '--records', args.records, '--k', str(args.k),
                          '--seeds', ','.join(str(s) for s in seeds), '--dtype', args.dtype,
                          '--purpose', args.purpose, '--threads', str(args.threads),
                          '--output-root', str(args.output_root.resolve())],
                         paths['logs'] / 'resource.log', TIME_LIMIT, MEMORY_LIMIT, args.threads)
        if result['status'] != 'completed':
            raise RuntimeError(f"Bounded fit {result['status']}; evidence kept in {paths['logs']}")
        metadata = validate_saved(args.output_root, run_id, require_success=False)
        write_json(paths['metrics'] / 'success.json', {'archive_sha256': metadata['archive_sha256']})
        validate_saved(args.output_root, run_id)
        print(json.dumps({key: metadata[key] for key in RUN_FIELDS}, indent=2))
        print(json.dumps(result, indent=2))
    else:
        verify_source()
        geometries, ids = prepare(args.records)
        print(json.dumps(dict(
            mode='preflight_no_fit', records=args.records, n=len(ids),
            geometries={name: list(coordinates.shape) for name, coordinates in geometries.items()},
            dtype=str(geometries[args.geometry].dtype), planned_run_id=run_id,
            requested_seeds=seeds, effective_starts=[effective_start(s) for s in seeds],
            settings=dict(FIXED_SETTINGS, n_clusters=args.k)), indent=2))


if __name__ == '__main__':
    main()
