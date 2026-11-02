from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import time
import traceback

import numpy as np
import pandas as pd
import psutil
from threadpoolctl import threadpool_info

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from src.data import DATASETS, deduplicate, load_dataset

ROOT = REPO / 'results/autonr'
SOURCES = {
    'alternative/autonr.py': '21e021871ad0d902494bbecd5952001459e7407530291163039666b7b9c7ec79',
    'alternative/nrkmeans.py': 'f97143dba2cf6ca25a25a554567192c97a668176f1ae8e552531bea86763701e',
    'utils/_information_theory.py': '4267c915fe4887bd1b300ab1d56e519d383f3d41635346b1bfb98a0ff06ab6b3',
}
ORTHOGONAL_ATOL = 1e-6


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_json(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def verify_source():
    import clustpy
    from clustpy.alternative import AutoNR
    package = Path(clustpy.__file__).resolve().parent
    if importlib.metadata.version('clustpy') != '0.0.2':
        raise ValueError('Review the changed ClustPy version before fitting')
    for relative, expected in SOURCES.items():
        if digest(package / relative) != expected:
            raise ValueError(f'ClustPy source mismatch: {relative}')
    if Path(inspect.getfile(AutoNR)).resolve() != package / 'alternative/autonr.py':
        raise ValueError('AutoNR import is not the checked implementation')
    return AutoNR


def constructor(seed):
    return dict(nrkmeans_repetitions=15, outliers=True, max_subspaces=None,
                max_n_clusters=None, mdl_for_noisespace=True, max_distance=None,
                precision=None, similarity_threshold=1e-5, random_state=seed,
                debug=True)


def canonical_input(ids, X):
    ids, X = np.asarray(ids), np.asarray(X)
    if (ids.ndim != 1 or ids.dtype.kind not in 'US'
            or len(np.unique(ids)) != len(ids) or np.any(ids == '')):
        raise ValueError('Protein IDs must be unique nonempty strings')
    if (X.ndim != 2 or X.shape[0] != len(ids) or min(X.shape) < 2
            or X.dtype.kind not in 'fi' or not np.isfinite(X).all()):
        raise ValueError('Coordinates must be finite, numeric and aligned with IDs')
    order = np.argsort(ids, kind='stable')
    return ids[order], np.ascontiguousarray(X[order], dtype=np.float64)


GEOMETRIES = {'raw': None, 'l2': (None, True),
              'pca95': (0.95, False), 'pca90': (0.90, False),
              'pca80': (0.80, False), 'pca95_l2': (0.95, True),
              'pca90_l2': (0.90, True), 'pca80_l2': (0.80, True)}


def pca(X, variance):
    from sklearn.decomposition import PCA
    fitted = PCA(n_components=variance, svd_solver='full', whiten=False).fit(X)
    return np.ascontiguousarray(fitted.transform(X)), dict(
        pca_components=fitted.components_, pca_mean=fitted.mean_,
        pca_explained_variance_ratio=fitted.explained_variance_ratio_,
        pca_explained_variance=fitted.explained_variance_,
        pca_singular_values=fitted.singular_values_,
        pca_fit_n=np.array(X.shape[0]))


def unit_rows(X):
    from sklearn.preprocessing import normalize
    if np.any(np.linalg.norm(X, axis=1) == 0):
        raise ValueError('Zero-length row cannot be L2-normalised')
    X = np.ascontiguousarray(normalize(X, norm='l2'))
    np.testing.assert_allclose(np.linalg.norm(X, axis=1), 1, rtol=0, atol=1e-12)
    return X


def prepare(records, geometry='raw', transform_out=None):
    ds = load_dataset('phosphatase')
    stored_dtype = str(ds.X.dtype)
    ids, X = canonical_input(ds.ids, ds.X)
    if geometry not in GEOMETRIES:
        raise ValueError('Unknown input geometry')
    if GEOMETRIES[geometry] is not None:
        variance, l2 = GEOMETRIES[geometry]
        if variance is not None:
            X, basis = pca(X, variance)
            if transform_out is not None:
                transform_out.update(basis)
        if l2:
            X = unit_rows(X)
    if records == 'dedup':
        mask = np.isin(ids, deduplicate(ds).ids)
        ids, X = ids[mask], np.ascontiguousarray(X[mask])
    return ids, X, stored_dtype


def input_signature(ids, X):
    return dict(n=len(ids), d=X.shape[1], fit_dtype=str(X.dtype),
                subset_hash=hashlib.sha256('\n'.join(ids).encode()).hexdigest(),
                coordinates_sha256=hashlib.sha256(X.tobytes()).hexdigest())


def coding_quantities(X):
    from clustpy.alternative.autonr import _check_input_parameters
    bounds = _check_input_parameters(X, 15, None, None, None, None)
    if not np.isfinite(bounds[2:]).all() or min(bounds[2:]) <= 0:
        raise ValueError('Native distance scale and precision must be finite and positive')
    return dict(max_subspaces=int(bounds[0]), max_n_clusters=int(bounds[1]),
                max_distance=float(bounds[2]), precision=float(bounds[3]))


def transform_record(geometry):
    if GEOMETRIES[geometry] is None:
        return 'none'
    variance, l2 = GEOMETRIES[geometry]
    if variance is None:
        return dict(method='row-wise L2 normalisation of raw input',
                    normalisation='sklearn.preprocessing.normalize, norm=l2',
                    fit_population='none: each row is scaled on its own')
    record = dict(method='PCA', variance_target=variance, solver='full', whiten=False,
                  fit_population='all canonical phosphatase records')
    if l2:
        record.update(method='PCA then row-wise L2 normalisation',
                      normalisation='sklearn.preprocessing.normalize, norm=l2, after PCA',
                      variance_note='variance_target describes PCA before normalisation')
    return record


def provenance(ids, X, stored_dtype, records, seed, geometry='raw'):
    verify_source()
    try:
        git_sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True,
                                          stderr=subprocess.DEVNULL).strip()
        dirty = subprocess.check_output(['git', 'status', '--porcelain'], cwd=REPO, text=True,
                                        stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        git_sha, dirty = 'unknown', ''
    return dict(input=input_signature(ids, X), records=records, geometry=geometry,
                stored_dtype=stored_dtype,
                transform=transform_record(geometry),
                constructor=constructor(seed),
                resolved=coding_quantities(X), git_sha=git_sha + ('-dirty' if dirty else ''),
                runner_sha256=digest(__file__), data_helper_sha256=digest(REPO / 'src/data.py'),
                input_files={str(p.relative_to(REPO)): digest(p)
                             for p in DATASETS['phosphatase'] if p is not None},
                library_sources=SOURCES, python=sys.version, platform=platform.platform(),
                versions={name: importlib.metadata.version(name) for name in
                          ('clustpy', 'numpy', 'scipy', 'scikit-learn', 'pandas', 'psutil')},
                threadpools=threadpool_info(), thread_environment={k: os.environ.get(k) for k in
                    ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS')},
                inner_defaults=dict(max_iter=300, n_init=1, cost_type='default',
                                    threshold_negative_eigenvalue=-1e-7),
                monitoring='psutil sampled process-tree RSS/CPU and available system RAM; no hard cap')


def fit_arrays(ids, X, seed):
    AutoNR = verify_source()
    model = AutoNR(**constructor(seed)).fit(X)
    nr = model.nrkmeans_
    arrays = dict(protein_id=ids, label=model.labels_, V=nr.V,
                  m=np.asarray(nr.m), n_clusters=np.asarray(model.n_clusters_))
    for j in range(len(nr.m)):
        arrays[f'P_{j}'] = np.asarray(nr.P[j])
        arrays[f'centres_{j}'] = nr.cluster_centers[j]
        arrays[f'scatter_{j}'] = nr.scatter_matrices_[j]
        np.testing.assert_allclose(X @ nr.V[:, nr.P[j]], nr.transform_subspace(X, j),
                                   rtol=0, atol=ORTHOGONAL_ATOL)
    total, global_cost, subspace_costs = nr.calculate_mdl_costs(X)
    if not np.isfinite([total, global_cost, *subspace_costs]).all():
        raise ValueError('Nonfinite final MDL')
    np.testing.assert_allclose(total, model.mdl_costs_, rtol=1e-10, atol=1e-6)
    history = [dict(index=i, costs=float(h.costs), full_space=bool(h.full_space_execution),
                    operation=h.originates_from_operation) for i, h in enumerate(model.all_mdl_costs_)]
    info = dict(mdl=float(total), global_mdl=float(global_cost),
                subspace_mdl=[float(c) for c in subspace_costs], history=history,
                outer_termination='unknown: release does not expose a stop reason',
                inner_termination='unknown: release does not expose iterations or convergence')
    validate_arrays(arrays, len(ids), X.shape[1])
    return arrays, info


def validate_arrays(a, n, d):
    ids, labels, V, m, counts = (a[k] for k in ('protein_id', 'label', 'V', 'm', 'n_clusters'))
    if ids.shape != (n,) or ids.dtype.kind not in 'US' or len(np.unique(ids)) != n:
        raise ValueError('Invalid saved protein IDs')
    if not np.array_equal(ids, np.sort(ids)):
        raise ValueError('Saved IDs are not canonical')
    if m.ndim != 1 or m.dtype.kind not in 'iu' or len(m) == 0 or np.any(m < 1):
        raise ValueError('Invalid subspace dimensions')
    s = len(m)
    if counts.shape != (s,) or counts.dtype.kind not in 'iu' or np.any(counts < 1):
        raise ValueError('Invalid cluster counts')
    if labels.shape != (n, s) or labels.dtype.kind not in 'iu':
        raise ValueError('Labels must retain one integer column per subspace')
    if V.shape != (d, d) or not np.isfinite(V).all():
        raise ValueError('Invalid rotation')
    np.testing.assert_allclose(V.T @ V, np.eye(d), rtol=0, atol=ORTHOGONAL_ATOL)
    projections = []
    for j, (width, k) in enumerate(zip(m, counts)):
        P, centres, scatter = (a[f'{key}_{j}'] for key in ('P', 'centres', 'scatter'))
        if P.shape != (width,) or P.dtype.kind not in 'iu':
            raise ValueError('Projection and dimension disagree')
        projections.extend(P.tolist())
        if (centres.shape != (k, d) or scatter.shape != (d, d)
                or not np.isfinite(centres).all() or not np.isfinite(scatter).all()):
            raise ValueError('Invalid centres/scatter')
        column = labels[:, j]
        if np.any(column < -1) or not np.array_equal(np.unique(column[column >= 0]), np.arange(k)):
            raise ValueError('Labels and realised cluster count disagree')
    if sorted(projections) != list(range(d)) or int(sum(m)) != d:
        raise ValueError('Projections must allocate each coordinate exactly once')
    if any(v.dtype.hasobject for v in a.values()):
        raise ValueError('Object arrays are not allowed')


def save_tables(path, a, info, configuration_id):
    spaces, clusters = [], []
    for j, (width, k) in enumerate(zip(a['m'], a['n_clusters'])):
        labels = a['label'][:, j]
        spaces.append(dict(configuration_id=configuration_id, subspace=j, dimensions=int(width),
                           n_clusters=int(k), noise_space=bool(k == 1), n=len(labels),
                           n_outliers=int(np.sum(labels == -1)), mdl=info['subspace_mdl'][j]))
        for label, count in zip(*np.unique(labels, return_counts=True)):
            clusters.append(dict(configuration_id=configuration_id, subspace=j,
                                 label=int(label), is_outlier=bool(label == -1), n=int(count)))
    pd.DataFrame(spaces).to_csv(path / 'subspaces.csv', index=False)
    pd.DataFrame(clusters).to_csv(path / 'clusters.csv', index=False)
    pd.DataFrame(info['history']).to_csv(path / 'search.csv', index=False)


def validate_saved(path, require_success=True):
    config = json.loads((path / 'configuration.json').read_text())
    result = json.loads((path / 'result.json').read_text())
    if digest(path / 'model.npz') != result['archive_sha256']:
        raise ValueError('Saved archive hash mismatch')
    if digest(path / 'configuration.json') != result['configuration_id']:
        raise ValueError('Configuration hash mismatch')
    with np.load(path / 'model.npz', allow_pickle=False) as archive:
        a = {k: archive[k] for k in archive.files}
    sig = config['input']
    validate_arrays(a, sig['n'], sig['d'])
    basis = {}
    ids, X, _ = prepare(config['records'], config.get('geometry', 'raw'), basis)
    if input_signature(ids, X) != sig or not np.array_equal(ids, a['protein_id']):
        raise ValueError('Saved IDs/coordinates differ from current input')
    for name, expected in basis.items():
        np.testing.assert_array_equal(a[name], expected)
    for name, expected in result['table_hashes'].items():
        if digest(path / name) != expected:
            raise ValueError(f'Table hash mismatch: {name}')
    if require_success:
        success = json.loads((path / 'SUCCESS.json').read_text())
        if success['result_sha256'] != digest(path / 'result.json'):
            raise ValueError('Success marker does not match result')
    return a, result


def worker(path):
    try:
        config = json.loads((path / 'configuration.json').read_text())
        if config['runner_sha256'] != digest(__file__) or config['data_helper_sha256'] != digest(REPO / 'src/data.py'):
            raise ValueError('Code changed after preflight')
        basis = {}
        ids, X, _ = prepare(config['records'], config.get('geometry', 'raw'), basis)
        if input_signature(ids, X) != config['input']:
            raise ValueError('Input changed after preflight')
        started = time.monotonic()
        arrays, info = fit_arrays(ids, X, config['constructor']['random_state'])
        info['fit_seconds'] = time.monotonic() - started
        arrays.update(basis)
        if basis:
            info['pca_retained_variance'] = float(np.sum(basis['pca_explained_variance_ratio']))
        np.savez_compressed(path / 'model.npz', **arrays)
        info['archive_sha256'] = digest(path / 'model.npz')
        info['configuration_id'] = digest(path / 'configuration.json')
        save_tables(path, arrays, info, info['configuration_id'])
        pd.DataFrame([dict(configuration_id=info['configuration_id'], **config['input'],
                           seed=config['constructor']['random_state'], mdl=info['mdl'],
                           fit_seconds=info['fit_seconds'], subspaces=len(arrays['m']),
                           git_sha=config['git_sha'], outer_termination=info['outer_termination'],
                           inner_termination=info['inner_termination'])]).to_csv(path / 'run.csv', index=False)
        info['table_hashes'] = {name: digest(path / name) for name in
                                ('subspaces.csv', 'clusters.csv', 'search.csv', 'run.csv')}
        write_json(path / 'result.json', info)
        validate_saved(path, require_success=False)
    except BaseException:
        (path / 'failure.txt').write_text(traceback.format_exc(), encoding='utf-8')
        raise


def stop_tree(pid):
    try:
        parent = psutil.Process(pid)
        processes = parent.children(recursive=True) + [parent]
    except psutil.NoSuchProcess:
        return
    for process in reversed(processes):
        try:
            process.terminate()
        except psutil.NoSuchProcess:
            pass
    _, alive = psutil.wait_procs(processes, timeout=5)
    for process in alive:
        try:
            process.kill()
        except psutil.NoSuchProcess:
            pass


def run(path, records, seed, geometry='raw'):
    import zipfile
    archived = path.parent.parent / 'archive/local_pca_evidence.zip'
    if archived.exists():
        with zipfile.ZipFile(archived) as saved:
            if f'results/autonr/attempts/{path.name}/configuration.json' in saved.namelist():
                raise FileExistsError(f'Run ID is already preserved in {archived}: {path.name}')
    path.mkdir(parents=True, exist_ok=False)
    ids, X, stored = prepare(records, geometry)
    write_json(path / 'configuration.json', provenance(ids, X, stored, records, seed, geometry))
    (path / 'runner_snapshot.py').write_bytes(Path(__file__).read_bytes())
    started = time.monotonic()
    cancelled = False
    with (path / 'stdout.log').open('x') as stdout, (path / 'stderr.log').open('x') as stderr:
        process = subprocess.Popen([sys.executable, '-u', str(Path(__file__).resolve()),
                                    '_worker', '--attempt', str(path.resolve())],
                                   stdout=stdout, stderr=stderr, cwd=REPO)
        write_json(path / 'process.json', dict(pid=process.pid, started_unix=time.time()))
        try:
            with (path / 'resources.csv').open('x', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=['elapsed_seconds', 'rss_bytes',
                    'cpu_seconds', 'available_system_bytes'])
                writer.writeheader()
                while process.poll() is None:
                    if (path / 'CANCEL').exists():
                        cancelled = True
                        stop_tree(process.pid)
                        break
                    rss, cpu = 0, 0.0
                    try:
                        parent = psutil.Process(process.pid)
                        for p in [parent] + parent.children(recursive=True):
                            try:
                                rss += p.memory_info().rss
                                times = p.cpu_times()
                                cpu += times.user + times.system
                            except psutil.NoSuchProcess:
                                pass
                    except psutil.NoSuchProcess:
                        pass
                    writer.writerow(dict(elapsed_seconds=time.monotonic() - started,
                                         rss_bytes=rss, cpu_seconds=cpu,
                                         available_system_bytes=psutil.virtual_memory().available))
                    stream.flush()
                    time.sleep(1)
        except BaseException:
            cancelled = True
            stop_tree(process.pid)
            raise
        finally:
            code = process.wait()
            write_json(path / 'process_outcome.json', dict(returncode=code, cancelled=cancelled,
                       elapsed_seconds=time.monotonic() - started))
    if cancelled or code != 0:
        raise RuntimeError(f'Attempt did not complete: {path}; inspect stderr.log and failure.txt')
    validate_saved(path, require_success=False)
    write_json(path / 'SUCCESS.json', dict(result_sha256=digest(path / 'result.json')))
    print(f'Validated result: {path}')


def figures(path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    validate_saved(path)
    table = pd.read_csv(path / 'clusters.csv')
    target = path / 'figures'
    target.mkdir(exist_ok=False)
    for subspace, rows in table.groupby('subspace'):
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(rows['label'].astype(str), rows['n'])
        ax.set(xlabel='Cluster label (-1 = outlier in this subspace)', ylabel='Proteins',
               title=f'AutoNR subspace {subspace}: all retained records')
        fig.tight_layout()
        fig.savefig(target / f'cluster_sizes_{subspace}.png', dpi=150)
        plt.close(fig)


def internal_scores(X, labels):
    from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
    keep = labels >= 0
    n, k = int(keep.sum()), len(np.unique(labels[keep]))
    row = dict(n_scored=n, n_outliers=int((~keep).sum()), n_clusters=k,
               silhouette=None, calinski_harabasz=None, davies_bouldin=None,
               score_status='undefined: requires 2 <= clusters < scored records')
    if 2 <= k < n:
        coords, partition = X[keep], labels[keep]
        row.update(silhouette=float(silhouette_score(coords, partition, metric='euclidean')),
                   calinski_harabasz=float(calinski_harabasz_score(coords, partition)),
                   davies_bouldin=float(davies_bouldin_score(coords, partition)), score_status='defined')
    return row


def analyse(path, destination):
    config = json.loads((path / 'configuration.json').read_text())
    geometry = config.get('geometry', 'raw')
    a, result = validate_saved(path)
    ids, X, _ = prepare(config['records'], geometry)
    rows = []
    for j, width in enumerate(a['m']):
        rows.append(dict(configuration_id=result['configuration_id'], subspace=j,
                         input_geometry=geometry, input_dimensions=X.shape[1],
                         learned_dimensions=int(width), n_total=len(ids),
                         **internal_scores(X, a['label'][:, j])))
    destination.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(rows).to_csv(destination / 'internal_scores.csv', index=False)
    table = pd.read_csv(path / 'clusters.csv')
    table.to_csv(destination / 'cluster_sizes.csv', index=False)
    write_json(destination / 'analysis.json', dict(
        source_attempt=str(path.resolve()), source_result_sha256=digest(path / 'result.json'),
        analysis_code_sha256=digest(__file__), records=config['records'],
        scoring_geometry=f'{geometry} input; Euclidean silhouette; per-space outliers excluded',
        versions={name: importlib.metadata.version(name) for name in ('numpy', 'scikit-learn', 'matplotlib')},
        limitations='One seed; no stability or biological claim. Different masks limit direct score comparisons.'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    nrows = (len(rows) + 2) // 3
    fig, axes = plt.subplots(nrows, 3, figsize=(15, 3.5 * nrows), squeeze=False)
    for ax, (j, group) in zip(axes.flat, table.groupby('subspace')):
        bars = ax.bar(group.label.astype(str), group.n)
        ax.bar_label(bars, fontsize=8, padding=2)
        ax.margins(y=.18)
        ax.set(xlabel='Label (-1 = outlier)', ylabel='Proteins',
               title=f'Space {j}: {a["m"][j]} dimensions, k={a["n_clusters"][j]}')
    for ax in list(axes.flat)[len(rows):]:
        ax.set_visible(False)
    fig.suptitle(f'AutoNR {geometry}, {config["records"]}, seed {config["constructor"]["random_state"]}: all {len(ids)} proteins')
    fig.tight_layout(rect=(0, 0, 1, .97))
    fig.savefig(destination / 'cluster_sizes.png', dpi=150)
    plt.close(fig)
    write_record(path, destination, config, result)
    print(pd.DataFrame(rows).to_string(index=False))


def write_record(path, destination, config, result):
    snapshot = path / 'runner_snapshot.py'
    if digest(snapshot) != config['runner_sha256']:
        raise ValueError('Runner snapshot does not match the configuration')
    if (path / 'record.json').exists():
        existing = json.loads((path / 'record.json').read_text())
        for key, expected in dict(configuration_sha256=result['configuration_id'],
                                  model_sha256=result['archive_sha256'],
                                  source_sha256=config['runner_sha256']).items():
            if existing[key] != expected:
                raise ValueError(f'Existing record mismatch: {key}')
        return
    write_json(path / 'record.json', dict(
        schema_version=1, run_id=path.name, original_run_id=path.name,
        status='validated_and_analysed', geometry=config.get('geometry', 'raw'),
        records=config['records'], seed=config['constructor']['random_state'],
        platform='local Windows', input=config['input'],
        fit_directory=path.resolve().relative_to(REPO).as_posix(),
        analysis_directory=destination.resolve().relative_to(REPO).as_posix(),
        provenance_directory=None, configuration_sha256=result['configuration_id'],
        model_sha256=result['archive_sha256'], source_snapshot=snapshot.name,
        source_sha256=config['runner_sha256'],
        note='Written by the runner after local validation and analysis.'))


def subspace_agreement(left, right):
    from sklearn.metrics import adjusted_rand_score
    shared, li, ri = np.intersect1d(left['protein_id'], right['protein_id'], return_indices=True)
    rows = []
    for i, lk in enumerate(left['n_clusters']):
        for j, rk in enumerate(right['n_clusters']):
            x, y = left['label'][li, i], right['label'][ri, j]
            keep = (x >= 0) & (y >= 0)
            nx, ny = len(np.unique(x[keep])), len(np.unique(y[keep]))
            status = 'defined'
            if lk == 1 or rk == 1:
                status = 'not scored: noise space'
            elif keep.sum() < 2 or nx < 2 or ny < 2:
                status = 'not scored: insufficient shared clusters'
            rows.append(dict(left_subspace=i, right_subspace=j, shared_ids=len(shared),
                             shared_non_outliers=int(keep.sum()), left_shared_clusters=nx,
                             right_shared_clusters=ny, status=status,
                             ari=float(adjusted_rand_score(x[keep], y[keep])) if status == 'defined' else None))
    return pd.DataFrame(rows)


def compare_records(left_path, right_path, destination):
    left, _ = validate_saved(left_path)
    right, _ = validate_saved(right_path)
    lc = json.loads((left_path / 'configuration.json').read_text())
    rc = json.loads((right_path / 'configuration.json').read_text())
    if lc['records'] != 'full' or rc['records'] != 'dedup':
        raise ValueError('Supply full as the left attempt and dedup as the right')
    if lc.get('geometry', 'raw') != rc.get('geometry', 'raw') or lc['constructor'] != rc['constructor']:
        raise ValueError('Pair requires identical geometry and constructor settings')
    if lc['versions'] != rc['versions'] or lc['library_sources'] != rc['library_sources']:
        raise ValueError('Review differing library environments before comparison')
    if not np.isin(right['protein_id'], left['protein_id']).all():
        raise ValueError('Dedup IDs are not a subset')
    for key in left:
        if key.startswith('pca_'):
            np.testing.assert_array_equal(left[key], right[key])
    geometry = lc.get('geometry', 'raw')
    ids, X, _ = prepare('full', geometry)
    dedup_ids, dedup_X, _ = prepare('dedup', geometry)
    np.testing.assert_array_equal(dedup_X, X[np.isin(ids, dedup_ids)])
    scores = subspace_agreement(left, right)
    destination.mkdir(parents=True, exist_ok=False)
    scores.to_csv(destination / 'subspace_ari.csv', index=False)
    write_json(destination / 'comparison.json', dict(
        left=str(left_path.resolve()), right=str(right_path.resolve()),
        left_result_sha256=digest(left_path / 'result.json'), right_result_sha256=digest(right_path / 'result.json'),
        analysis_code_sha256=digest(__file__), sklearn_version=importlib.metadata.version('scikit-learn'),
        identical_saved_pca_basis=True if 'pca_components' in left else None,
        input_geometry=geometry, shared_ids=len(dedup_ids),
        interpretation=f'Seed-{lc["constructor"]["random_state"]} whole-procedure record sensitivity; no cross-seed stability claim.',
        noise_policy='Noise-space pairs and constant overlap partitions are left unscored.'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    clustered = scores[(scores.left_subspace.isin(np.flatnonzero(left['n_clusters'] > 1))) &
                       (scores.right_subspace.isin(np.flatnonzero(right['n_clusters'] > 1)))]
    if len(clustered):
        fig, axes = plt.subplots(2, 1, figsize=(14, 13))
        for ax, value, title in zip(axes, ['ari', 'shared_non_outliers'],
                ['Adjusted Rand index (higher = more agreement)', 'Shared non-outlier proteins per pair']):
            matrix = clustered.pivot(index='left_subspace', columns='right_subspace', values=value)
            kwargs = dict(vmin=-1, vmax=1, cmap='coolwarm') if value == 'ari' else dict(cmap='viridis')
            im = ax.imshow(matrix.to_numpy(), aspect='auto', **kwargs)
            for i in range(matrix.shape[0]):
                for j in range(matrix.shape[1]):
                    v = matrix.iloc[i, j]
                    ax.text(j, i, f'{v:.2f}' if value == 'ari' else f'{v:.0f}', ha='center', va='center', fontsize=8,
                            bbox=dict(facecolor='white', alpha=.65, edgecolor='none', pad=.2))
            ax.set(xticks=range(len(matrix.columns)), xticklabels=matrix.columns,
                   yticks=range(len(matrix.index)), yticklabels=matrix.index,
                   xlabel='Dedup subspace (independent numbering)', ylabel='Full subspace', title=title)
            fig.colorbar(im, ax=ax)
        fig.suptitle(f'AutoNR {geometry}: full vs dedup, seed {lc["constructor"]["random_state"]}; noise spaces omitted')
        fig.tight_layout(rect=(0, 0, 1, .97))
        fig.savefig(destination / 'subspace_agreement.png', dpi=150)
        plt.close(fig)
    print(f'Wrote {len(scores)} subspace pairs to {destination}')


def evidence_bytes(root, relative):
    import zipfile
    path = root / relative
    if path.is_file():
        return path.read_bytes()
    with zipfile.ZipFile(root / 'archive/local_pca_evidence.zip') as archive:
        return archive.read('results/autonr/' + relative)


def seed_variation_figure(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    table = pd.read_csv(root / 'metrics/local_pca_runs.csv')
    arms = sorted(set(zip(table.geometry, table.records)))
    fig, axes = plt.subplots(1, 3, figsize=(15, 7), sharey=True)
    for y, (g, rec) in enumerate(arms):
        rows = table[(table.geometry == g) & (table.records == rec)].sort_values('seed')
        pos = y + (rows.seed.to_numpy() - 4.5) * .065
        values = [rows.clustered_subspaces, 100 * rows.any_outlier_count / rows.n, rows.fit_seconds / 60]
        for ax, value in zip(axes, values):
            points = ax.scatter(value, pos, c=rows.seed, vmin=0, vmax=9, cmap='viridis', s=24)
    for ax, label in zip(axes, ['Clustered subspaces', 'Outlier in any space (% of records)', 'Elapsed fit time (minutes, log scale)']):
        ax.set_xlabel(label)
        ax.grid(axis='x', alpha=.25)
    axes[0].set_yticks(range(len(arms)), [f'{g}, {r}' for g, r in arms])
    axes[0].invert_yaxis()
    axes[2].set_xscale('log')
    fig.suptitle('AutoNR local PCA: all ten seeds in each arm')
    fig.text(.20, .025, 'Outliers are per space; the middle panel counts distinct proteins. Seed 0 ran separately; later seeds used two concurrent fits.\nPCA95 seed 5 timings include about 4.3-hour gaps in resource sampling; cause unverified. No elapsed-time correction was applied.', fontsize=9)
    fig.subplots_adjust(left=.20, right=.93, bottom=.14, top=.91, wspace=.15)
    fig.colorbar(points, ax=axes, label='Seed', ticks=range(10), fraction=.018, pad=.02)
    (root / 'figures').mkdir(exist_ok=True)
    fig.savefig(root / 'figures/local_pca_seed_variation.png', dpi=160)
    plt.close(fig)


def summarise(root):
    import io
    import itertools
    manifest = json.loads((root / 'provenance/local_pca_batch/manifest.json').read_text())
    specs = [dict(r, seed=0) for r in manifest['references']] + manifest['jobs']
    expected = {(g, r, s) for g in GEOMETRIES if g.startswith('pca')
                for r in ('full', 'dedup') for s in range(10)}
    if len(specs) != 120 or {(r['geometry'], r['records'], r['seed']) for r in specs} != expected:
        raise ValueError('Batch does not contain the declared 120 distinct configurations')
    metrics, assignments = root / 'metrics', root / 'assignments/local_pca_seeds'
    metrics.mkdir(exist_ok=True)
    assignments.mkdir(parents=True, exist_ok=True)
    frames = {k: [] for k in ('runs', 'subspaces', 'clusters', 'internal_scores')}
    models, inputs, configs, checks = {}, {}, {}, []
    sha = lambda value: hashlib.sha256(value).hexdigest()
    for spec in sorted(specs, key=lambda r: (r['geometry'], r['records'], r['seed'])):
        ident, g, rec, seed = spec['id'], spec['geometry'], spec['records'], spec['seed']
        read = lambda name: evidence_bytes(root, f'attempts/{ident}/{name}')
        config, result = (json.loads(read(f'{name}.json')) for name in ('configuration', 'result'))
        assert config.get('geometry') == g and config['records'] == rec
        assert config['constructor'] == dict(manifest['constructor'], random_state=seed)
        assert config['library_sources'] == manifest['library_sources']
        ref = next(r for r in manifest['references'] if (r['geometry'], r['records']) == (g, rec))
        assert config['versions'] == ref['versions'] and config['input'] == ref['input']
        assert sha(read('configuration.json')) == result['configuration_id']
        assert sha(read('model.npz')) == result['archive_sha256']
        assert sha(read('runner_snapshot.py')) == config['runner_sha256']
        assert json.loads(read('SUCCESS.json'))['result_sha256'] == sha(read('result.json'))
        for name, value in result['table_hashes'].items():
            assert sha(read(name)) == value, (ident, name)
        outcome = json.loads(read('process_outcome.json'))
        assert outcome['returncode'] == 0 and not outcome['cancelled']
        analysis = json.loads(evidence_bytes(root, f'analysis/{ident}/analysis.json'))
        assert analysis['source_result_sha256'] == sha(read('result.json'))
        with np.load(io.BytesIO(read('model.npz')), allow_pickle=False) as saved:
            a = {key: saved[key] for key in saved.files}
        validate_arrays(a, config['input']['n'], config['input']['d'])
        if (g, rec) not in inputs:
            basis = {}
            ids, X, _ = prepare(rec, g, basis)
            inputs[g, rec] = ids, X, basis
        ids, X, basis = inputs[g, rec]
        assert input_signature(ids, X) == config['input']
        np.testing.assert_array_equal(ids, a['protein_id'])
        for name, value in basis.items():
            np.testing.assert_array_equal(a[name], value)
        models[g, rec, seed] = a
        configs[g, rec, seed] = ident
        metadata = dict(run_id=ident, geometry=g, records=rec, seed=seed)
        for name in frames:
            relative = f'analysis/{ident}/internal_scores.csv' if name == 'internal_scores' else f'attempts/{ident}/{"run" if name == "runs" else name}.csv'
            frame = pd.read_csv(io.BytesIO(evidence_bytes(root, relative)))
            for key, value in metadata.items():
                frame[key] = value
            if name == 'runs':
                assert len(frame) == 1 and np.isfinite(frame.mdl.iloc[0])
                frame['clustered_subspaces'] = int((a['n_clusters'] > 1).sum())
                frame['noise_subspaces'] = int((a['n_clusters'] == 1).sum())
                frame['any_outlier_count'] = int((a['label'] == -1).any(axis=1).sum())
                frame['stderr_bytes'] = len(read('stderr.log'))
                resources = pd.read_csv(io.BytesIO(read('resources.csv')))
                frame['peak_sampled_rss_bytes'] = resources.rss_bytes.max()
                frame['max_resource_sample_gap_seconds'] = resources.elapsed_seconds.diff().max()
            elif name == 'subspaces':
                np.testing.assert_array_equal(frame.n_clusters, a['n_clusters'])
                np.testing.assert_array_equal(frame.dimensions, a['m'])
                np.testing.assert_array_equal(frame.n_outliers, (a['label'] == -1).sum(axis=0))
            elif name == 'clusters':
                for j, rows in frame.groupby('subspace'):
                    labels, sizes = np.unique(a['label'][:, j], return_counts=True)
                    np.testing.assert_array_equal(rows.label, labels)
                    np.testing.assert_array_equal(rows.n, sizes)
            else:
                np.testing.assert_array_equal(frame.n_outliers, (a['label'] == -1).sum(axis=0))
                np.testing.assert_array_equal(frame.n_scored, (a['label'] >= 0).sum(axis=0))
                np.testing.assert_array_equal(frame.n_clusters, a['n_clusters'])
            frames[name].append(frame)
        target = assignments / f'{ident}.npz'
        np.savez_compressed(target, **{k: a[k] for k in ('protein_id', 'label', 'm', 'n_clusters')})
        with np.load(target, allow_pickle=False) as saved:
            for key in saved.files:
                np.testing.assert_array_equal(saved[key], a[key])
        checks.append(dict(**metadata, model_sha256=result['archive_sha256'],
                           configuration_sha256=result['configuration_id'], checked=True))
    for g in {s['geometry'] for s in specs}:
        ids, X, basis = inputs[g, 'full']
        di, dx, db = inputs[g, 'dedup']
        assert np.isin(di, ids).all()
        np.testing.assert_array_equal(dx, X[np.isin(ids, di)])
        for key in basis:
            np.testing.assert_array_equal(basis[key], db[key])
    for name, chunks in frames.items():
        frames[name] = pd.concat(chunks, ignore_index=True)
        frames[name].to_csv(metrics / f'local_pca_{name}.csv', index=False)
    summary = frames['runs'].groupby(['geometry', 'records']).agg(
        seeds=('seed', 'count'), clustered_min=('clustered_subspaces', 'min'),
        clustered_max=('clustered_subspaces', 'max'), clustered_mean=('clustered_subspaces', 'mean'),
        clustered_sd=('clustered_subspaces', 'std'), outliers_min=('any_outlier_count', 'min'),
        outliers_max=('any_outlier_count', 'max'), fit_seconds_min=('fit_seconds', 'min'),
        fit_seconds_median=('fit_seconds', 'median'), fit_seconds_max=('fit_seconds', 'max'))
    summary.to_csv(metrics / 'local_pca_summary.csv')
    print('Validated all 120 PCA fits and partition exports.', flush=True)
    pair_counts = {}
    for kind in ('seed', 'records'):
        chunks = []
        for g in sorted({s['geometry'] for s in specs}):
            pairs = (((g, rec, a), (g, rec, b)) for rec in ('full', 'dedup')
                     for a, b in itertools.combinations(range(10), 2)) if kind == 'seed' else (
                     ((g, 'full', s), (g, 'dedup', s)) for s in range(10))
            for left, right in pairs:
                frame = subspace_agreement(models[left], models[right])
                frame['geometry'] = g
                frame['left_run_id'], frame['right_run_id'] = configs[left], configs[right]
                frame['left_seed'], frame['right_seed'] = left[2], right[2]
                frame['left_records'], frame['right_records'] = left[1], right[1]
                chunks.append(frame)
            print(f'{kind} agreement: {g}', flush=True)
        table = pd.concat(chunks, ignore_index=True)
        table.to_csv(metrics / f'local_pca_{kind}_ari.csv', index=False)
        pair_counts[kind] = dict(run_pairs=len(chunks), subspace_pairs=len(table),
                                defined=int((table.status == 'defined').sum()))
    audit = dict(date='2026-09-29', validated_runs=len(checks), checks=checks,
                 comparisons=pair_counts, source_sha256=digest(__file__),
                 versions={k: importlib.metadata.version(k) for k in ('numpy', 'pandas', 'scikit-learn')},
                 limitations='Successful return is not convergence. Native warnings retained. No biological scoring or aggregate ARI.')
    (root / 'validation/local_pca_wrapup.json').write_text(json.dumps(audit, indent=2) + '\n')
    seed_variation_figure(root)
    print(summary.to_string(), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', nargs='?', default='preflight',
                        choices=['preflight', 'run', 'validate', 'figures', 'analyse', 'compare-records', 'summarise', '_worker'])
    parser.add_argument('--run-id')
    parser.add_argument('--records', choices=['full', 'dedup'], default='full')
    parser.add_argument('--geometry', choices=list(GEOMETRIES), default='raw',
                        help='PCA and PCA+L2 are separate input experiments; raw is unchanged')
    parser.add_argument('--seed', type=int, default=0, help='Reproducibility seed; no quality selection')
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--attempt', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--analysis-output', type=Path,
                        help='New destination for derived tables/plots, outside the saved return')
    parser.add_argument('--other-attempt', type=Path, help='Dedup attempt directory for compare-records')
    args = parser.parse_args()
    if args.mode == 'summarise':
        summarise(args.root)
        return
    if args.mode == '_worker':
        worker(args.attempt)
        return
    if args.mode == 'preflight':
        ids, X, stored = prepare(args.records, args.geometry)
        print(json.dumps(provenance(ids, X, stored, args.records, args.seed, args.geometry), indent=2))
        return
    if not args.run_id or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*', args.run_id):
        parser.error('--run-id must contain only letters, digits, underscores or hyphens')
    path = args.root / 'attempts' / args.run_id
    if args.mode == 'run':
        run(path, args.records, args.seed, args.geometry)
    elif args.mode == 'validate':
        validate_saved(path)
        print(f'Validated: {path}')
    elif args.mode == 'analyse':
        if args.analysis_output is None:
            parser.error('analyse requires --analysis-output')
        analyse(path, args.analysis_output)
    elif args.mode == 'compare-records':
        if args.analysis_output is None or args.other_attempt is None:
            parser.error('compare-records requires --analysis-output and --other-attempt')
        compare_records(path, args.other_attempt, args.analysis_output)
    else:
        figures(path)


if __name__ == '__main__':
    main()
