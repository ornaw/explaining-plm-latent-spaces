from __future__ import annotations

import hashlib
from importlib import metadata
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
CLONE = REPO / 'external/graph-part'
ENV = {**os.environ, 'PYTHONIOENCODING': 'utf-8', 'PYTHONPATH': str(CLONE),
       'PYTHONDONTWRITEBYTECODE': '1'}
GRAPHPART_COMMIT = 'f470726216bc7ec4cb60f6424548bfd60c550e42'
THRESHOLDS = (0.30, 0.40)
PARTITIONS = 5

FASTA = HERE / 'graphpart_825088fecab1/sequences_graphpart.fasta'
EDGES = HERE / 'edges_shortest.checkpoint'
EXPECTED_SHA256 = {
    FASTA: '1105a977a6ffeb6eb5c0777316ba86d50473fd6fb986cb2b2197273f76ccd09e',
    EDGES: 'c4b7347b158051c0d691487b71294a0ed9d90085af68462c79f32fc95d1f6656',
}
EXPECTED_SEQUENCES = 1480


def sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def git(*args):
    return subprocess.run(['git', '-C', str(CLONE), *args], capture_output=True, text=True,
                          check=True).stdout.strip()


def check_inputs():
    head = git('rev-parse', 'HEAD')
    if head != GRAPHPART_COMMIT:
        raise RuntimeError(f'external/graph-part is at {head}, not the audited {GRAPHPART_COMMIT}')
    if git('status', '--porcelain', '--untracked-files=no'):
        raise RuntimeError('external/graph-part has edits to tracked files')
    try:
        metadata.version('graph-part')
    except metadata.PackageNotFoundError:
        pass
    else:
        raise RuntimeError('graph-part is installed in this environment; this script runs the clone')
    for path, digest in EXPECTED_SHA256.items():
        if sha256(path) != digest:
            raise RuntimeError(f'{path.name} changed: sha256 is not {digest}')
    ids = [line[1:].strip() for line in FASTA.read_text().splitlines() if line.startswith('>')]
    if len(set(ids)) != len(ids) or len(ids) != EXPECTED_SEQUENCES:
        raise RuntimeError(f'Expected {EXPECTED_SEQUENCES} unique IDs, found {len(set(ids))}')
    return ids


def run_graphpart(threshold, out_csv, *, fasta=FASTA, partitions=PARTITIONS, labels_name=None):
    command = [sys.executable, '-c', 'from graph_part import run_graph_part; run_graph_part()',
               'precomputed',
               '--fasta-file', str(fasta), '--edge-file', str(EDGES), '--metric-column', '2',
               '--transformation', 'one-minus', '--threshold', str(threshold),
               '--partitions', str(partitions), '--out-file', str(out_csv)]
    if labels_name is not None:
        command.extend(['--labels-name', labels_name])
    started = time.time()
    proc = subprocess.run(command, capture_output=True, text=True, encoding='utf-8',
                          env=ENV)
    return command, proc, round(time.time() - started, 1)


def main():
    ids = check_inputs()
    source = subprocess.run([sys.executable, '-c', 'import graph_part; print(graph_part.__file__)'],
                            capture_output=True, text=True, check=True, env=ENV).stdout.strip()
    if not Path(source).resolve().is_relative_to(CLONE):
        raise RuntimeError(f'graph_part imported from {source}, not the clone')
    out_dir = REPO / 'results/graphpart' / time.strftime('folds_%Y%m%dT%H%M%SZ', time.gmtime())
    out_dir.mkdir(parents=True)
    runs = []
    for threshold in THRESHOLDS:
        tag = 't' + f'{threshold:.2f}'.replace('.', 'p')
        out_csv = out_dir / f'folds_{tag}.csv'
        command, proc, elapsed = run_graphpart(threshold, out_csv)
        (out_dir / f'log_{tag}.txt').write_text(proc.stdout + proc.stderr, encoding='utf-8')
        run = dict(threshold=threshold, tag=tag, returncode=proc.returncode, elapsed_s=elapsed,
                   command=' '.join(Path(c).name if c == sys.executable else c for c in command))
        if proc.returncode != 0 or not out_csv.exists():
            run.update(ok=False, note=(proc.stdout + proc.stderr).strip().splitlines()[-1:])
            print(tag, 'FAILED, see', f'log_{tag}.txt', flush=True)
            runs.append(run)
            continue

        folds = pd.read_csv(out_csv)
        kept = set(folds['AC'])
        if kept - set(ids):
            raise RuntimeError(f'{tag}: output IDs missing from the input FASTA')
        removed = sorted(set(ids) - kept)
        (out_dir / f'removed_{tag}.txt').write_text(''.join(r + '\n' for r in removed))
        assert len(kept) + len(removed) == EXPECTED_SEQUENCES

        repeat_csv = out_dir / f'repeat_{tag}.csv'
        run_graphpart(threshold, repeat_csv)
        repeat = pd.read_csv(repeat_csv).set_index('AC')['cluster'].sort_index()
        repeat_csv.unlink()
        repeat_csv.with_name(f'repeat_{tag}_report.json').unlink(missing_ok=True)
        same = folds.set_index('AC')['cluster'].sort_index().equals(repeat)
        if not same:
            raise RuntimeError(f'{tag}: a repeat run gave different folds')

        sizes = folds['cluster'].value_counts().sort_index()
        run.update(ok=True, retained=len(kept), removed=len(removed), repeat_identical=same,
                   fold_sizes={int(k): int(v) for k, v in sizes.items()})
        print(f'{tag}: retained {len(kept)}, removed {len(removed)}, folds {run["fold_sizes"]}',
              flush=True)
        runs.append(run)

    freeze = subprocess.run([sys.executable, '-m', 'pip', 'freeze'], capture_output=True,
                            text=True, check=True).stdout
    (out_dir / 'pip_freeze.txt').write_text(freeze, encoding='utf-8')
    config = dict(
        run_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        graphpart_commit=GRAPHPART_COMMIT,
        graphpart_source=Path(source).resolve().relative_to(REPO).as_posix(),
        python=sys.version.split()[0], platform=platform.platform(),
        identity='NW / shortest, from edges_shortest.checkpoint',
        partitions=PARTITIONS, labels_name=None,
        input_sha256={p.relative_to(REPO).as_posix(): d for p, d in EXPECTED_SHA256.items()},
        script_sha256=sha256(Path(__file__)),
        sequences_in=EXPECTED_SEQUENCES, runs=runs)
    (out_dir / 'run_config.json').write_text(json.dumps(config, indent=2) + '\n', encoding='utf-8')
    print('wrote', out_dir.relative_to(REPO).as_posix())


if __name__ == '__main__':
    main()
