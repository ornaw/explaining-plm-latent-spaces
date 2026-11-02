from __future__ import annotations

import argparse
import contextlib
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import multiprocessing
from pathlib import Path
import sys
import time
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
from threadpoolctl import threadpool_limits

from src.algo.autonr import autonr_standalone as anr

REPO = anr.REPO
MODEL_KEY, MODEL_STEM = "esm2_3b", "esm2_t36_3B_UR50D"
H5_PATH = REPO / "data/phosphatase/phosphatase/esm2_t36_3B_UR50D_full_length.h5"
MANIFEST_ROOT = REPO / "results/esm2" / MODEL_STEM
PROTT5_H5, ANNOTATION_PATH, FASTA_PATH = anr.DATASETS["phosphatase"]
EXPECTED_N, WIDTH = 1_587, 2_560
GEOMETRY, RECORDS = "pca90", "full"
VARIANCE = anr.GEOMETRIES[GEOMETRY][0]
EXPECTED_WIDTH = 45
SEEDS = tuple(range(10))
OUT = REPO / "results/autonr" / MODEL_STEM
BUNDLE = Path(__file__).with_name("esm2_autonr_hf_inputs.zip")
BUNDLE_FILES = (
    "src/data.py", "src/esm2_analysis.py", "src/esm2_embeddings.py",
    "src/algo/autonr/autonr_standalone.py", "src/algo/autonr/esm2_autonr_standalone.py",
    "tests/__init__.py", "tests/test_esm2_autonr_standalone.py",
    "data/phosphatase/phosphatase/esm2_t36_3B_UR50D_full_length.h5",
    "data/phosphatase/phosphatase/tmp/prot_t5.h5",
    "data/phosphatase/phosphatase/data_standard.parquetbundle",
    "data/phosphatase/phosphatase/tmp/sequences.fasta",
    "results/esm2/esm2_t36_3B_UR50D/embedding_manifest.json",
)


def prepare(records="full", geometry="pca90", transform_out=None):
    from src import esm2_analysis
    if (records, geometry) != (RECORDS, GEOMETRY):
        raise ValueError(f"ESM-2 AutoNR runs {RECORDS} records at {GEOMETRY} only")
    model = esm2_analysis.load_model_dataset(
        h5_path=H5_PATH, output_root=MANIFEST_ROOT, annotation_path=ANNOTATION_PATH,
        fasta_path=FASTA_PATH, reference_h5=PROTT5_H5, repository_root=REPO,
    )
    if model.model_key != MODEL_KEY or model.data.X.shape != (EXPECTED_N, WIDTH):
        raise ValueError(f"expected {MODEL_KEY} at {(EXPECTED_N, WIDTH)}, got {model.data.X.shape}")
    stored = str(model.data.X.dtype)
    ids, X = anr.canonical_input(np.asarray(model.data.ids, dtype=str), model.data.X)
    with threadpool_limits(limits=1):
        X, basis = anr.pca(X, VARIANCE)
    if X.shape[1] != EXPECTED_WIDTH:
        raise ValueError(f"PCA at {VARIANCE:.0%} gave {X.shape[1]} components, not {EXPECTED_WIDTH}")
    if transform_out is not None:
        transform_out.update(basis)
    return ids, X, stored


anr.prepare = prepare


def run_id(seed: int) -> str:
    return f"esm2_3b_pca90_full_seed{seed:02d}"


def provenance(ids, X, stored, seed, threads):
    record = anr.provenance(ids, X, stored, RECORDS, seed, GEOMETRY)
    record.update(
        embedding=dict(model_key=MODEL_KEY, stem=MODEL_STEM, width=WIDTH, pca_width=int(X.shape[1])),
        esm2_input_files={str(p.relative_to(REPO)): anr.digest(p) for p in (H5_PATH, MANIFEST_ROOT / "embedding_manifest.json")},
        esm2_runner_sha256=anr.digest(__file__),
        execution=dict(in_process=True, blas_threads=threads),
    )
    return record


def fit_one(seed: int, root: Path, threads: int) -> str:
    path = Path(root) / "attempts" / run_id(seed)
    if (path / "SUCCESS.json").exists():
        anr.validate_saved(path)
        return f"seed {seed}: already complete, revalidated"
    if path.exists():
        moved = Path(root) / "interrupted" / f"{path.name}_{time.strftime('%Y%m%dT%H%M%S')}"
        moved.parent.mkdir(parents=True, exist_ok=True)
        path.rename(moved)
    path.mkdir(parents=True)
    with threadpool_limits(limits=threads):
        ids, X, stored = prepare()
        anr.write_json(path / "configuration.json", provenance(ids, X, stored, seed, threads))
        (path / "runner_snapshot.py").write_bytes(Path(anr.__file__).read_bytes())
        (path / "esm2_runner_snapshot.py").write_bytes(Path(__file__).read_bytes())
        started = time.monotonic()
        with (path / "stdout.log").open("x") as log, contextlib.redirect_stdout(log):
            anr.worker(path)
    anr.validate_saved(path, require_success=False)
    anr.write_json(path / "SUCCESS.json", dict(result_sha256=anr.digest(path / "result.json")))
    return f"seed {seed}: fitted and validated in {time.monotonic() - started:.0f} s"


def run(root: Path, seeds, workers: int, threads: int) -> None:
    anr.verify_source()
    pending = [s for s in seeds if not (Path(root) / "attempts" / run_id(s) / "SUCCESS.json").exists()]
    print(f"{len(pending)} of {len(seeds)} seeds to fit; {workers} at a time, {threads} BLAS threads each", flush=True)
    failures = []
    if workers == 1:
        for seed in pending:
            print(fit_one(seed, root, threads), flush=True)
    else:
        with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("spawn")) as pool:
            futures = {pool.submit(fit_one, seed, root, threads): seed for seed in pending}
            for future in as_completed(futures):
                try:
                    print(future.result(), flush=True)
                except Exception as error:
                    failures.append(futures[future])
                    print(f"seed {futures[future]} failed: {error!r}", flush=True)
    if failures:
        raise RuntimeError(f"Seeds failed: {sorted(failures)}; each attempt keeps failure.txt")
    validate(root, seeds)


def validate(root: Path, seeds) -> None:
    rows = []
    for seed in seeds:
        a, result = anr.validate_saved(Path(root) / "attempts" / run_id(seed))
        rows.append(dict(seed=seed, subspaces=len(a["m"]), dimensions=a["m"].tolist(),
                         clusters=a["n_clusters"].tolist(), fit_seconds=round(result["fit_seconds"])))
        print(rows[-1], flush=True)
    print(f"Validated {len(rows)} of {len(seeds)} seeds", flush=True)


def preflight() -> None:
    anr.verify_source()
    basis = {}
    ids, X, stored = prepare(transform_out=basis)
    print(json.dumps(dict(**anr.input_signature(ids, X), stored_dtype=stored,
                          retained_variance=float(np.sum(basis["pca_explained_variance_ratio"])),
                          constructor=anr.constructor(0)), indent=2))


def bundle() -> None:
    files = {name: anr.digest(REPO / name) for name in BUNDLE_FILES}
    with zipfile.ZipFile(BUNDLE, "x", compression=zipfile.ZIP_DEFLATED) as z:
        for name in BUNDLE_FILES:
            z.write(REPO / name, name)
        z.writestr("transfer_manifest.json", json.dumps(dict(files=files), indent=2, sort_keys=True))
    print(BUNDLE, hashlib.sha256(BUNDLE.read_bytes()).hexdigest())


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").partition("\n")[0])
    parser.add_argument("mode", nargs="?", default="preflight", choices=("preflight", "run", "validate", "bundle"))
    parser.add_argument("--root", type=Path, default=OUT)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--workers", type=int, default=1, help="seeds fitted side by side (execution choice)")
    parser.add_argument("--threads", type=int, default=4, help="BLAS threads per seed (execution choice)")
    args = parser.parse_args()
    if not set(args.seeds) <= set(SEEDS):
        raise ValueError("Seeds 0 to 9 only")
    if args.mode == "preflight":
        preflight()
    elif args.mode == "run":
        run(args.root, args.seeds, args.workers, args.threads)
    elif args.mode == "validate":
        validate(args.root, args.seeds)
    else:
        bundle()


if __name__ == "__main__":
    main()
