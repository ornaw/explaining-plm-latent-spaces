from __future__ import annotations

import argparse
import hashlib
import importlib.util
from pathlib import Path
import sys
from types import ModuleType
from typing import Any

ADAPTER_SHA256 = "adedba78b4b6ba25d16f9ea9845f0b339e365d736626e3dd42ebc4ea55aa3bbd"
REFERENCE_INITIAL_K = 15
REFERENCE_THRESHOLD = 0.90
ARMS = (
    ("prott5_raw1024_full", "prott5", None, False),
    ("esm2_3b_raw2560_full", "esm2_3b", None, False),
)
SEEDS = tuple(range(10))
OUTPUT_BASE_NAME = "dipdeck_reference_k15"


def digest(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_adapter(path: Path) -> ModuleType:
    if digest(path) != ADAPTER_SHA256:
        raise ValueError("The library-loss adapter differs from the tested one; do not bypass this check")
    spec = importlib.util.spec_from_file_location("dipdeck_library_loss", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def block_spec(sequence: int) -> dict[str, Any]:
    if not 1 <= sequence <= len(ARMS) * len(SEEDS):
        raise ValueError(f"sequence must be between 1 and {len(ARMS) * len(SEEDS)}")
    arm, embedding, pca, dedup = ARMS[(sequence - 1) // len(SEEDS)]
    seed = SEEDS[(sequence - 1) % len(SEEDS)]
    return {
        "sequence": sequence, "arm": arm, "embedding": embedding,
        "pca_dimensions": pca, "dedup": dedup, "seed": seed,
        "block_id": f"k15_{sequence:02d}_{arm}_seed{seed:02d}",
        "grid_fits": 0, "reference_fits": 1,
        "reference_initial_k": REFERENCE_INITIAL_K, "driver_sha256": driver_sha256(),
    }


def driver_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def set_reference_k(ds: ModuleType) -> ModuleType:
    ds.SCIENTIFIC_REFERENCE_INITIAL_CLUSTERS = REFERENCE_INITIAL_K
    configs = ds._scientific_reference_configs(SEEDS)
    if {(c.n_clusters_init, c.dip_merge_threshold, c.pretrain_epochs, c.clustering_epochs)
            for c in configs} != {(REFERENCE_INITIAL_K, REFERENCE_THRESHOLD, 100, 50)}:
        raise ValueError("The reference configuration is not initial K 15, threshold 0.90, 100/50 epochs")
    return ds


def execute(adapter: ModuleType, ds: ModuleType, base: Path, archives: Path, sequence: int,
            source_root: Path, *, validate_only: bool = False) -> Path:
    if base.name != OUTPUT_BASE_NAME:
        raise ValueError(f"Output base must be named {OUTPUT_BASE_NAME} to keep it apart from the tested batch")
    if ds.SCIENTIFIC_REFERENCE_INITIAL_CLUSTERS != REFERENCE_INITIAL_K:
        raise ValueError("Call set_reference_k first")
    block = block_spec(sequence)
    root = base / block["block_id"]
    adapter.check_identity(root, block, create=not validate_only)
    lock = root / ".running"
    try:
        handle = lock.open("x", encoding="utf-8")
    except FileExistsError as error:
        raise RuntimeError(
            f"{lock} exists. Do not run this block twice. If its runtime has stopped, "
            "remove only this lock file and rerun."
        ) from error
    try:
        with handle:
            handle.write("reference\n")
        if (root / "reference.complete.json").exists() or validate_only:
            adapter.check_completion(ds, root, block, "reference")
            print(f"SKIP {block['block_id']} reference: completed output validates", flush=True)
        else:
            print(f"RUN {block['block_id']} reference at initial K {REFERENCE_INITIAL_K}", flush=True)
            adapter.run_stage(ds, root, block, "reference", source_root)
        return adapter.package(ds, root, block, "reference", archives)
    finally:
        lock.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").partition("\n")[0])
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output-base", type=Path, required=True)
    parser.add_argument("--archives", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--sequence", type=int, required=True)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    sys.path.insert(0, str(args.source_root.resolve()))
    adapter = load_adapter(args.adapter.resolve())
    ds = set_reference_k(adapter.load_standalone(args.repo.resolve(), require_commit=not args.validate_only))
    execute(adapter, ds, args.output_base.resolve(), args.archives.resolve(), args.sequence,
            args.source_root.resolve(), validate_only=args.validate_only)


if __name__ == "__main__":
    main()
