from __future__ import annotations

import argparse
from contextlib import contextmanager
from functools import partial
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import shutil
import subprocess
import sys
from types import ModuleType
from typing import Any, Iterator
import zipfile


PROJECT_COMMIT = "project-folder"
STANDALONE_SHA256 = "6e63002b5aa9d788baa61a1dd2295499e853a4050a6b3286092884727dcf70f5"
LOSS = {
    "ssl_loss": "mean_squared_error",
    "ssl_loss_reduction": "sum_features_mean_samples",
    "ssl_loss_weight": 1.0,
    "clustering_loss_weight": 1.0,
}
ARMS = (
    ("prott5_raw1024_full", "prott5", None, False),
    ("esm2_3b_raw2560_full", "esm2_3b", None, False),
    ("prott5_pca67_full", "prott5", 67, False),
    ("esm2_3b_pca45_full", "esm2_3b", 45, False),
    ("prott5_pca113_full", "prott5", 113, False),
    ("prott5_pca34_full", "prott5", 34, False),
    ("prott5_raw1024_dedup", "prott5", None, True),
    ("prott5_pca67_dedup", "prott5", 67, True),
    ("prott5_pca113_dedup", "prott5", 113, True),
    ("prott5_pca34_dedup", "prott5", 34, True),
)


def block_spec(sequence: int) -> dict[str, Any]:
    if not 1 <= sequence <= len(ARMS) * 10:
        raise ValueError("sequence must be between 1 and 100")
    arm, embedding, pca, dedup = ARMS[(sequence - 1) // 10]
    seed = (sequence - 1) % 10
    return {
        "sequence": sequence, "arm": arm, "embedding": embedding,
        "pca_dimensions": pca, "dedup": dedup, "seed": seed,
        "block_id": f"{sequence:03d}_{arm}_seed{seed:02d}",
        "grid_fits": 100, "reference_fits": 1,
    }


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_standalone(repo: Path, *, require_commit: bool) -> ModuleType:
    path = repo / "src" / "algo" / "dipdeck" / "dipdeck_standalone.py"
    if digest(path) != STANDALONE_SHA256:
        raise ValueError("DipDECK runner differs from the tested source; do not bypass this check")
    spec = importlib.util.spec_from_file_location("dipdeck_library_loss_base", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def make_factory(ds: ModuleType):
    import torch
    from clustpy.deep import DipDECK
    from clustpy.deep._utils import mean_squared_error

    if inspect.signature(DipDECK.__init__).parameters["ssl_loss_fn"].default is not mean_squared_error:
        raise ValueError("Pinned DipDECK no longer defaults to the expected library loss")
    calls = []

    def factory(config, device, source_root):
        estimator, runtime = ds._create_dipdeck(config, device, source_root)
        if not isinstance(estimator.ssl_loss_fn, torch.nn.MSELoss) or estimator.ssl_loss_fn.reduction != "mean":
            raise ValueError("Original factory no longer supplies MSELoss(mean)")
        estimator.ssl_loss_fn = mean_squared_error
        calls.append(config)
        print(
            f"New-loss fit {len(calls)}: seed={config.seed}, "
            f"K={config.n_clusters_init}, threshold={config.dip_merge_threshold}",
            flush=True,
        )
        return estimator, runtime

    return factory, calls


@contextmanager
def reference_loss_reporting(ds: ModuleType) -> Iterator[None]:
    original = ds._persist_and_reopen

    def persist(paths, row, run_row, cluster_rows, payload):
        run_row.update(LOSS)
        return original(paths, row, run_row, cluster_rows, payload)

    ds._persist_and_reopen = persist
    try:
        yield
    finally:
        ds._persist_and_reopen = original


def options(block: dict[str, Any]) -> dict[str, Any]:
    return {
        "embedding": block["embedding"], "dedup": block["dedup"],
        "pca_dimensions": block["pca_dimensions"], "seeds": (block["seed"],),
    }


def owned_paths(ds: ModuleType, root: Path, block: dict[str, Any], kind: str):
    space = ds._input_space(block["embedding"], block["pca_dimensions"])
    if kind == "reference":
        return ds._scientific_reference_paths(
            root, block["seed"], dedup=block["dedup"], input_space=space
        )
    return ds._scientific_paths(root, dedup=block["dedup"], input_space=space)


def identity(block: dict[str, Any]) -> dict[str, Any]:
    return {
        **block, **LOSS, "project_commit": PROJECT_COMMIT,
        "standalone_sha256": STANDALONE_SHA256, "adapter_sha256": digest(Path(__file__)),
    }


def check_identity(root: Path, block: dict[str, Any], *, create: bool) -> None:
    expected = identity(block)
    path = root / "block.json"
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != expected:
            raise ValueError(f"Different block, loss or adapter already occupies {root}")
    elif create:
        if root.exists() and any(root.iterdir()):
            raise ValueError(f"Nonempty output without block provenance: {root}")
        root.mkdir(parents=True, exist_ok=True)
        write_json(path, expected)
        shutil.copyfile(__file__, root / "dipdeck_library_loss.py")
    else:
        raise FileNotFoundError(path)


def validate_stage(ds: ModuleType, root: Path, block: dict[str, Any], kind: str):
    kwargs = options(block)
    if kind == "reference":
        runs = ds._validate_scientific_reference_outputs(root, **kwargs)
        for key, value in LOSS.items():
            if not (runs[key] == value).all():
                raise ValueError(f"Reference reports the wrong {key}")
    else:
        runs = ds._validate_scientific_outputs(
            owned_paths(ds, root, block, kind), expected_rows=100, **kwargs
        )
    if len(runs) != (1 if kind == "reference" else 100):
        raise ValueError(f"Wrong number of {kind} fits")
    return runs


def check_completion(ds: ModuleType, root: Path, block: dict[str, Any], kind: str):
    import pandas as pd

    check_identity(root, block, create=False)
    marker = json.loads((root / f"{kind}.complete.json").read_text(encoding="utf-8"))
    if marker["block_id"] != block["block_id"] or marker["kind"] != kind:
        raise ValueError("Completion marker belongs to another block")
    for relative, expected in marker["sha256"].items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root.resolve()) or digest(path) != expected:
            raise ValueError(f"Completed output changed: {relative}")
    runs = validate_stage(ds, root, block, kind)
    provenance = pd.read_csv(root / f"{kind}_loss.csv")
    if provenance["row"].tolist() != runs["row"].tolist():
        raise ValueError("Loss provenance does not cover exactly the validated rows")
    for key, value in LOSS.items():
        if not (provenance[key] == value).all():
            raise ValueError(f"Wrong {key} in loss provenance")
    return runs


def complete_stage(ds: ModuleType, root: Path, block: dict[str, Any], kind: str, runs) -> None:
    provenance = runs[["row"]].copy()
    for key, value in LOSS.items():
        provenance[key] = value
    provenance.to_csv(root / f"{kind}_loss.csv", index=False)
    files = list(owned_paths(ds, root, block, kind).root.rglob("*"))
    files += [root / "block.json", root / "dipdeck_library_loss.py",
              root / f"{kind}_loss.csv", root / f"{kind}_environment.json"]
    write_json(root / f"{kind}.complete.json", {
        "block_id": block["block_id"], "kind": kind, "fits": len(runs),
        "sha256": {path.relative_to(root).as_posix(): digest(path)
                   for path in sorted(files) if path.is_file()},
    })


def package(ds: ModuleType, root: Path, block: dict[str, Any], kind: str, archives: Path) -> Path:
    kinds = ("reference",) if kind == "reference" else ("reference", "grid")
    files = {root / "block.json", root / "dipdeck_library_loss.py"}
    for part in kinds:
        check_completion(ds, root, block, part)
        files.update(path for path in owned_paths(ds, root, block, part).root.rglob("*") if path.is_file())
        files.update(root / name for name in (
            f"{part}.complete.json", f"{part}_loss.csv", f"{part}_environment.json"
        ))
    archives.mkdir(parents=True, exist_ok=True)
    suffix = "reference" if kind == "reference" else "complete"
    target = archives / f"{block['block_id']}_{suffix}.zip"
    temporary = target.with_suffix(".partial.zip")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as packed:
        for path in sorted(files):
            packed.write(path, (Path(root.name) / path.relative_to(root)).as_posix())
    with zipfile.ZipFile(temporary) as packed:
        if packed.testzip() is not None or len(packed.namelist()) != len(files):
            raise ValueError(f"Archive failed validation: {temporary}")
    temporary.replace(target)
    print(f"Output archive: {target}", flush=True)
    print(f"Output SHA256: {digest(target)}", flush=True)
    return target


def run_stage(ds: ModuleType, root: Path, block: dict[str, Any], kind: str, source_root: Path) -> None:
    import torch

    factory, calls = make_factory(ds)
    if not torch.cuda.is_available() or "T4" not in torch.cuda.get_device_name(0):
        raise RuntimeError("Use a T4 runtime for this batch; no automatic hardware substitution")
    write_json(root / f"{kind}_environment.json", {
        "python": sys.version, "torch": torch.__version__,
        "cuda_device": torch.cuda.get_device_name(0),
        "clustpy_commit": ds.CLUSTPY_COMMIT, "dipdeck_sha256": ds.DIPDECK_SOURCE_SHA256,
    })
    kwargs = options(block)
    with reference_loss_reporting(ds):
        run = ds._run_scientific_reference if kind == "reference" else ds._run_scientific_grid
        run(root, device="cuda", source_root=source_root, estimator_factory=factory, **kwargs)
    expected = 1 if kind == "reference" else 100
    if len(calls) != expected:
        raise ValueError(f"Expected {expected} new-loss fits, constructed {len(calls)}")
    runs = validate_stage(ds, root, block, kind)
    complete_stage(ds, root, block, kind, runs)
    print(f"VALIDATED {block['block_id']}: {kind}, {len(runs)} fits", flush=True)


def execute(ds: ModuleType, base: Path, archives: Path, sequence: int, kind: str,
            source_root: Path, *, validate_only: bool = False) -> None:
    if base.name != "dipdeck_library_loss":
        raise ValueError("Output base must be named dipdeck_library_loss to isolate the old results")
    block = block_spec(sequence)
    root = base / block["block_id"]
    check_identity(root, block, create=not validate_only)
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
            handle.write(f"{kind}\n")
        if kind == "grid":
            check_completion(ds, root, block, "reference")
        marker = root / f"{kind}.complete.json"
        if marker.exists() or validate_only:
            check_completion(ds, root, block, kind)
            print(f"SKIP {block['block_id']} {kind}: completed output validates", flush=True)
        else:
            print(f"RUN {block['block_id']} {kind}; incomplete output of this stage is replaced", flush=True)
            run_stage(ds, root, block, kind, source_root)
        package(ds, root, block, kind, archives)
    finally:
        lock.unlink()


def merge_arm(ds: ModuleType, base: Path, destination: Path, arm: str) -> None:
    import pandas as pd

    if base.name != "dipdeck_library_loss" or destination.name != "dipdeck_library_loss_merged":
        raise ValueError("Use dedicated dipdeck_library_loss and dipdeck_library_loss_merged directories")
    if destination == base or destination.is_relative_to(base) or base.is_relative_to(destination):
        raise ValueError("Merged output must be separate from the block source tree")
    blocks = [block_spec(i) for i in range(1, 101) if block_spec(i)["arm"] == arm]
    sources = [base / block["block_id"] for block in blocks]
    for root, block in zip(sources, blocks):
        for kind in ("reference", "grid"):
            check_completion(ds, root, block, kind)
    marker = destination / "library_loss.json"
    expected = {**LOSS, "adapter_sha256": digest(Path(__file__)),
                "project_commit": PROJECT_COMMIT, "standalone_sha256": STANDALONE_SHA256}
    if marker.exists():
        if json.loads(marker.read_text(encoding="utf-8")) != expected:
            raise ValueError("Merged root belongs to another loss or adapter")
    elif destination.exists() and any(destination.iterdir()):
        raise ValueError("Refusing to merge into an existing tree without new-loss provenance")
    destination.mkdir(parents=True, exist_ok=True)
    write_json(marker, expected)
    for suffix in ("loss.csv", "blocks.json"):
        (destination / f"{arm}_{suffix}").unlink(missing_ok=True)
    bound = {}
    if blocks[0]["dedup"]:
        for name in ("_scientific_paths", "_scientific_reference_paths",
                     "_validate_scientific_outputs", "_validate_scientific_reference_outputs"):
            bound[name] = getattr(ds, name)
            setattr(ds, name, partial(bound[name], dedup=True))
    try:
        ds._merge_seed_halves(
            sources, destination, embedding=blocks[0]["embedding"],
            pca_dimensions=blocks[0]["pca_dimensions"],
        )
    finally:
        for name, function in bound.items():
            setattr(ds, name, function)
    destination.mkdir(parents=True, exist_ok=True)
    provenance = pd.concat(
        [pd.read_csv(root / f"{kind}_loss.csv") for root in sources for kind in ("reference", "grid")],
        ignore_index=True,
    )
    if len(provenance) != 1010 or provenance["row"].duplicated().any():
        raise ValueError("Merged arm does not contain 1,010 distinct fits")
    provenance.to_csv(destination / f"{arm}_loss.csv", index=False)
    write_json(destination / f"{arm}_blocks.json", [identity(block) for block in blocks])
    print(f"MERGED and validated {arm}: 1,000 fold fits and 10 references", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output-base", type=Path, required=True)
    parser.add_argument("--sequence", type=int)
    parser.add_argument("--kind", choices=("reference", "grid"))
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--archives", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--merge-arm", choices=[arm[0] for arm in ARMS])
    parser.add_argument("--merge-output", type=Path)
    args = parser.parse_args()
    if args.merge_arm:
        if args.merge_output is None:
            parser.error("--merge-arm requires --merge-output")
        ds = load_standalone(args.repo.resolve(), require_commit=False)
        merge_arm(ds, args.output_base.resolve(), args.merge_output.resolve(), args.merge_arm)
    else:
        if args.sequence is None or args.kind is None or args.archives is None or args.source_root is None:
            parser.error("running a block requires --sequence, --kind, --archives and --source-root")
        sys.path.insert(0, str(args.source_root.resolve()))
        ds = load_standalone(args.repo.resolve(), require_commit=not args.validate_only)
        execute(ds, args.output_base.resolve(), args.archives.resolve(), args.sequence,
                args.kind, args.source_root.resolve(), validate_only=args.validate_only)


if __name__ == "__main__":
    main()
