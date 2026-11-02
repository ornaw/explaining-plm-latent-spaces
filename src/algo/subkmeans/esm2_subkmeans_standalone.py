from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import itertools
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import pandas as pd

from src.algo.kmeans import esm2_kmeans_standalone as ekm
from src.algo.subkmeans import subkmeans_standalone as skm

REPO = ekm.REPO
MODEL_STEM = ekm.MODEL_STEM
OUT = REPO / "results/subkmeans" / MODEL_STEM
KMEANS = ekm.output_paths()
GEOMETRY, RECORDS = "raw2560", "full"
EXPECTED_K = 11
EXPECTED_N, WIDTH = ekm.EXPECTED_N, ekm.WIDTH
OUTER_SEEDS = skm.OUTER_SEEDS
RESTARTS = skm.RESTARTS_PER_FIT
THREADS = skm.THREADS
DTYPE = "float64"
PREFIX = f"subkmeans_{MODEL_STEM}"
TABLES = {name: f"{PREFIX}_{name}.csv" for name in
          ("runs", "restarts", "clusters", "scores", "objective_vs_kmeans", "partition_agreement",
           "agreement_summary")}
MANIFEST_FIELDS = ["row", "geometry", "records", "requested_k", "outer_seed", "k_source",
                   "restart_seeds", "effective_starts", "fit_dtype"]


def digest(path: Path) -> str:
    return skm.digest(path)


def matrix_sha256(X: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(X, dtype=np.float64).tobytes()).hexdigest()


def coordinates() -> tuple[np.ndarray, np.ndarray]:
    model = ekm.load_esm2()
    X = ekm.spaces(model.data.X)[GEOMETRY]
    ids, X = skm.canonical_input(np.asarray(model.data.ids, dtype=str), X)
    if X.shape != (EXPECTED_N, WIDTH) or X.dtype != np.float64 or list(ids) != list(model.data.ids):
        raise ValueError("ESM-2 coordinates are not the 1,587 x 2,560 float64 rows in protein-ID order")
    return ids, X


def modal_k(selection_path: Path = KMEANS.selection) -> tuple[int, dict[int, int]]:
    selection = pd.read_csv(selection_path)
    primary = selection[selection["is_primary"].astype(bool) & (selection["geometry"] == GEOMETRY)]
    if sorted(primary["seed"].astype(int)) != list(OUTER_SEEDS) or set(primary["records"]) != {RECORDS}:
        raise ValueError("the ESM-2 selection table needs one primary full-record row per seed 0 to 9")
    k = ekm.km._modal_k(primary)[GEOMETRY]
    counts = primary["selected_k"].astype(int).value_counts().sort_index()
    return int(k), {int(key): int(value) for key, value in counts.items()}


def run_id(seed: int, k: int) -> str:
    return f"SKM-E3B-R2560-F-S{seed:02d}-K{k:02d}"


def build_manifest(k: int) -> list[dict]:
    used, overflow, rows = set(), itertools.count(skm.OVERFLOW_SEED_BASE), []
    for seed in OUTER_SEEDS:
        seeds = skm.restart_seed_pool(seed * RESTARTS, used, overflow)
        rows.append(dict(row=run_id(seed, k), geometry=GEOMETRY, records=RECORDS, requested_k=k,
                         outer_seed=seed, k_source=f"ESM-2 k-means modal K, {GEOMETRY} primary",
                         restart_seeds=seeds, effective_starts=[skm.effective_start(s) for s in seeds],
                         fit_dtype=DTYPE))
    starts = [start for row in rows for start in row["effective_starts"]]
    if len(starts) != len(OUTER_SEEDS) * RESTARTS or len(set(starts)) != len(starts):
        raise ValueError("two restarts in the group share an effective start")
    return rows


def git_revision() -> str:
    try:
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain", "--", ".", f":(exclude){OUT.relative_to(REPO).as_posix()}/**"],
            cwd=REPO, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return head + ("-dirty" if status else "")


def inputs() -> dict[str, str]:
    paths = [ekm.H5_PATH, ekm.MANIFEST_ROOT / "embedding_manifest.json", KMEANS.selection, KMEANS.runs]
    return {path.relative_to(REPO).as_posix(): digest(path) for path in paths}


def provenance(k: int, counts: dict[int, int], X: np.ndarray, ids: np.ndarray) -> dict:
    return dict(
        requested_k=k, selected_k_counts=counts, outer_seeds=list(OUTER_SEEDS), restarts_per_fit=RESTARTS,
        fit_dtype=DTYPE, threads=THREADS, settings=skm.FIXED_SETTINGS, source_commit=skm.COMMIT,
        source_hashes=skm.SOURCES, coordinates_sha256=matrix_sha256(X), n=len(ids), d=int(X.shape[1]),
        subset_hash=skm.subset_hash(ids.tolist()), inputs=inputs(),
        code={path.relative_to(REPO).as_posix(): digest(path) for path in
              (Path(__file__).resolve(), Path(skm.__file__).resolve(), Path(ekm.__file__).resolve(),
               REPO / "src/esm2_analysis.py")},
        versions={p: importlib.metadata.version(p) for p in
                  ("clustpy", "numpy", "scipy", "scikit-learn", "threadpoolctl", "pandas")})


SCIENCE_KEYS = ("requested_k", "outer_seeds", "restarts_per_fit", "fit_dtype", "threads", "settings",
                "source_commit", "source_hashes", "coordinates_sha256", "n", "d", "subset_hash", "inputs",
                "versions")


def freeze_manifest(root: Path, rows: list[dict], frozen: dict) -> pd.DataFrame:
    directory = root / "manifest"
    table_path, meta_path = directory / "manifest.csv", directory / "manifest.json"
    table = pd.DataFrame([{**row, "restart_seeds": " ".join(map(str, row["restart_seeds"])),
                           "effective_starts": " ".join(map(str, row["effective_starts"]))} for row in rows],
                         columns=MANIFEST_FIELDS)
    if not meta_path.exists():
        directory.mkdir(parents=True, exist_ok=True)
        table.to_csv(table_path, index=False)
        skm.write_json(meta_path, {**frozen, "built": time.strftime("%Y-%m-%d %H:%M:%S")})
    recorded = json.loads(meta_path.read_text())
    differing = [key for key in SCIENCE_KEYS if json.dumps(recorded.get(key), sort_keys=True)
                 != json.dumps(json.loads(json.dumps(frozen[key])), sort_keys=True)]
    reopened = pd.read_csv(table_path, dtype=str)
    if differing or not reopened.equals(table.astype(str)):
        raise ValueError(f"the frozen manifest differs from this run: {differing or 'manifest.csv'}")
    return reopened


def _paths(root: Path, rid: str) -> list[Path]:
    return [root / part / rid for part in ("assignments", "metrics", "logs")]


def set_aside(root: Path, rid: str) -> Path | None:
    present = [p for p in _paths(root, rid) if p.exists()]
    if not present or (root / "metrics" / rid / "success.json").exists():
        return None
    target = root / "interrupted" / f"{rid}_{time.strftime('%Y%m%d_%H%M%S')}"
    for path in present:
        destination = target / path.parent.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(path), str(destination))
    return target


def fit_row(root: Path, row: dict, ids: np.ndarray, X: np.ndarray, frozen: dict, revision: str) -> dict:
    from threadpoolctl import threadpool_info, threadpool_limits

    rid, k, seeds = row["row"], int(row["requested_k"]), [int(s) for s in row["restart_seeds"]]
    paths = skm.reserve_run(root, rid)
    skm.write_json(paths["logs"] / "request.json", dict(
        run_id=rid, outer_seed=int(row["outer_seed"]), requested_k=k, requested_seeds=seeds,
        effective_starts=[skm.effective_start(s) for s in seeds], settings=dict(skm.FIXED_SETTINGS, n_clusters=k),
        threads=THREADS, project_revision=revision))
    started = time.perf_counter()
    with threadpool_limits(limits=THREADS):
        arrays, evidence = skm.fit_restarts(ids, X, k, seeds, DTYPE)
        pools = threadpool_info()
    selected, diagnostics = evidence["selected"], evidence["diagnostics"]
    metadata = dict(
        run_id=rid, purpose="scientific", geometry=GEOMETRY, records=RECORDS, n=len(ids), d=int(X.shape[1]),
        subset_hash=skm.subset_hash(ids.tolist()), requested_k=k, returned_k=selected["returned_k"],
        m=selected["m"], fit_dtype=DTYPE, restart_count=len(seeds), selected_seed=selected["seed"],
        selected_effective_start=selected["effective_start"], objective=selected["objective"],
        fit_seconds=selected["fit_seconds"], optimisation_status=selected["optimisation_status"],
        iterations=selected["iterations"], **diagnostics,
        settings=dict(skm.FIXED_SETTINGS, n_clusters=k), requested_seeds=seeds, restarts=evidence["restarts"],
        source_commit=skm.COMMIT, source_hashes=skm.SOURCES, project_revision=revision,
        embedding=MODEL_STEM, outer_seed=int(row["outer_seed"]), coordinates_sha256=frozen["coordinates_sha256"],
        inputs=frozen["inputs"], code=frozen["code"], versions=frozen["versions"],
        python=sys.version, executable=sys.executable, threads=THREADS, thread_pools=pools)
    with (paths["logs"] / "native.log").open("x", encoding="utf-8") as log:
        for restart in evidence["restarts"]:
            log.write(restart["native_log"] + "\n" + "\n".join(restart["warnings"]) + "\n")
    skm.save_result(root, rid, arrays, metadata)
    checked = skm.validate_saved(root, rid, require_success=False)
    skm.write_json(paths["metrics"] / "success.json", {"archive_sha256": checked["archive_sha256"],
                                                       "wall_seconds": time.perf_counter() - started})
    return checked


def read_run(root: Path, rid: str) -> dict:
    metadata = skm.validate_saved(root, rid, require_success=False)
    success = json.loads((root / "metrics" / rid / "success.json").read_text())
    if success["archive_sha256"] != metadata["archive_sha256"]:
        raise ValueError(f"{rid}: success record and archive disagree")
    return {**metadata, "wall_seconds": success["wall_seconds"]}


def grid_tables(root: Path, manifest: pd.DataFrame) -> dict[str, pd.DataFrame]:
    runs, restarts, clusters = [], [], []
    for row in manifest.itertuples():
        if not (root / "metrics" / row.row / "success.json").exists():
            continue
        metadata = read_run(root, row.row)
        runs.append({**{f: metadata[f] for f in skm.RUN_FIELDS}, "outer_seed": int(row.outer_seed),
                     "wall_seconds": metadata["wall_seconds"], "archive_sha256": metadata["archive_sha256"],
                     "project_revision": metadata["project_revision"], "embedding": MODEL_STEM})
        restarts.append(pd.read_csv(root / "metrics" / row.row / "restarts.csv"))
        clusters.append(pd.read_csv(root / "metrics" / row.row / "cluster_sizes.csv"))
    frames = {"runs": pd.DataFrame(runs),
              "restarts": pd.concat(restarts, ignore_index=True) if restarts else pd.DataFrame(),
              "clusters": pd.concat(clusters, ignore_index=True) if clusters else pd.DataFrame()}
    (root / "metrics").mkdir(parents=True, exist_ok=True)
    for name, frame in frames.items():
        frame.to_csv(root / "metrics" / TABLES[name], index=False)
    return frames


def prepare() -> tuple[np.ndarray, np.ndarray, int, list[dict], dict]:
    skm.verify_source()
    ids, X = coordinates()
    k, counts = modal_k()
    if k != EXPECTED_K:
        raise ValueError(f"ESM-2 k-means' modal K is {k}, expected {EXPECTED_K}")
    return ids, X, k, build_manifest(k), provenance(k, counts, X, ids)


def run_scientific(root: Path = OUT, limit: int | None = None) -> dict:
    root = Path(root)
    ids, X, k, rows, frozen = prepare()
    manifest = freeze_manifest(root, rows, frozen)
    revision = git_revision()
    by_row = {row["row"]: row for row in rows}
    done, moved, started = 0, [], time.perf_counter()
    pending = [rid for rid in manifest.row if not (root / "metrics" / rid / "success.json").exists()]
    for number, rid in enumerate(pending[:limit] if limit else pending, start=1):
        aside = set_aside(root, rid)
        if aside:
            moved.append(str(aside.relative_to(root)))
        print(f"[{number}/{len(pending)}] {rid} elapsed {time.perf_counter() - started:.0f}s", flush=True)
        result = fit_row(root, by_row[rid], ids, X, frozen, revision)
        print(f"  K {result['returned_k']}, m {result['m']}, captured {result['captured_energy']:.12f}, "
              f"moved {result['moved_under_full_reassignment']}, objective {result['objective']:.3f}", flush=True)
        done += 1
    tables = grid_tables(root, manifest)
    summary = dict(planned=len(manifest), validated=len(tables["runs"]), fitted_now=done,
                   moved_aside=moved, git=revision, wall_seconds=time.perf_counter() - started)
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def kmeans_partition(seed: int, k: int) -> tuple[pd.Series, dict]:
    runs = pd.read_csv(KMEANS.runs)
    match = runs[(runs["geometry"] == GEOMETRY) & (runs["seed"] == seed) & (runs["requested_k"] == k)
                 & (runs["realised_k"] == k)]
    if len(match) != 1:
        raise ValueError(f"ESM-2 k-means has {len(match)} K = {k} partitions at seed {seed}")
    base = match.iloc[0]
    with np.load(KMEANS.assignments / base["archive"], allow_pickle=False) as archive:
        payload = dict(protein_id=archive["protein_id"].astype(str), label=archive["label"])
    return base, payload


def score(root: Path = OUT) -> dict[str, pd.DataFrame]:
    root = Path(root)
    runs = pd.read_csv(root / "metrics" / TABLES["runs"])
    if sorted(runs["outer_seed"]) != list(OUTER_SEEDS):
        raise ValueError("score needs all ten fits")
    frozen = json.loads((root / "manifest" / "manifest.json").read_text())["inputs"]
    if frozen != inputs():
        raise ValueError("the ESM-2 inputs or k-means tables changed since the manifest was frozen")
    ids, X = coordinates()
    payloads, kpayloads, scores, objective = {}, {}, [], []
    for run in runs.itertuples():
        read_run(root, run.run_id)
        with np.load(root / "assignments" / run.run_id / "partition.npz", allow_pickle=False) as archive:
            payload = dict(protein_id=archive["protein_id"].astype(str), label=archive["label"])
        if not np.array_equal(payload["protein_id"], ids):
            raise ValueError(f"{run.run_id} does not cover the 1,587 proteins in ID order")
        payloads[run.run_id] = payload
        scores.append(dict(run_id=run.run_id, outer_seed=int(run.outer_seed), requested_k=int(run.requested_k),
                           realised_k=int(run.returned_k), n_scored=len(ids),
                           scoring_space=f"{GEOMETRY} input coordinates, Euclidean",
                           **skm.internal_scores(X, payload["label"]),
                           **skm.control_summaries(X, payload["label"], int(run.outer_seed))))
        base, kp = kmeans_partition(int(run.outer_seed), int(run.requested_k))
        if not np.array_equal(kp["protein_id"], ids):
            raise ValueError(f"{base['archive']} covers different proteins")
        kpayloads[int(run.outer_seed)] = (str(base["archive"]), kp)
        s, km_wcss = skm.wcss_from_labels(X, payload["label"]), skm.wcss_from_labels(X, kp["label"])
        objective.append(dict(run_id=run.run_id, outer_seed=int(run.outer_seed), requested_k=int(run.requested_k),
                              kmeans_archive=str(base["archive"]), kmeans_role=str(base["role"]),
                              subkmeans_wcss=s, kmeans_wcss_recomputed=km_wcss,
                              kmeans_recorded_minus_recomputed=float(base["inertia"]) - km_wcss,
                              subkmeans_recorded_minus_recomputed=float(run.wcss_label_means) - s,
                              wcss_gap=s - km_wcss, relative_gap=(s - km_wcss) / km_wcss))
    agreement = []
    names = list(payloads)
    for left, right in itertools.combinations(names, 2):
        n, ari = skm.aligned_ari(payloads[left], payloads[right])
        agreement.append(dict(comparison="subkmeans_seed_pairs", left=left, right=right, n_common=n, adjusted_rand=ari))
    for (ls, (la, lp)), (rs, (ra, rp)) in itertools.combinations(sorted(kpayloads.items()), 2):
        n, ari = skm.aligned_ari(lp, rp)
        agreement.append(dict(comparison="kmeans_seed_pairs", left=la, right=ra, n_common=n, adjusted_rand=ari))
    for run in runs.itertuples():
        archive, kp = kpayloads[int(run.outer_seed)]
        n, ari = skm.aligned_ari(payloads[run.run_id], kp)
        agreement.append(dict(comparison="subkmeans_vs_kmeans_same_seed", left=run.run_id, right=archive,
                              n_common=n, adjusted_rand=ari))
    agreement = pd.DataFrame(agreement)
    summary = (agreement.groupby("comparison")["adjusted_rand"].agg(["count", "median", "min", "max"])
               .reset_index().rename(columns={"min": "minimum", "max": "maximum"}))
    frames = {"scores": pd.DataFrame(scores), "objective_vs_kmeans": pd.DataFrame(objective),
              "partition_agreement": agreement, "agreement_summary": summary}
    for name, frame in frames.items():
        frame.to_csv(root / "metrics" / TABLES[name], index=False)
    return frames


def timing() -> float:
    from threadpoolctl import threadpool_limits
    ids, X, k, rows, _ = prepare()
    started = time.perf_counter()
    with threadpool_limits(limits=THREADS):
        _, evidence = skm.fit_restarts(ids, X, k, rows[0]["restart_seeds"][:1], DTYPE)
    seconds = time.perf_counter() - started
    print(f"one restart: {seconds:.0f} s, m {evidence['selected']['m']}, "
          f"iterations {evidence['selected']['iterations']}; ten fits of ten restarts at this rate: "
          f"{seconds * RESTARTS * len(OUTER_SEEDS) / 3600:.1f} h", flush=True)
    return seconds


def preflight() -> None:
    ids, X, k, rows, frozen = prepare()
    print(f"{len(ids)} proteins x {X.shape[1]}, coordinates {frozen['coordinates_sha256'][:12]}; K {k} "
          f"(selected-K counts {frozen['selected_k_counts']}); {len(rows)} fits x {RESTARTS} restarts, "
          f"{len({s for r in rows for s in r['effective_starts']})} distinct starts. Nothing fitted or written.")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").partition("\n")[0])
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--timing", action="store_true", help="fit one restart, timed; write nothing")
    mode.add_argument("--run-scientific", action="store_true", help="run every pending fit and write the tables")
    mode.add_argument("--score", action="store_true", help="score the saved partitions")
    parser.add_argument("--limit", type=int, default=None, help="with --run-scientific: at most this many fits")
    parser.add_argument("--output-root", type=Path, default=OUT)
    args = parser.parse_args(argv)
    if args.timing:
        timing()
    elif args.run_scientific:
        run_scientific(args.output_root, args.limit)
    elif args.score:
        score(args.output_root)
    else:
        preflight()


if __name__ == "__main__":
    main()
