from __future__ import annotations

import argparse
import shutil
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import pdist
from sklearn.cluster import AgglomerativeClustering

from src.algo.agglomerative import agglomerative_standalone as ag
from src.algo.kmeans import esm2_kmeans_standalone as esm2_kmeans

REPO = ag.REPO
MODEL_STEM = esm2_kmeans.MODEL_STEM
EXPECTED_N = esm2_kmeans.EXPECTED_N
WIDTH = esm2_kmeans.WIDTH
RECORDS = "full"
TREE_ARMS = (
    ("raw2560", "complete", "cosine"),
    ("raw2560", "average", "cosine"),
    ("l2_raw2560", "ward", "euclidean"),
)
GEOMETRY_CODES = {"raw2560": "R2560", "l2_raw2560": "L2R2560"}
K_VALUES = ag.K_VALUES
KMEANS_GEOMETRY_FOR = {("raw2560", "cosine"): "l2_raw2560", ("l2_raw2560", "euclidean"): "l2_raw2560"}
ESM2_KMEANS_MODAL_K = {"raw2560": 11, "l2_raw2560": 11}
KMEANS_SELECTION = (REPO / "results/kmeans" / MODEL_STEM / "metrics"
                    / f"kmeans_{MODEL_STEM}_elbow_selection.csv")
PROTT5_TREE = {
    ("raw2560", "complete", "cosine"): "AG-R1024-COC-F",
    ("raw2560", "average", "cosine"): "AG-R1024-AVC-F",
    ("l2_raw2560", "ward", "euclidean"): "AG-L2R1024-WAE-F",
}
PROTT5_ASSIGNMENTS = REPO / "results/agglomerative/assignments/agglomerative_standalone"
PROTT5_REPORTED_K = 15
ELIGIBLE_TREE = ("raw2560", "complete", "cosine")

EXPECTED_PARTITIONS = 147
if EXPECTED_PARTITIONS != len(TREE_ARMS) * len(K_VALUES):
    raise RuntimeError("the ESM-2 agglomerative axes no longer match the planned count")
if {ag.reported_k("raw1024", "cosine"), ag.reported_k("l2_raw1024", "euclidean")} != {PROTT5_REPORTED_K}:
    raise RuntimeError("the ProtT5 reported K for the compared arms is no longer 15")


@dataclass(frozen=True)
class OutputPaths:
    root: Path
    runs: Path
    clusters: Path
    agreement: Path
    assignments: Path
    trees: Path


def output_paths(root: Path | None = None) -> OutputPaths:
    result_root = Path(root) if root is not None else REPO / "results/agglomerative" / MODEL_STEM
    metrics = result_root / "metrics"
    prefix = f"agglomerative_{MODEL_STEM}"
    return OutputPaths(root=result_root, runs=metrics / f"{prefix}_runs.csv",
                       clusters=metrics / f"{prefix}_clusters.csv",
                       agreement=metrics / f"{prefix}_partition_agreement.csv",
                       assignments=result_root / "assignments", trees=result_root / "trees")


def check_modal_k(selection_csv: Path = KMEANS_SELECTION) -> dict[str, int]:
    selection = pd.read_csv(selection_csv)
    primary = selection[selection["is_primary"].astype(bool)]
    modal = {}
    for geometry in ESM2_KMEANS_MODAL_K:
        counts = Counter(primary.loc[primary["geometry"] == geometry, "selected_k"].astype(int))
        if not counts:
            raise ValueError(f"no primary ESM-2 k-means selections for {geometry}")
        top = max(counts.values())
        modal[geometry] = min(k for k, c in counts.items() if c == top)
    if modal != ESM2_KMEANS_MODAL_K:
        raise ValueError(f"ESM-2 k-means modal K changed: table {modal}, code {ESM2_KMEANS_MODAL_K}")
    return modal


def reported_k(geometry: str, distance: str) -> int:
    return ESM2_KMEANS_MODAL_K[KMEANS_GEOMETRY_FOR[(geometry, distance)]]


def tree_id(geometry: str, method: str, distance: str) -> str:
    return (f"AG-E3B-{GEOMETRY_CODES[geometry]}-{ag.LINKAGE_CODES[method]}"
            f"{ag.DISTANCE_CODES[distance]}-F")


def partition_id(geometry: str, method: str, distance: str, k: int) -> str:
    if k < 2:
        raise ValueError("a partition needs at least two clusters")
    return f"{tree_id(geometry, method, distance)}-K{k:02d}"


def fit_arm(X: np.ndarray, ids: list[str], geometry: str, method: str, distance: str,
            k_values=K_VALUES):
    order = ag._canonical_order(ids)
    Z = ag.fit_tree(X[order], method, distance)
    runs, clusters, payloads = [], [], {}
    for k in k_values:
        labels = ag._labels_in_input_order(ag.cut(Z, k), order)
        row = partition_id(geometry, method, distance, k)
        realised = len(np.unique(labels))
        runs.append({"row": row, "tree": tree_id(geometry, method, distance), "geometry": geometry,
                     "linkage": method, "distance": distance, "requested_k": k, "realised_k": realised,
                     "cluster_count_shortfall": realised != k,
                     "reported_partition": k == reported_k(geometry, distance),
                     "eligible_cross_method": (geometry, method, distance) == ELIGIBLE_TREE
                     and k == reported_k(geometry, distance),
                     "n": len(labels), **ag._size_summary(labels)})
        clusters.extend(ag._cluster_rows(row, labels))
        payloads[row] = {"protein_id": np.asarray(ids, dtype=str), "label": labels}
    return Z, order, runs, clusters, payloads


def sklearn_check(X, ids, geometry, method, distance, payloads, k_values=K_VALUES) -> list[dict]:
    order = ag._canonical_order(ids)
    rows = []
    for k in k_values:
        model = AgglomerativeClustering(n_clusters=k, linkage=method, metric=distance)
        labels = ag._labels_in_input_order(model.fit_predict(X[order]).astype(np.int32), order)
        row = partition_id(geometry, method, distance, k)
        n, ari = ag._aligned_ari(payloads[row], {"protein_id": np.asarray(ids), "label": labels})
        rows.append(ag._agreement("scipy_vs_sklearn", row, f"sklearn:{row}", n, ari, k=k))
    return rows


def row_order_check(X, ids, geometry, method, distance, payloads, k_values=K_VALUES) -> list[dict]:
    shuffle = np.random.default_rng(ag.ORDER_CHECK_SEED).permutation(len(ids))
    Z = ag.fit_tree(X[shuffle], method, distance)
    rows = []
    for k in k_values:
        labels = ag._labels_in_input_order(ag.cut(Z, k), shuffle)
        row = partition_id(geometry, method, distance, k)
        n, ari = ag._aligned_ari(payloads[row], {"protein_id": np.asarray(ids), "label": labels})
        rows.append(ag._agreement("row_order", row, f"shuffled:{row}", n, ari, k=k))
    return rows


def prott5_agreement(payloads, prott5_assignments: Path | None = None) -> list[dict]:
    prott5_assignments = prott5_assignments or PROTT5_ASSIGNMENTS
    rows = []
    for (geometry, method, distance), prott5_tree in PROTT5_TREE.items():
        k = reported_k(geometry, distance)
        pairs = (("prott5_vs_esm2_reported", PROTT5_REPORTED_K, k),
                 ("prott5_vs_esm2_matched_k", k, k),
                 ("prott5_vs_esm2_matched_k", PROTT5_REPORTED_K, PROTT5_REPORTED_K))
        for check, prott5_k, esm2_k in pairs:
            name = f"{prott5_tree}-K{prott5_k:02d}"
            with np.load(prott5_assignments / f"{name}.npz", allow_pickle=False) as z:
                other = {"protein_id": z["protein_id"].astype(str), "label": z["label"]}
            row = partition_id(geometry, method, distance, esm2_k)
            n, ari = ag._aligned_ari(other, payloads[row])
            rows.append(ag._agreement(check, name, row, n, ari, k=esm2_k, prott5_k=prott5_k))
    return rows


def build_all(coordinates, ids, k_values=K_VALUES, checks: bool = True, progress: bool = False):
    runs, clusters, agreement, payloads, trees = [], [], [], {}, {}
    for geometry, method, distance in TREE_ARMS:
        started = perf_counter()
        X = coordinates[geometry]
        Z, order, r, c, p = fit_arm(X, ids, geometry, method, distance, k_values)
        trees[tree_id(geometry, method, distance)] = {"Z": Z, "protein_id": np.asarray(ids, dtype=str)[order]}
        coph = float(cophenet(Z, pdist(X[order], metric=distance))[0])
        for row in r:
            row["cophenetic_r"] = coph
            if row["reported_partition"]:
                labels = p[row["row"]]["label"]
                row.update({"silhouette_n_scored": len(labels), "calinski_harabasz_n_scored": len(labels),
                            "davies_bouldin_n_scored": len(labels),
                            **ag._internal_scores(X, labels, distance),
                            **ag._control_summaries(X, labels, distance)})
        runs.extend(r)
        clusters.extend(c)
        payloads.update(p)
        if checks:
            agreement.extend(sklearn_check(X, ids, geometry, method, distance, payloads, k_values))
            agreement.extend(row_order_check(X, ids, geometry, method, distance, payloads, k_values))
        if progress:
            print(f"{tree_id(geometry, method, distance)}: {perf_counter() - started:.1f}s", flush=True)
    if checks:
        agreement.extend(prott5_agreement(payloads))
    return pd.DataFrame(runs), pd.DataFrame(clusters), pd.DataFrame(agreement), payloads, trees


def with_provenance(runs, clusters, agreement, model) -> dict[str, pd.DataFrame]:
    common = {"dataset": ag.DATASET, "embedding": MODEL_STEM, "embedding_checkpoint": model.checkpoint,
              "embedding_revision": model.revision, "embedding_sha256": model.manifest["artifact"]["sha256"],
              "records": RECORDS, "subset_hash": model.data.subset_hash, "git_sha": ag._git_sha(),
              **ag.LIBRARY_VERSIONS}
    runs = runs.copy()
    runs["space"] = runs["geometry"]
    runs["n_dims"] = WIDTH
    runs["input_normalisation"] = runs["geometry"].map(lambda g: "l2" if g.startswith("l2") else "none")
    runs["clustering_metric"] = runs["distance"]
    runs["scoring_metric"] = runs["distance"].map(
        lambda d: f"silhouette {d}; calinski_harabasz and davies_bouldin euclidean")
    runs["seed"] = "n/a, deterministic"
    runs["control_permutations"] = ag.CONTROL_PERMUTATIONS
    runs["control_seed"] = ag.CONTROL_SEED
    runs["kmeans_k_source"] = runs.apply(lambda r: KMEANS_GEOMETRY_FOR[(r["geometry"], r["distance"])], axis=1)
    runs["archive"] = runs["row"] + ".npz"
    return {"runs": runs.assign(**common), "clusters": clusters.assign(**common),
            "agreement": agreement.assign(**common)}


def reset_outputs(paths: OutputPaths) -> None:
    for directory in (paths.runs.parent, paths.assignments, paths.trees):
        if directory.exists():
            shutil.rmtree(directory)
        directory.mkdir(parents=True)


def validate_reopened(tables, paths: OutputPaths, ids: list[str], expected_partitions: int = EXPECTED_PARTITIONS
                      ) -> dict[str, int]:
    runs, agreement = tables["runs"], tables["agreement"]
    archives = sorted(p.name for p in paths.assignments.glob("*.npz"))
    if len(runs) != expected_partitions or len(archives) != expected_partitions:
        raise ValueError(f"expected {expected_partitions} partitions, got {len(runs)} rows, {len(archives)} archives")
    if set(archives) != set(runs["archive"]) or runs["row"].duplicated().any():
        raise ValueError("archive names do not match run rows")
    for archive in paths.assignments.glob("*.npz"):
        with np.load(archive, allow_pickle=False) as z:
            if set(z.files) != {"protein_id", "label"} or list(z["protein_id"].astype(str)) != list(ids):
                raise ValueError(f"{archive.name} breaks the partition contract")
    trees = sorted(paths.trees.glob("*.npz"))
    if len(trees) != len(TREE_ARMS):
        raise ValueError(f"expected {len(TREE_ARMS)} trees, got {len(trees)}")
    for tree in trees:
        with np.load(tree, allow_pickle=False) as z:
            if set(z.files) != {"Z", "protein_id"} or len(z["Z"]) != len(z["protein_id"]) - 1:
                raise ValueError(f"{tree.name} breaks the tree contract")
    sizes = tables["clusters"].groupby("row")["size"].sum()
    if not sizes.reindex(runs["row"]).to_numpy().tolist() == runs["n"].tolist():
        raise ValueError("cluster sizes do not add up to n")
    for name, table in tables.items():
        for column in ("dataset", "embedding_sha256", "subset_hash", "git_sha", "scipy_version"):
            if table[column].isna().any():
                raise ValueError(f"provenance column {column} has gaps in {name}")
    return {"partitions": len(runs), "trees": len(trees), "agreement_rows": len(agreement)}


def check_expectations(tables) -> list[str]:
    agreement, runs = tables["agreement"], tables["runs"]
    failures = []
    for check in ("scipy_vs_sklearn", "row_order"):
        bad = agreement[(agreement["check"] == check) & (agreement["ari"] != 1.0)]
        if not bad.empty:
            failures.append(f"{check}: {len(bad)} comparisons below 1.0, first {bad.iloc[0]['left']}")
    if runs["cluster_count_shortfall"].astype(bool).any():
        failures.append("a partition realised fewer clusters than requested")
    return failures


def persist(tables, payloads, trees, paths: OutputPaths, ids, expected_partitions=EXPECTED_PARTITIONS):
    reset_outputs(paths)
    for name, path in (("runs", paths.runs), ("clusters", paths.clusters), ("agreement", paths.agreement)):
        tables[name].to_csv(path, index=False)
    for row, payload in payloads.items():
        np.savez_compressed(paths.assignments / f"{row}.npz", protein_id=np.asarray(payload["protein_id"], dtype=str),
                            label=np.asarray(payload["label"], dtype=np.int32))
    for name, tree in trees.items():
        np.savez_compressed(paths.trees / f"{name}.npz", Z=tree["Z"], protein_id=np.asarray(tree["protein_id"], dtype=str))
    reopened = {name: pd.read_csv(path) for name, path in
                (("runs", paths.runs), ("clusters", paths.clusters), ("agreement", paths.agreement))}
    return reopened, validate_reopened(reopened, paths, ids, expected_partitions)


def run_scientific(root: Path | None = None) -> dict[str, int]:
    ag._require_versions()
    check_modal_k()
    paths = output_paths(root)
    reset_outputs(paths)
    print(f"Cleared {paths.root}; loading {MODEL_STEM}.", flush=True)
    model = esm2_kmeans.load_esm2()
    ids = list(model.data.ids)
    coordinates = esm2_kmeans.spaces(model.data.X)
    runs, clusters, agreement, payloads, trees = build_all(coordinates, ids, progress=True)
    tables = with_provenance(runs, clusters, agreement, model)
    reopened, counts = persist(tables, payloads, trees, paths, ids)
    print(f"Output validation passed: {counts}", flush=True)
    print(reopened["agreement"].groupby("check")["ari"].agg(["count", "min", "median", "max"]).round(4).to_string(), flush=True)
    failures = check_expectations(reopened)
    if failures:
        raise RuntimeError("checks failed, stop and decide: " + "; ".join(failures))
    print("Checks passed.", flush=True)
    return counts


def preflight() -> None:
    ag._require_versions()
    modal = check_modal_k()
    model = esm2_kmeans.load_esm2()
    coordinates = esm2_kmeans.spaces(model.data.X)
    geometry, method, distance = ELIGIBLE_TREE
    started = perf_counter()
    _, _, runs, _, _ = fit_arm(coordinates[geometry], list(model.data.ids), geometry, method, distance,
                               (reported_k(geometry, distance),))
    row = runs[0]
    print(f"ESM-2 k-means modal K: {modal}")
    print(f"{row['row']}: realised K {row['realised_k']}, largest cluster {row['largest_cluster_share']:.3f}, "
          f"{perf_counter() - started:.1f}s")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-scientific", action="store_true",
                        help="run everything and replace results/agglomerative/esm2_t36_3B_UR50D/")
    arguments = parser.parse_args(argv)
    if arguments.run_scientific:
        run_scientific()
    else:
        preflight()


if __name__ == "__main__":
    main()
