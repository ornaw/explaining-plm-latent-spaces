from __future__ import annotations

import argparse
import importlib.metadata
import shutil
import subprocess
import sys
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import cophenet, dendrogram, fcluster, linkage
from scipy.spatial.distance import pdist
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.preprocessing import normalize

from data import REPO, Dataset, deduplicate, load_dataset


DATASET = "phosphatase"
EXPECTED_VERSIONS = {"scipy": "1.18.0", "scikit-learn": "1.9.0"}
EXPECTED_FULL_N = 1_587
EXPECTED_DEDUP_N = 1_480
PCA_DIMS = (113, 67, 34)
GEOMETRIES = ("raw1024", "l2_raw1024", "pca113", "pca67", "pca34")
K_VALUES = tuple(range(2, 51))
CONTROL_PERMUTATIONS = 5
CONTROL_SEED = 0
ORDER_CHECK_SEED = 0
SMALL_CLUSTER = 5
DENDROGRAM_LEAVES = 30

TREE_ARMS = (
    ("raw1024", "single", "cosine"),
    ("raw1024", "average", "cosine"),
    ("raw1024", "complete", "cosine"),
    ("raw1024", "weighted", "cosine"),
    ("raw1024", "single", "euclidean"),
    ("raw1024", "average", "euclidean"),
    ("raw1024", "complete", "euclidean"),
    ("raw1024", "ward", "euclidean"),
    ("l2_raw1024", "ward", "euclidean"),
)
PCA_LINKAGES = (
    ("single", "cosine"),
    ("average", "cosine"),
    ("complete", "cosine"),
    ("weighted", "cosine"),
    ("single", "euclidean"),
    ("average", "euclidean"),
    ("complete", "euclidean"),
    ("ward", "euclidean"),
)
KMEANS_MODAL_K = {"raw1024": 14, "l2_raw1024": 15}
KMEANS_GEOMETRY_FOR = {
    ("raw1024", "cosine"): "l2_raw1024",
    ("raw1024", "euclidean"): "raw1024",
    ("l2_raw1024", "euclidean"): "l2_raw1024",
}
SKLEARN_LINKAGES = {"single", "average", "complete", "ward"}

GEOMETRY_CODES = {
    "raw1024": "R1024",
    "l2_raw1024": "L2R1024",
    "pca113": "P113",
    "pca67": "P067",
    "pca34": "P034",
}
LINKAGE_CODES = {"single": "SI", "average": "AV", "complete": "CO", "weighted": "WE", "ward": "WA"}
DISTANCE_CODES = {"cosine": "C", "euclidean": "E"}
RECORD_CODES = {"full": "F", "deduplicated": "D"}
DIMENSIONS = {"raw1024": 1_024, "l2_raw1024": 1_024, "pca113": 113, "pca67": 67, "pca34": 34}

EXPECTED_SWEEP_PARTITIONS = 882
EXPECTED_PCA_PARTITIONS = 48
EXPECTED_PARTITIONS = 930
EXPECTED_TREES = 18
if (EXPECTED_SWEEP_PARTITIONS != len(TREE_ARMS) * len(RECORD_CODES) * len(K_VALUES)
        or EXPECTED_PCA_PARTITIONS != len(PCA_DIMS) * len(PCA_LINKAGES) * len(RECORD_CODES)
        or EXPECTED_PARTITIONS != EXPECTED_SWEEP_PARTITIONS + EXPECTED_PCA_PARTITIONS
        or EXPECTED_TREES != len(TREE_ARMS) * len(RECORD_CODES)):
    raise RuntimeError("the grid axes no longer match the planned counts")

ELIGIBLE_ROWS = ("AG-R1024-COC-F-K15",)

LIBRARY_VERSIONS = {
    "numpy_version": importlib.metadata.version("numpy"),
    "pandas_version": importlib.metadata.version("pandas"),
    "scipy_version": importlib.metadata.version("scipy"),
    "sklearn_version": importlib.metadata.version("scikit-learn"),
    "matplotlib_version": importlib.metadata.version("matplotlib"),
}


@dataclass(frozen=True)
class OutputPaths:
    root: Path
    runs: Path
    clusters: Path
    agreement: Path
    assignments: Path
    trees: Path
    dendrogram_figure: Path
    cophenetic_figure: Path


def _output_paths(root: Path | None = None) -> OutputPaths:
    result_root = root if root is not None else REPO / "results" / "agglomerative"
    metrics = result_root / "metrics"
    figures = result_root / "figures"
    return OutputPaths(
        root=result_root,
        runs=metrics / "agglomerative_standalone_runs.csv",
        clusters=metrics / "agglomerative_standalone_clusters.csv",
        agreement=metrics / "agglomerative_standalone_partition_agreement.csv",
        assignments=result_root / "assignments" / "agglomerative_standalone",
        trees=result_root / "trees",
        dendrogram_figure=figures / "agglomerative_standalone_dendrogram.png",
        cophenetic_figure=figures / "agglomerative_standalone_cophenetic.png",
    )


def _git_sha() -> str:
    try:
        revision = subprocess.run(
            ["git", "describe", "--always"],
            cwd=REPO, capture_output=True, text=True, check=True,
        )
        status = subprocess.run(
            ["git", "status", "--porcelain", "--", ".", ":(exclude)results/agglomerative/**"],
            cwd=REPO, capture_output=True, text=True, check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    git_sha = revision.stdout.strip() or "unknown"
    return f"{git_sha}-dirty" if status.stdout.strip() else git_sha


def _require_versions() -> None:
    for package, expected in EXPECTED_VERSIONS.items():
        installed = importlib.metadata.version(package)
        if installed != expected:
            raise RuntimeError(f"agglomerative requires {package} {expected}, found {installed}")


def _deduplicated_indices(full_ids: list[str], deduplicated_ids: list[str]) -> np.ndarray:
    full_index = pd.Index(full_ids)
    if full_index.has_duplicates:
        raise ValueError("full protein IDs are not unique")
    positions = full_index.get_indexer(pd.Index(deduplicated_ids))
    if (positions < 0).any():
        raise ValueError("a deduplicated protein is absent from the full dataset")
    return positions


def _spaces(
    full: Dataset,
    deduplicated: Dataset,
) -> tuple[dict[str, dict[str, np.ndarray]], dict[int, float]]:
    raw = np.ascontiguousarray(full.X, dtype=np.float64)
    if raw.shape[1] != 1_024 or not np.isfinite(raw).all():
        raise ValueError(f"expected finite 1,024-d coordinates, got {raw.shape}")
    pca = PCA(n_components=max(PCA_DIMS), svd_solver="full", whiten=False)
    transformed = pca.fit_transform(raw)
    full_spaces = {
        "raw1024": raw,
        "l2_raw1024": normalize(raw, norm="l2", axis=1, copy=True),
        **{f"pca{d}": np.ascontiguousarray(transformed[:, :d]) for d in PCA_DIMS},
    }
    retained = {d: float(pca.explained_variance_ratio_[:d].sum()) for d in PCA_DIMS}
    positions = _deduplicated_indices(full.ids, deduplicated.ids)
    if not np.array_equal(full.X[positions], deduplicated.X):
        raise ValueError("deduplicated coordinates do not match the full matrix")
    spaces = {
        "full": full_spaces,
        "deduplicated": {
            geometry: np.ascontiguousarray(coordinates[positions], dtype=np.float64)
            for geometry, coordinates in full_spaces.items()
        },
    }
    for records, by_geometry in spaces.items():
        for geometry, coordinates in by_geometry.items():
            if coordinates.shape[1] != DIMENSIONS[geometry] or not np.isfinite(coordinates).all():
                raise ValueError(f"bad coordinates for {records}/{geometry}")
        if not np.allclose(np.linalg.norm(by_geometry["l2_raw1024"], axis=1), 1.0):
            raise ValueError(f"{records} unit-length coordinates are not unit length")
    return spaces, retained


def _check_modal_k(selection_csv: Path | None = None) -> dict[str, int]:
    path = selection_csv or REPO / "results/kmeans/metrics/kmeans_standalone_elbow_selection.csv"
    selection = pd.read_csv(path)
    primary = selection[selection["is_primary"].astype(bool) & (selection["records"] == "full")]
    modal = {}
    for geometry in KMEANS_MODAL_K:
        values = primary.loc[primary["geometry"] == geometry, "selected_k"]
        if values.empty:
            raise ValueError(f"no primary k-means selections for {geometry}")
        modal[geometry] = int(min(values.mode()))
    if modal != KMEANS_MODAL_K:
        raise ValueError(f"k-means modal K changed: table gives {modal}, code has {KMEANS_MODAL_K}")
    return modal


def reported_k(geometry: str, distance: str) -> int:
    if geometry.startswith("pca"):
        geometry = "raw1024"
    return KMEANS_MODAL_K[KMEANS_GEOMETRY_FOR[(geometry, distance)]]


def tree_id(geometry: str, method: str, distance: str, records: str) -> str:
    return (f"AG-{GEOMETRY_CODES[geometry]}-{LINKAGE_CODES[method]}{DISTANCE_CODES[distance]}"
            f"-{RECORD_CODES[records]}")


def partition_id(geometry: str, method: str, distance: str, records: str, k: int) -> str:
    if k < 2:
        raise ValueError("a partition needs at least two clusters")
    return f"{tree_id(geometry, method, distance, records)}-K{k:02d}"


def _canonical_order(ids: list[str]) -> np.ndarray:
    if len(ids) != len(set(ids)):
        raise ValueError("protein IDs must be unique")
    return np.argsort(np.asarray(ids), kind="stable")


def fit_tree(X: np.ndarray, method: str, distance: str) -> np.ndarray:
    if method == "ward" and distance != "euclidean":
        raise ValueError("ward is only defined for Euclidean distance")
    Z = linkage(np.ascontiguousarray(X, dtype=np.float64), method=method, metric=distance)
    if Z.shape != (len(X) - 1, 4) or not np.isfinite(Z).all():
        raise ValueError("linkage returned an unexpected matrix")
    return Z


def cut(Z: np.ndarray, k: int) -> np.ndarray:
    raw = fcluster(Z, t=k, criterion="maxclust")
    _, labels = np.unique(raw, return_inverse=True)
    return labels.astype(np.int32)


def _labels_in_input_order(labels_canonical: np.ndarray, order: np.ndarray) -> np.ndarray:
    labels = np.empty_like(labels_canonical)
    labels[order] = labels_canonical
    return labels


def _internal_scores(X: np.ndarray, labels: np.ndarray, distance: str) -> dict[str, float]:
    scores = {
        "silhouette": float(silhouette_score(X, labels, metric=distance)),
        "calinski_harabasz": float(calinski_harabasz_score(X, labels)),
        "davies_bouldin": float(davies_bouldin_score(X, labels)),
    }
    if not all(np.isfinite(v) for v in scores.values()):
        raise ValueError("internal scores must be finite")
    return scores


def _control_summaries(X: np.ndarray, labels: np.ndarray, distance: str) -> dict[str, float]:
    rng = np.random.default_rng(CONTROL_SEED)
    controls = [_internal_scores(X, rng.permutation(labels), distance)
                for _ in range(CONTROL_PERMUTATIONS)]
    out = {}
    for metric in ("silhouette", "calinski_harabasz", "davies_bouldin"):
        values = np.asarray([c[metric] for c in controls])
        out.update({
            f"control_{metric}_minimum": float(values.min()),
            f"control_{metric}_median": float(np.median(values)),
            f"control_{metric}_maximum": float(values.max()),
        })
    return out


def _size_summary(labels: np.ndarray) -> dict[str, float | int]:
    _, counts = np.unique(labels, return_counts=True)
    return {
        "largest_cluster_share": float(counts.max() / len(labels)),
        "clusters_under_5": int((counts < SMALL_CLUSTER).sum()),
        "singleton_clusters": int((counts == 1).sum()),
    }


def _cluster_rows(row: str, labels: np.ndarray) -> list[dict[str, object]]:
    clusters, counts = np.unique(labels, return_counts=True)
    return [{"row": row, "cluster": int(c), "size": int(n), "fraction": float(n / len(labels))}
            for c, n in zip(clusters, counts, strict=True)]


def _partition_row(
    geometry: str, method: str, distance: str, records: str, k: int, labels: np.ndarray,
    role: str,
) -> dict[str, object]:
    realised = len(np.unique(labels))
    return {
        "row": partition_id(geometry, method, distance, records, k),
        "tree": tree_id(geometry, method, distance, records),
        "geometry": geometry,
        "linkage": method,
        "distance": distance,
        "records": records,
        "role": role,
        "requested_k": k,
        "realised_k": realised,
        "cluster_count_shortfall": realised != k,
        "reported_partition": k == reported_k(geometry, distance) and role == "cut_sweep",
        "eligible_cross_method": partition_id(geometry, method, distance, records, k)
        in ELIGIBLE_ROWS,
        "n": len(labels),
        **_size_summary(labels),
    }


def fit_arm(
    X: np.ndarray, ids: list[str], geometry: str, method: str, distance: str, records: str,
    k_values: tuple[int, ...], role: str,
) -> tuple[np.ndarray, np.ndarray, list[dict], list[dict], dict[str, np.ndarray]]:
    order = _canonical_order(ids)
    Z = fit_tree(X[order], method, distance)
    runs, clusters, payloads = [], [], {}
    for k in k_values:
        labels = _labels_in_input_order(cut(Z, k), order)
        run = _partition_row(geometry, method, distance, records, k, labels, role)
        runs.append(run)
        clusters.extend(_cluster_rows(run["row"], labels))
        payloads[run["row"]] = {"protein_id": np.asarray(ids, dtype=str), "label": labels}
    return Z, order, runs, clusters, payloads


def _aligned_ari(left: dict[str, np.ndarray], right: dict[str, np.ndarray]) -> tuple[int, float]:
    frames = []
    for side, payload in (("left", left), ("right", right)):
        frames.append(pd.DataFrame({"protein_id": np.asarray(payload["protein_id"], dtype=str),
                                    side: np.asarray(payload["label"])}))
    aligned = frames[0].merge(frames[1], on="protein_id", how="inner", validate="one_to_one")
    aligned = aligned[(aligned["left"] >= 0) & (aligned["right"] >= 0)]
    if aligned.empty:
        raise ValueError("partitions share no clustered proteins")
    return len(aligned), float(adjusted_rand_score(aligned["left"], aligned["right"]))


def _agreement(check: str, left: str, right: str, n: int, ari: float, **extra) -> dict[str, object]:
    return {"check": check, "left": left, "right": right, "n_compared": n, "ari": ari, **extra}


def _sklearn_check(
    X: np.ndarray, ids: list[str], geometry: str, method: str, distance: str, records: str,
    payloads: dict[str, dict[str, np.ndarray]], k_values: tuple[int, ...],
) -> list[dict]:
    order = _canonical_order(ids)
    rows = []
    for k in k_values:
        model = AgglomerativeClustering(n_clusters=k, linkage=method, metric=distance)
        labels = _labels_in_input_order(model.fit_predict(X[order]).astype(np.int32), order)
        row = partition_id(geometry, method, distance, records, k)
        n, ari = _aligned_ari(payloads[row], {"protein_id": np.asarray(ids), "label": labels})
        rows.append(_agreement("scipy_vs_sklearn", row, f"sklearn:{row}", n, ari, k=k))
    return rows


def _row_order_check(
    X: np.ndarray, ids: list[str], geometry: str, method: str, distance: str, records: str,
    payloads: dict[str, dict[str, np.ndarray]], k_values: tuple[int, ...],
) -> list[dict]:
    shuffle = np.random.default_rng(ORDER_CHECK_SEED).permutation(len(ids))
    Z = fit_tree(X[shuffle], method, distance)
    rows = []
    for k in k_values:
        labels = _labels_in_input_order(cut(Z, k), shuffle)
        row = partition_id(geometry, method, distance, records, k)
        n, ari = _aligned_ari(payloads[row], {"protein_id": np.asarray(ids), "label": labels})
        rows.append(_agreement("row_order", row, f"shuffled:{row}", n, ari, k=k))
    return rows


def _unit_length_check(
    spaces: dict[str, np.ndarray], ids: list[str], records: str, k_values: tuple[int, ...],
) -> list[dict]:
    order = _canonical_order(ids)
    rows = []
    for method in ("single", "complete", "average"):
        Zc = fit_tree(spaces["raw1024"][order], method, "cosine")
        Ze = fit_tree(spaces["l2_raw1024"][order], method, "euclidean")
        for k in k_values:
            left = {"protein_id": np.asarray(ids), "label": _labels_in_input_order(cut(Zc, k), order)}
            right = {"protein_id": np.asarray(ids), "label": _labels_in_input_order(cut(Ze, k), order)}
            n, ari = _aligned_ari(left, right)
            rows.append(_agreement(
                "unit_length_equivalence",
                partition_id("raw1024", method, "cosine", records, k),
                f"{method} euclidean on l2_raw1024, K{k:02d}", n, ari, k=k,
            ))
    return rows


def _load_kmeans(row: str, root: Path | None = None) -> dict[str, np.ndarray]:
    base = root or REPO / "results/kmeans/assignments/kmeans_standalone"
    with np.load(base / f"{row}.npz") as z:
        return {"protein_id": z["protein_id"].astype(str), "label": z["label"]}


SCREEN_KMEANS_ROWS = tuple(f"KM-L2R1024-F-S{s}-K15-K" for s in ("00", "01", "02", "04"))


def _screen_check(payloads: dict[str, dict[str, np.ndarray]], kmeans_root: Path | None = None
                  ) -> list[dict]:
    rows = []
    for ag_row in ("AG-R1024-COC-F-K15", "AG-L2R1024-WAE-F-K15"):
        for km_row in SCREEN_KMEANS_ROWS:
            n, ari = _aligned_ari(payloads[ag_row], _load_kmeans(km_row, kmeans_root))
            rows.append(_agreement("screen_reproduction", ag_row, km_row, n, ari, k=15))
    return rows


def _full_vs_dedup(payloads: dict[str, dict[str, np.ndarray]]) -> list[dict]:
    rows = []
    for geometry, method, distance in TREE_ARMS:
        k = reported_k(geometry, distance)
        left = partition_id(geometry, method, distance, "full", k)
        right = partition_id(geometry, method, distance, "deduplicated", k)
        n, ari = _aligned_ari(payloads[left], payloads[right])
        rows.append(_agreement("full_vs_deduplicated", left, right, n, ari, k=k))
    return rows


def _pca_vs_raw(payloads: dict[str, dict[str, np.ndarray]]) -> list[dict]:
    rows = []
    for d in PCA_DIMS:
        for method, distance in PCA_LINKAGES:
            k = reported_k("raw1024", distance)
            for records in RECORD_CODES:
                left = partition_id(f"pca{d}", method, distance, records, k)
                right = partition_id("raw1024", method, distance, records, k)
                n, ari = _aligned_ari(payloads[left], payloads[right])
                rows.append(_agreement("pca_vs_raw", left, right, n, ari, k=k))
    return rows


def build_all(
    spaces: dict[str, dict[str, np.ndarray]],
    ids_by_records: dict[str, list[str]],
    k_values: tuple[int, ...] = K_VALUES,
    kmeans_root: Path | None = None,
    screen: bool = True,
    progress: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict, dict]:
    runs, clusters, agreement, payloads, trees = [], [], [], {}, {}
    for records, ids in ids_by_records.items():
        for geometry, method, distance in TREE_ARMS:
            start = perf_counter()
            X = spaces[records][geometry]
            Z, order, r, c, p = fit_arm(X, ids, geometry, method, distance, records,
                                         k_values, "cut_sweep")
            trees[tree_id(geometry, method, distance, records)] = {
                "Z": Z, "protein_id": np.asarray(ids, dtype=str)[order],
            }
            coph = float(cophenet(Z, pdist(X[order], metric=distance))[0])
            for row in r:
                row["cophenetic_r"] = coph
                if row["reported_partition"]:
                    labels = p[row["row"]]["label"]
                    row.update({
                        "silhouette_n_scored": len(labels),
                        "calinski_harabasz_n_scored": len(labels),
                        "davies_bouldin_n_scored": len(labels),
                        **_internal_scores(X, labels, distance),
                        **_control_summaries(X, labels, distance),
                    })
            runs.extend(r)
            clusters.extend(c)
            payloads.update(p)
            if method in SKLEARN_LINKAGES:
                agreement.extend(_sklearn_check(X, ids, geometry, method, distance, records,
                                                payloads, k_values))
            agreement.extend(_row_order_check(X, ids, geometry, method, distance, records,
                                              payloads, k_values))
            if progress:
                print(f"{tree_id(geometry, method, distance, records)}: "
                      f"{perf_counter() - start:.1f}s", flush=True)
        agreement.extend(_unit_length_check(spaces[records], ids, records, k_values))
        for d in PCA_DIMS:
            for method, distance in PCA_LINKAGES:
                k = reported_k("raw1024", distance)
                _, _, r, c, p = fit_arm(spaces[records][f"pca{d}"], ids, f"pca{d}", method,
                                        distance, records, (k,), "pca_sensitivity")
                runs.extend(r)
                clusters.extend(c)
                payloads.update(p)
    agreement.extend(_full_vs_dedup(payloads))
    agreement.extend(_pca_vs_raw(payloads))
    if screen:
        agreement.extend(_screen_check(payloads, kmeans_root))
    return (pd.DataFrame(runs), pd.DataFrame(clusters), pd.DataFrame(agreement), payloads, trees)


def _with_provenance(
    runs: pd.DataFrame, clusters: pd.DataFrame, agreement: pd.DataFrame,
    datasets: dict[str, Dataset], retained: dict[int, float],
) -> dict[str, pd.DataFrame]:
    common = {"dataset": DATASET, "git_sha": _git_sha(), **LIBRARY_VERSIONS}
    runs = runs.copy()
    runs["subset_hash"] = runs["records"].map({r: d.subset_hash for r, d in datasets.items()})
    runs["space"] = runs["geometry"]
    runs["n_dims"] = runs["geometry"].map(DIMENSIONS)
    runs["variance_retained"] = runs["geometry"].map(
        {f"pca{d}": v for d, v in retained.items()})
    runs["input_normalisation"] = runs["geometry"].map(
        lambda g: "l2" if g == "l2_raw1024" else "none")
    runs["clustering_metric"] = runs["distance"]
    runs["scoring_metric"] = runs["distance"].map(
        lambda d: f"silhouette {d}; calinski_harabasz and davies_bouldin euclidean")
    runs["seed"] = "n/a, deterministic"
    runs["control_permutations"] = CONTROL_PERMUTATIONS
    runs["control_seed"] = CONTROL_SEED
    runs["archive"] = runs["row"].map(lambda row: f"{row}.npz")
    runs = runs.assign(**common)
    meta = runs.set_index("row")[["subset_hash"]]
    clusters = clusters.join(meta, on="row").assign(**common)
    agreement = agreement.assign(**common)
    return {"runs": runs, "clusters": clusters, "agreement": agreement}


def _reset_outputs(paths: OutputPaths) -> None:
    for directory in (paths.runs.parent, paths.dendrogram_figure.parent, paths.assignments,
                      paths.trees):
        if directory.exists():
            shutil.rmtree(directory)
        directory.mkdir(parents=True)


def _plot_dendrograms(trees: dict, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, name in zip(axes, ("AG-R1024-COC-F", "AG-R1024-AVC-F"), strict=True):
        dendrogram(trees[name]["Z"], truncate_mode="lastp", p=DENDROGRAM_LEAVES, ax=ax,
                   no_labels=True, color_threshold=0)
        ax.set_title(f"{name}: top {DENDROGRAM_LEAVES} branches")
        ax.set_ylabel("merge height (cosine distance)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_cophenetic(runs: pd.DataFrame, path: Path) -> None:
    trees = (runs[runs["role"] == "cut_sweep"]
             .drop_duplicates("tree")[["tree", "records", "cophenetic_r"]])
    fig, ax = plt.subplots(figsize=(8, 4.5))
    full = trees[trees["records"] == "full"]
    ax.barh(full["tree"], full["cophenetic_r"])
    ax.set_xlabel("cophenetic correlation (full records)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _validate_reopened(tables: dict[str, pd.DataFrame], paths: OutputPaths,
                       expected_partitions: int, expected_trees: int) -> dict[str, int]:
    runs = tables["runs"]
    archives = sorted(p.name for p in paths.assignments.glob("*.npz"))
    if len(runs) != expected_partitions or len(archives) != expected_partitions:
        raise ValueError(f"expected {expected_partitions} partitions, got {len(runs)} rows "
                         f"and {len(archives)} archives")
    if set(archives) != set(runs["archive"]) or runs["row"].duplicated().any():
        raise ValueError("archive names do not match run-table rows")
    trees = sorted(paths.trees.glob("*.npz"))
    if len(trees) != expected_trees:
        raise ValueError(f"expected {expected_trees} trees, got {len(trees)}")
    ids_by_records = {}
    for archive in paths.assignments.glob("*.npz"):
        with np.load(archive) as z:
            if set(z.files) != {"protein_id", "label"} or z["protein_id"].shape != z["label"].shape:
                raise ValueError(f"{archive.name} breaks the partition contract")
            records = archive.stem.split("-")[3]
            ids = tuple(z["protein_id"].astype(str))
            if ids_by_records.setdefault(records, ids) != ids:
                raise ValueError(f"{archive.name} holds a different protein list from its records")
    for tree in trees:
        with np.load(tree) as z:
            if set(z.files) != {"Z", "protein_id"} or len(z["Z"]) != len(z["protein_id"]) - 1:
                raise ValueError(f"{tree.name} breaks the tree contract")
    for column in ("dataset", "subset_hash", "git_sha", "scipy_version", "sklearn_version"):
        if runs[column].isna().any():
            raise ValueError(f"provenance column {column} has gaps")
    sizes = tables["clusters"].groupby("row")["size"].sum()
    n_by_row = runs.set_index("row")["n"]
    if set(sizes.index) != set(n_by_row.index) or not np.array_equal(
            sizes.to_numpy(), n_by_row.loc[sizes.index].to_numpy()):
        raise ValueError("cluster sizes do not add up to n")
    return {"partitions": len(runs), "trees": len(trees),
            "agreement_rows": len(tables["agreement"])}


def check_expectations(tables: dict[str, pd.DataFrame]) -> list[str]:
    agreement, runs = tables["agreement"], tables["runs"]
    failures = []
    for check in ("scipy_vs_sklearn", "row_order"):
        bad = agreement[(agreement["check"] == check) & (agreement["ari"] != 1.0)]
        if not bad.empty:
            failures.append(f"{check}: {len(bad)} comparisons below 1.0, first {bad.iloc[0]['left']}")
    unit = agreement[(agreement["check"] == "unit_length_equivalence")
                     & agreement["left"].str.contains(r"-(?:SI|CO)C-", regex=True)]
    if (unit["ari"] != 1.0).any():
        failures.append("unit_length_equivalence: single or complete differs from cosine")
    if runs["cluster_count_shortfall"].astype(bool).any():
        failures.append(f"{int(runs['cluster_count_shortfall'].astype(bool).sum())} partitions "
                        "realised fewer clusters than requested")
    return failures


def persist(tables: dict[str, pd.DataFrame], payloads: dict, trees: dict, paths: OutputPaths,
            expected_partitions: int = EXPECTED_PARTITIONS,
            expected_trees: int = EXPECTED_TREES) -> dict[str, int]:
    _reset_outputs(paths)
    for name, path in (("runs", paths.runs), ("clusters", paths.clusters),
                       ("agreement", paths.agreement)):
        tables[name].to_csv(path, index=False)
    for row, payload in payloads.items():
        np.savez_compressed(paths.assignments / f"{row}.npz",
                            protein_id=np.asarray(payload["protein_id"], dtype=str),
                            label=np.asarray(payload["label"], dtype=np.int32))
    for name, tree in trees.items():
        np.savez_compressed(paths.trees / f"{name}.npz", Z=tree["Z"],
                            protein_id=np.asarray(tree["protein_id"], dtype=str))
    reopened = {name: pd.read_csv(path) for name, path in (
        ("runs", paths.runs), ("clusters", paths.clusters), ("agreement", paths.agreement))}
    counts = _validate_reopened(reopened, paths, expected_partitions, expected_trees)
    if all(name in trees for name in ("AG-R1024-COC-F", "AG-R1024-AVC-F")):
        _plot_dendrograms(trees, paths.dendrogram_figure)
    _plot_cophenetic(reopened["runs"], paths.cophenetic_figure)
    return counts


def _summary(agreement: pd.DataFrame) -> pd.DataFrame:
    return (agreement.groupby("check")["ari"]
            .agg(["count", "min", "median", "max"]).round(4))


def _load() -> tuple[dict[str, Dataset], dict[str, dict[str, np.ndarray]], dict[int, float]]:
    full = load_dataset(DATASET)
    deduplicated = deduplicate(full)
    if (len(full), len(deduplicated)) != (EXPECTED_FULL_N, EXPECTED_DEDUP_N):
        raise ValueError(f"expected {EXPECTED_FULL_N} and {EXPECTED_DEDUP_N} records, got "
                         f"{len(full)} and {len(deduplicated)}")
    spaces, retained = _spaces(full, deduplicated)
    return {"full": full, "deduplicated": deduplicated}, spaces, retained


def run_scientific() -> dict[str, int]:
    _require_versions()
    _check_modal_k()
    paths = _output_paths()
    _reset_outputs(paths)
    print("Cleared owned agglomerative outputs; loading phosphatase.", flush=True)
    datasets, spaces, retained = _load()
    runs, clusters, agreement, payloads, trees = build_all(
        spaces, {r: d.ids for r, d in datasets.items()}, progress=True)
    tables = _with_provenance(runs, clusters, agreement, datasets, retained)
    counts = persist(tables, payloads, trees, paths)
    print(f"Output validation passed: {counts}", flush=True)
    print(_summary(tables["agreement"]).to_string(), flush=True)
    failures = check_expectations(tables)
    if failures:
        raise RuntimeError("step 4 checks failed, stop and decide: " + "; ".join(failures))
    print("Step 4 checks passed.", flush=True)
    return counts


def preflight() -> None:
    _require_versions()
    modal = _check_modal_k()
    datasets, spaces, retained = _load()
    ids = datasets["full"].ids
    start = perf_counter()
    Z, order, runs, clusters, payloads = fit_arm(
        spaces["full"]["raw1024"], ids, "raw1024", "complete", "cosine", "full", (15,),
        "cut_sweep")
    seconds = perf_counter() - start
    row = runs[0]
    print(f"modal K from the k-means table: {modal}")
    print(f"variance retained: {retained}")
    print(f"{row['row']}: realised K {row['realised_k']}, largest cluster "
          f"{row['largest_cluster_share']:.3f}, fit and cut {seconds:.2f}s")
    print(f"planned: {EXPECTED_PARTITIONS} partitions, {EXPECTED_TREES} trees")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-scientific", action="store_true",
                        help="run the full grid and replace results/agglomerative/ outputs")
    arguments = parser.parse_args(argv)
    if arguments.run_scientific:
        run_scientific()
    else:
        preflight()


if __name__ == "__main__":
    main()
