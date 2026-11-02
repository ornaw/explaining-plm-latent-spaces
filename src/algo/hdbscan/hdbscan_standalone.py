from __future__ import annotations

import importlib.metadata
import shutil
import subprocess
import sys
import textwrap
import warnings
from pathlib import Path
from typing import cast

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import hdbscan
import hdbscan.validity as hdbscan_validity
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from hdbscan.validity import validity_index
from matplotlib.axes import Axes
from matplotlib.image import AxesImage
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse
from seaborn.utils import relative_luminance
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, pairwise_distances, silhouette_score
from scipy.special import logsumexp

from data import REPO, deduplicate, load_dataset


DATASET = "phosphatase"
PCA_DIMS = (113, 67, 34)
PCA_SPACE_NAMES = tuple(f"pca{dimensions}" for dimensions in PCA_DIMS)
MIN_CLUSTER_SIZES = (5, 10, 20, 40)
MIN_SAMPLES_VALUES = (1, 5, 10, 20, 40)
BASELINE_MIN_SAMPLES = 5
COSINE_NORM_FLOOR = 1e-8
METRIC_ALGORITHMS = {"euclidean": "prims_kdtree", "cosine": "generic"}
EXPECTED_FULL_N = 1_587
EXPECTED_DEDUP_N = 1_480
LIBRARY_VERSIONS = {
    f"{column}_version": importlib.metadata.version(package)
    for package, column in (
        ("hdbscan", "hdbscan"),
        ("numpy", "numpy"),
        ("scikit-learn", "sklearn"),
        ("pandas", "pandas"),
        ("matplotlib", "matplotlib"),
        ("seaborn", "seaborn"),
    )
}



def _git_sha() -> str:
    try:
        done = subprocess.run(
            ["git", "describe", "--always", "--dirty"],
            cwd=REPO, capture_output=True, text=True, check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return done.stdout.strip() or "unknown"


GIT_SHA = _git_sha()

HDBSCAN_RESULTS_DIR = REPO / "results" / "hdbscan"
RUNS_PATH = HDBSCAN_RESULTS_DIR / "metrics" / "hdbscan_standalone_runs.csv"
CLUSTERS_PATH = HDBSCAN_RESULTS_DIR / "metrics" / "hdbscan_standalone_clusters.csv"
AGREEMENT_PATH = HDBSCAN_RESULTS_DIR / "metrics" / "hdbscan_standalone_partition_agreement.csv"
PAIRED_PATH = HDBSCAN_RESULTS_DIR / "metrics" / "hdbscan_standalone_min_samples_paired.csv"
WRITEUP_PATH = HDBSCAN_RESULTS_DIR / "hdbscan.md"
ASSIGNMENTS_DIR = HDBSCAN_RESULTS_DIR / "assignments" / "hdbscan_standalone_grid"
FIGURE_PATH = (
    HDBSCAN_RESULTS_DIR / "figures" / "hdbscan_standalone_min_samples_05_euclidean_grid.png"
)
METRIC_FIGURE_PATH = (
    HDBSCAN_RESULTS_DIR / "figures" / "hdbscan_standalone_min_samples_05_metric_heatmaps.png"
)
AGREEMENT_FIGURE_PATH = (
    HDBSCAN_RESULTS_DIR / "figures" / "hdbscan_standalone_min_samples_05_partition_agreement.png"
)
EUCLIDEAN_COSINE_COVERAGE_FIGURE_PATH = (
    HDBSCAN_RESULTS_DIR
    / "figures"
    / "hdbscan_standalone_min_samples_05_euclidean_cosine_clustered_coverage.png"
)
MIN_SAMPLES_FIGURE_PATH = (
    HDBSCAN_RESULTS_DIR / "figures" / "hdbscan_standalone_min_samples_noise.png"
)
MIN_SAMPLES_CLUSTERS_FIGURE_PATH = (
    HDBSCAN_RESULTS_DIR / "figures" / "hdbscan_standalone_min_samples_clusters.png"
)
MIN_SAMPLES_DBCV_FIGURE_PATH = (
    HDBSCAN_RESULTS_DIR / "figures" / "hdbscan_standalone_min_samples_dbcv.png"
)
CONDENSED_TREE_FIGURE_PATH = (
    HDBSCAN_RESULTS_DIR
    / "figures"
    / "hdbscan_standalone_deduplicated_raw1024_min_samples_05_condensed_trees.png"
)
CONDENSED_TREE_PANEL_DIR = (
    HDBSCAN_RESULTS_DIR / "figures" / "hdbscan_standalone_deduplicated_raw1024_condensed_tree_panels"
)
LEGACY_FIGURE_PATHS = (
    HDBSCAN_RESULTS_DIR / "figures" / "hdbscan_standalone_euclidean_grid.png",
    HDBSCAN_RESULTS_DIR / "figures" / "hdbscan_standalone_metric_heatmaps.png",
    HDBSCAN_RESULTS_DIR / "figures" / "hdbscan_standalone_partition_agreement.png",
    HDBSCAN_RESULTS_DIR
    / "figures"
    / "hdbscan_standalone_deduplicated_raw1024_condensed_trees.png",
)

BASE_GRID = (
    ("EA", "full", "raw1024", 1_024, "euclidean"),
    ("EB", "full", "pca113", 113, "euclidean"),
    ("EC", "full", "pca67", 67, "euclidean"),
    ("ED", "full", "pca34", 34, "euclidean"),
    ("EE", "deduplicated", "raw1024", 1_024, "euclidean"),
    ("EF", "deduplicated", "pca113", 113, "euclidean"),
    ("EG", "deduplicated", "pca67", 67, "euclidean"),
    ("EH", "deduplicated", "pca34", 34, "euclidean"),
    ("CA", "full", "raw1024", 1_024, "cosine"),
    ("CB", "full", "pca113", 113, "cosine"),
    ("CC", "full", "pca67", 67, "cosine"),
    ("CD", "full", "pca34", 34, "cosine"),
    ("CE", "deduplicated", "raw1024", 1_024, "cosine"),
    ("CF", "deduplicated", "pca113", 113, "cosine"),
    ("CG", "deduplicated", "pca67", 67, "cosine"),
    ("CH", "deduplicated", "pca34", 34, "cosine"),
    ("CI", "full", "centred_raw1024", 1_024, "cosine"),
    ("CJ", "deduplicated", "centred_raw1024", 1_024, "cosine"),
)

EUCLIDEAN_SPACES = tuple(dict.fromkeys(
    space
    for _base, _records, space, _dimensions, metric in BASE_GRID
    if metric == "euclidean"
))
COSINE_INPUTS = tuple(dict.fromkeys(
    (records, space)
    for _base, records, space, _dimensions, metric in BASE_GRID
    if metric == "cosine"
))

GRID = tuple(
    (*base, min_cluster_size, min_samples)
    for base in BASE_GRID
    for min_cluster_size in MIN_CLUSTER_SIZES
    for min_samples in MIN_SAMPLES_VALUES
)

BASE_BY_CONFIGURATION = {
    (records, space, metric): base
    for base, records, space, _dimensions, metric in BASE_GRID
}
if len(BASE_BY_CONFIGURATION) != len(BASE_GRID):
    raise ValueError("two base rows share a record set, space and metric")
if len({base for base, *_rest in BASE_GRID}) != len(BASE_GRID):
    raise ValueError("two base rows share a prefix")


def _row_id(base: str, min_cluster_size: int, min_samples: int) -> str:
    return f"{base}{min_cluster_size:02d}{min_samples:02d}"


def _expected_row_id(
    records: str,
    space: str,
    metric: str,
    min_cluster_size: int,
    min_samples: int,
) -> str:
    try:
        base = BASE_BY_CONFIGURATION[(records, space, metric)]
    except KeyError as error:
        raise ValueError(f"unknown base configuration: {error.args[0]}") from error
    return _row_id(base, min_cluster_size, min_samples)

def _run_key(*parts: str) -> str:
    return ":".join(parts)


def _run_label(records: str, space: str, metric: str) -> str:
    return f"{records}: {SPACE_STYLES[space][0]}, {METRIC_LABELS[metric]}"

SPACE_STYLES = {
    "raw1024": ("Raw 1024", "#4c78a8"),
    "pca113": ("PCA 113", "#f58518"),
    "pca67": ("PCA 67", "#54a24b"),
    "pca34": ("PCA 34", "#e45756"),
    "centred_raw1024": ("Centred raw 1024", "#9c755f"),
}
RECORD_STYLES = {"full": "-", "deduplicated": "--"}
METRIC_LABELS = {"euclidean": "Euclidean", "cosine": "Cosine"}
OVAL_MIN_WIDTH_FRACTION = 0.015
TEXT_CONTRAST_CROSSOVER = 0.0525**0.5 - 0.05


def _deduplicated_indices(full_ids: list[str], dedup_ids: list[str]) -> np.ndarray:
    full_index = pd.Index(full_ids)
    if full_index.has_duplicates:
        raise ValueError("protein IDs are not unique")
    position = full_index.get_indexer(pd.Index(dedup_ids))
    absent = np.flatnonzero(position == -1)
    if absent.size:
        raise ValueError(
            f"deduplicated protein is absent from the full matrix: {dedup_ids[int(absent[0])]}"
        )
    return position


def _spaces(X: np.ndarray) -> tuple[dict[str, np.ndarray], dict[int, float]]:
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2 or X.shape[1] < max(PCA_DIMS):
        raise ValueError(f"PCA grid requires a 2-D matrix with at least {max(PCA_DIMS)} columns")
    if not np.isfinite(X).all():
        raise ValueError("PCA grid requires finite embeddings")
    centred = X - X.mean(axis=0)
    pca = PCA(n_components=max(PCA_DIMS), svd_solver="full", whiten=False)
    transformed = pca.fit_transform(X)
    retained = {
        dimensions: float(pca.explained_variance_ratio_[:dimensions].sum())
        for dimensions in PCA_DIMS
    }
    return {
        "raw1024": X,
        "centred_raw1024": centred,
        **{f"pca{d}": transformed[:, :d] for d in PCA_DIMS},
    }, retained


def _variance_retained(
    records: str,
    space: str,
    expected_dims: int,
    retained: dict[int, float],
) -> float | None:
    return retained[expected_dims] if records == "full" and space in PCA_SPACE_NAMES else None


def _require_cosine_input_norms(
    spaces_by_records: dict[str, dict[str, np.ndarray]],
) -> None:
    for records, space in COSINE_INPUTS:
        try:
            coordinates = spaces_by_records[records][space]
        except KeyError as error:
            raise ValueError(f"cosine grid references missing coordinates: {error.args[0]}") from error
        norms = np.linalg.norm(coordinates, axis=1)
        minimum = float(norms.min())
        if not np.isfinite(norms).all() or minimum <= COSINE_NORM_FLOOR:
            raise ValueError(
                f"cosine input {records}/{space} has minimum norm {minimum}; "
                f"expected finite norms above {COSINE_NORM_FLOOR}"
            )


def _cluster_ids(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels)
    if labels.ndim != 1 or labels.size == 0:
        raise ValueError("HDBSCAN labels must be a non-empty 1-D array")
    if not np.issubdtype(labels.dtype, np.integer) or (labels < -1).any():
        raise ValueError("HDBSCAN labels must be integers greater than or equal to -1")
    cluster_ids = np.unique(labels[labels != -1])
    expected = np.arange(len(cluster_ids), dtype=cluster_ids.dtype)
    if not np.array_equal(cluster_ids, expected):
        raise ValueError("HDBSCAN cluster labels are not contiguous from zero")
    return cluster_ids


def _stable_all_points_core_distance(distance_matrix: np.ndarray, d: int) -> np.ndarray:
    if d <= 0:
        raise ValueError("DBCV dimension must be positive")
    if distance_matrix.ndim != 2 or distance_matrix.shape[0] != distance_matrix.shape[1]:
        raise ValueError("DBCV core distances require a square distance matrix")
    if distance_matrix.shape[0] < 2:
        raise ValueError("DBCV core distances require at least two records")
    if not np.isfinite(distance_matrix).all() or (distance_matrix < 0).any():
        raise ValueError("DBCV distances must be finite and non-negative")

    positive = distance_matrix > 0
    log_terms = np.full(distance_matrix.shape, -np.inf, dtype=np.float64)
    log_terms[positive] = -float(d) * np.log(distance_matrix[positive])
    log_sums = logsumexp(log_terms, axis=1)
    result = np.zeros(distance_matrix.shape[0], dtype=np.float64)
    finite = np.isfinite(log_sums)
    result[finite] = np.exp(-(log_sums[finite] - np.log(distance_matrix.shape[0] - 1)) / d)
    return result


def _stable_distances_between_points(
    X: np.ndarray, labels: np.ndarray, cluster_id: int, metric: str,
) -> tuple[np.ndarray, np.ndarray]:
    subset_X = X[labels == cluster_id]
    distance_matrix = pairwise_distances(subset_X, metric=metric)
    core_distances = _stable_all_points_core_distance(distance_matrix, X.shape[1])
    core_dist_matrix = np.tile(core_distances, (core_distances.shape[0], 1))
    return np.dstack([distance_matrix, core_dist_matrix, core_dist_matrix.T]).max(axis=-1), core_distances


def _stable_validity_index(
    X: np.ndarray, labels: np.ndarray, metric: str,
) -> tuple[float, np.ndarray]:
    max_cluster_id = int(labels.max()) + 1
    core_distances: dict[int, np.ndarray] = {}
    density_sparseness: dict[int, float] = {}
    mst_nodes: dict[int, np.ndarray] = {}
    density_separation = np.full((max_cluster_id, max_cluster_id), np.inf, dtype=np.float64)
    cluster_scores = np.empty(max_cluster_id, dtype=np.float64)

    for cluster_id in range(max_cluster_id):
        if not np.any(labels == cluster_id):
            continue
        distances_for_mst, core_distances[cluster_id] = _stable_distances_between_points(
            X, labels, cluster_id, metric
        )
        mst_nodes[cluster_id], mst_edges = hdbscan_validity.internal_minimum_spanning_tree(
            distances_for_mst
        )
        density_sparseness[cluster_id] = float(mst_edges[:, 2].max())

    for left in range(max_cluster_id):
        if not np.any(labels == left):
            continue
        for right in range(left + 1, max_cluster_id):
            if not np.any(labels == right):
                continue
            separation = hdbscan_validity.density_separation(
                X, labels, left, right, mst_nodes[left], mst_nodes[right],
                core_distances[left], core_distances[right], metric=metric,
            )
            density_separation[left, right] = separation
            density_separation[right, left] = separation

    overall = 0.0
    n_samples = float(X.shape[0])
    for cluster_id in range(max_cluster_id):
        cluster_size = int(np.sum(labels == cluster_id))
        if cluster_size == 0:
            continue
        nearest_separation = float(density_separation[cluster_id].min())
        sparseness = density_sparseness[cluster_id]
        cluster_scores[cluster_id] = (nearest_separation - sparseness) / max(
            nearest_separation, sparseness
        )
        overall += (cluster_size / n_samples) * cluster_scores[cluster_id]
    return overall, cluster_scores


def _library_dbcv(
    X: np.ndarray, labels: np.ndarray, metric: str,
) -> tuple[float, np.ndarray] | None:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            with np.errstate(over="warn", under="warn"):
                overall, per_cluster = cast(
                    tuple[float, np.ndarray],
                    validity_index(X, labels, metric=metric, per_cluster_scores=True),
                )
    except RuntimeWarning as exc:
        if "overflow encountered in power" not in str(exc):
            raise
        return None
    return float(overall), np.asarray(per_cluster, dtype=float)


def _dbcv(X: np.ndarray, labels: np.ndarray, metric: str) -> tuple[dict, dict[int, float | None]]:
    cluster_ids = _cluster_ids(labels)
    if len(cluster_ids) < 2:
        return {
            "dbcv_scaled_by_clustered_fraction": None,
            "dbcv_status": "unavailable: fewer than two clusters",
        }, {int(cluster_id): None for cluster_id in cluster_ids}
    X = np.ascontiguousarray(X, dtype=np.float64)
    overall, per_cluster = _stable_validity_index(X, labels, metric)
    library_result = _library_dbcv(X, labels, metric)
    if library_result is not None:
        library_overall, library_per_cluster = library_result
        if not (
            np.isclose(overall, library_overall, rtol=1e-12, atol=1e-12)
            and np.allclose(per_cluster, library_per_cluster, rtol=1e-12, atol=1e-12)
        ):
            raise ValueError("log-space DBCV does not reproduce the finite hdbscan result")

    per_cluster = np.asarray(per_cluster, dtype=float)
    if len(per_cluster) != len(cluster_ids):
        raise ValueError("HDBSCAN DBCV did not return exactly one score per cluster")

    scores = {
        int(cluster_id): (
            float(per_cluster[int(cluster_id)])
            if np.isfinite(per_cluster[int(cluster_id)])
            else None
        )
        for cluster_id in cluster_ids
    }
    finite = bool(np.isfinite(overall) and all(score is not None for score in scores.values()))
    return {
        "dbcv_scaled_by_clustered_fraction": float(overall) if finite else None,
        "dbcv_status": "ok" if finite else "unavailable: non-finite hdbscan result",
    }, scores


def _silhouette(X: np.ndarray, labels: np.ndarray, metric: str) -> dict:
    clustered = labels != -1
    cluster_ids = _cluster_ids(labels)
    n_scored = int(clustered.sum())
    if len(cluster_ids) < 2 or n_scored <= len(cluster_ids):
        return {"silhouette": None, "silhouette_n_scored": n_scored}
    return {
        "silhouette": float(silhouette_score(X[clustered], labels[clustered], metric=metric)),
        "silhouette_n_scored": n_scored,
    }


def _cluster_persistence(values: np.ndarray, labels: np.ndarray) -> dict[int, float]:
    cluster_ids = _cluster_ids(labels)
    values = np.asarray(values, dtype=float)
    if len(values) != len(cluster_ids):
        raise ValueError("HDBSCAN did not return exactly one persistence value per cluster")
    return {int(cluster_id): float(values[int(cluster_id)]) for cluster_id in cluster_ids}


def _maximum_exact_multiplicity(X: np.ndarray) -> int:
    X = np.asarray(X)
    if X.ndim != 2 or len(X) == 0:
        raise ValueError("exact-multiplicity check requires a non-empty 2-D matrix")
    return int(np.unique(X, axis=0, return_counts=True)[1].max())


def _persistence_status(maximum_exact_multiplicity: int, min_samples: int) -> str:
    if maximum_exact_multiplicity < 1:
        raise ValueError("maximum exact multiplicity must be positive")
    if maximum_exact_multiplicity >= min_samples + 1:
        return (
            "unreliable: exact duplicates can make the persistence normaliser degenerate"
        )
    return "ok"


def _membership_probability(
    probabilities: np.ndarray,
    labels: np.ndarray,
) -> tuple[dict, dict[int, float]]:
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.shape != labels.shape:
        raise ValueError("membership probabilities do not match labels")
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("membership probabilities must be finite and between zero and one")
    noise = labels == -1
    if not np.all(probabilities[noise] == 0):
        raise ValueError("HDBSCAN returned non-zero membership probability for noise")
    clustered = ~noise
    cluster_ids = _cluster_ids(labels)
    n_zero = int((probabilities[clustered] == 0).sum())
    return {
        "mean_clustered_probability": (
            float(probabilities[clustered].mean()) if clustered.any() else None
        ),
        "mean_clustered_probability_status": (
            "ok"
            if n_zero == 0
            else f"degenerate: {n_zero} clustered records scored exactly zero"
        ),
    }, {
        int(cluster_id): float(probabilities[labels == cluster_id].mean())
        for cluster_id in cluster_ids
    }


def _noise_percentage(labels: np.ndarray) -> dict:
    _cluster_ids(labels)
    n_noise = int((labels == -1).sum())
    return {
        "n_clustered": int((labels != -1).sum()),
        "n_noise": n_noise,
        "noise_pct": 100 * n_noise / len(labels),
    }


def _partition_ari(
    left_ids: list[str],
    left_labels: np.ndarray,
    right_ids: list[str],
    right_labels: np.ndarray,
) -> dict:
    if len(left_ids) != len(left_labels) or len(right_ids) != len(right_labels):
        raise ValueError("partition IDs and labels have different lengths")
    left_index = pd.Index(left_ids)
    right_index = pd.Index(right_ids)
    if left_index.has_duplicates:
        raise ValueError("left partition protein IDs are not unique")
    if right_index.has_duplicates:
        raise ValueError("right partition protein IDs are not unique")
    right_position = right_index.get_indexer(left_index)
    common = right_position != -1
    if not common.any():
        raise ValueError("partitions have no protein IDs in common")
    left = np.asarray(left_labels[common], dtype=int)
    right = np.asarray(right_labels[right_position[common]], dtype=int)
    clustered_both = (left != -1) & (right != -1)
    n_clustered_both = int(clustered_both.sum())
    return {
        "ari_all": float(adjusted_rand_score(left, right)),
        "ari_clustered_both": (
            float(adjusted_rand_score(left[clustered_both], right[clustered_both]))
            if n_clustered_both >= 3
            else None
        ),
        "n_common": int(common.sum()),
        "n_clustered_both": n_clustered_both,
    }


def _partition_agreement_pairs(
    min_cluster_sizes: tuple[int, ...] = MIN_CLUSTER_SIZES,
    min_samples_values: tuple[int, ...] = MIN_SAMPLES_VALUES,
) -> list[tuple[str, str, str]]:
    base_pairs: list[tuple[str, str, str]] = []
    configurations = set(BASE_BY_CONFIGURATION)

    for base, records, space, _dimensions, metric in BASE_GRID:
        counterpart = ("deduplicated", space, metric)
        if records == "full" and counterpart in configurations:
            base_pairs.append((
                "full_vs_deduplicated",
                base,
                BASE_BY_CONFIGURATION[counterpart],
            ))

    for _base, records, space, _dimensions, metric in BASE_GRID:
        if metric == "euclidean" and space in PCA_SPACE_NAMES:
            base_pairs.append((
                "raw_vs_pca",
                BASE_BY_CONFIGURATION[(records, "raw1024", metric)],
                BASE_BY_CONFIGURATION[(records, space, metric)],
            ))

    for base, records, space, _dimensions, metric in BASE_GRID:
        if metric == "euclidean" and (records, space, "cosine") in configurations:
            base_pairs.append((
                "euclidean_vs_cosine",
                base,
                BASE_BY_CONFIGURATION[(records, space, "cosine")],
            ))

    for base, records, space, _dimensions, metric in BASE_GRID:
        if metric == "cosine" and space == "centred_raw1024":
            base_pairs.append((
                "uncentred_vs_centred_cosine",
                BASE_BY_CONFIGURATION[(records, "raw1024", metric)],
                base,
            ))

    for base, records, space, _dimensions, metric in BASE_GRID:
        if metric == "cosine" and space in PCA_SPACE_NAMES:
            base_pairs.append((
                "centred_vs_pca_cosine",
                BASE_BY_CONFIGURATION[(records, "centred_raw1024", metric)],
                base,
            ))

    pairs = [
        (
            comparison,
            _row_id(left_base, min_cluster_size, min_samples),
            _row_id(right_base, min_cluster_size, min_samples),
        )
        for min_cluster_size in min_cluster_sizes
        for min_samples in min_samples_values
        for comparison, left_base, right_base in base_pairs
    ]
    if any(left_row == right_row for _comparison, left_row, right_row in pairs):
        raise ValueError("partition pair compares a run with itself")
    if len({(comparison, left_row, right_row) for comparison, left_row, right_row in pairs}) != len(pairs):
        raise ValueError("partition-pair generator returned a duplicate pair")
    if len({frozenset((left_row, right_row)) for _comparison, left_row, right_row in pairs}) != len(pairs):
        raise ValueError("partition-pair generator returned a reversed pair")
    return pairs


def _validate_partition_pair_metadata(
    comparison: str,
    left_row: str,
    right_row: str,
    left: pd.Series,
    right: pd.Series,
) -> None:
    if left_row == right_row:
        raise ValueError("partition pair compares a run with itself")
    if int(left["min_cluster_size"]) != int(right["min_cluster_size"]):
        raise ValueError(f"partition pair {left_row}/{right_row} uses different cluster sizes")
    if int(left["min_samples"]) != int(right["min_samples"]):
        raise ValueError(f"partition pair {left_row}/{right_row} uses different min_samples")

    if comparison == "full_vs_deduplicated":
        if str(left["records"]) != "full" or str(right["records"]) != "deduplicated":
            raise ValueError(f"partition pair {left_row}/{right_row} has reversed record sets")
        for field in ("dataset", "space", "metric"):
            if left[field] != right[field]:
                raise ValueError(f"partition pair {left_row}/{right_row} differs in {field}")
        if int(right["n"]) != EXPECTED_DEDUP_N:
            raise ValueError(f"partition pair {left_row}/{right_row} has the wrong deduplicated size")
        return

    if comparison not in {
        "raw_vs_pca",
        "euclidean_vs_cosine",
        "uncentred_vs_centred_cosine",
        "centred_vs_pca_cosine",
    }:
        raise ValueError(f"unknown partition comparison family: {comparison}")
    for field in ("dataset", "records", "n", "subset_hash"):
        if left[field] != right[field]:
            raise ValueError(f"partition pair {left_row}/{right_row} differs in {field}")

    if comparison == "raw_vs_pca":
        valid = (
            left["space"] == "raw1024"
            and right["space"] in PCA_SPACE_NAMES
            and left["metric"] == right["metric"] == "euclidean"
        )
    elif comparison == "euclidean_vs_cosine":
        valid = (
            left["space"] == right["space"]
            and left["metric"] == "euclidean"
            and right["metric"] == "cosine"
        )
    elif comparison == "uncentred_vs_centred_cosine":
        valid = (
            left["space"] == "raw1024"
            and right["space"] == "centred_raw1024"
            and left["metric"] == right["metric"] == "cosine"
        )
    else:
        valid = (
            left["space"] == "centred_raw1024"
            and right["space"] in PCA_SPACE_NAMES
            and left["metric"] == right["metric"] == "cosine"
        )
    if not valid:
        raise ValueError(f"partition pair {left_row}/{right_row} does not match {comparison}")


def _validate_partition_pair_overlap(
    comparison: str,
    left_row: str,
    right_row: str,
    left: pd.Series,
    right: pd.Series,
    agreement: dict,
) -> None:
    n_common = int(agreement["n_common"])
    if comparison == "full_vs_deduplicated":
        if n_common != EXPECTED_DEDUP_N or n_common != int(right["n"]):
            raise ValueError(f"partition pair {left_row}/{right_row} has incomplete deduplicated overlap")
    elif n_common != int(left["n"]):
        raise ValueError(f"partition pair {left_row}/{right_row} has incomplete same-record overlap")


def _partition_agreement_table(
    assignments: dict[str, tuple[list[str], np.ndarray]],
    run_table: pd.DataFrame,
) -> pd.DataFrame:
    if run_table["row"].duplicated().any():
        raise ValueError("run table contains duplicate row IDs")
    metadata = run_table.set_index("row")
    pairs = _partition_agreement_pairs()
    expected_rows = {row for _comparison, left_row, right_row in pairs for row in (left_row, right_row)}
    missing = expected_rows - set(assignments)
    if missing:
        raise ValueError(f"partition assignments are missing rows: {sorted(missing)}")
    if not expected_rows.issubset(metadata.index):
        missing = sorted(expected_rows - set(metadata.index))
        raise ValueError(f"partition run metadata is missing rows: {missing}")

    rows = []
    for comparison, left_row, right_row in pairs:
        left = metadata.loc[left_row]
        right = metadata.loc[right_row]
        if not isinstance(left, pd.Series) or not isinstance(right, pd.Series):
            raise ValueError("partition run metadata must have unique row IDs")
        _validate_partition_pair_metadata(comparison, left_row, right_row, left, right)
        left_ids, left_labels = assignments[left_row]
        right_ids, right_labels = assignments[right_row]
        agreement = _partition_ari(left_ids, left_labels, right_ids, right_labels)
        _validate_partition_pair_overlap(
            comparison, left_row, right_row, left, right, agreement
        )
        rows.append({
            "comparison": comparison,
            "left_row": left_row,
            "right_row": right_row,
            "left_records": left["records"],
            "right_records": right["records"],
            "left_space": left["space"],
            "right_space": right["space"],
            "left_metric": left["metric"],
            "right_metric": right["metric"],
            "min_cluster_size": int(left["min_cluster_size"]),
            "min_samples": int(left["min_samples"]),
            **agreement,
        })
    return pd.DataFrame(rows)


def _run_one(
    row: str,
    records: str,
    space: str,
    metric: str,
    X: np.ndarray,
    ids: list[str],
    subset_hash: str,
    variance_retained: float | None,
    min_cluster_size: int,
    min_samples: int,
    maximum_exact_multiplicity: int,
) -> tuple[dict, list[dict], np.ndarray, hdbscan.HDBSCAN]:
    algorithm = METRIC_ALGORITHMS[metric]
    model = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric=metric,
        algorithm=algorithm,
    )
    if model.algorithm == "best":
        raise ValueError("pin the algorithm so the runs table records the MST path that ran")
    labels = model.fit_predict(np.ascontiguousarray(X, dtype=np.float64))

    clustered = labels != -1
    cluster_ids = _cluster_ids(labels)
    probabilities = np.asarray(model.probabilities_, dtype=float)
    persistence = _cluster_persistence(model.cluster_persistence_, labels)
    membership_summary, cluster_membership = _membership_probability(probabilities, labels)
    noise_summary = _noise_percentage(labels)
    dbcv_summary, cluster_dbcv = _dbcv(X, labels, model.metric)
    silhouette_summary = _silhouette(X, labels, model.metric)

    persistence_status = _persistence_status(maximum_exact_multiplicity, min_samples)

    cluster_rows = []
    for cluster_id in cluster_ids:
        members = labels == cluster_id
        cluster_rows.append({
            "row": row,
            "dataset": DATASET,
            "records": records,
            "subset_hash": subset_hash,
            "space": space,
            "metric": model.metric,
            "min_cluster_size": min_cluster_size,
            "min_samples": model.min_cluster_size if model.min_samples is None else model.min_samples,
            "cluster": int(cluster_id),
            "n": int(members.sum()),
            "dbcv": cluster_dbcv[int(cluster_id)],
            "persistence": persistence[int(cluster_id)],
            "persistence_status": persistence_status,
            "mean_membership_probability": cluster_membership[int(cluster_id)],
        })
    if sum(cluster_row["n"] for cluster_row in cluster_rows) != int(clustered.sum()):
        raise ValueError(f"row {row} cluster sizes do not sum to the clustered record count")
    if any(cluster_row["n"] < min_cluster_size for cluster_row in cluster_rows):
        raise ValueError(f"row {row} contains a cluster below minimum cluster size")

    ASSIGNMENTS_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        ASSIGNMENTS_DIR / f"{row}_{records}_{space}.npz",
        protein_id=np.asarray(ids, dtype=str),
        label=np.asarray(labels, dtype=np.int32),
        membership_probability=probabilities,
    )

    run_row = {
        "row": row,
        "dataset": DATASET,
        "records": records,
        "n": len(X),
        "subset_hash": subset_hash,
        "space": space,
        "n_dims": X.shape[1],
        "variance_retained": variance_retained,
        "metric": model.metric,
        "min_cluster_size": model.min_cluster_size,
        "min_samples": model.min_cluster_size if model.min_samples is None else model.min_samples,
        "cluster_selection_method": model.cluster_selection_method,
        "algorithm": algorithm,
        "n_clusters": len(cluster_ids),
        **dbcv_summary,
        **silhouette_summary,
        **noise_summary,
        **membership_summary,
        "git_sha": GIT_SHA,
        **LIBRARY_VERSIONS,
    }
    return run_row, cluster_rows, labels, model


def _min_samples_paired_table(run_table: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "dataset",
        "records",
        "n",
        "subset_hash",
        "space",
        "metric",
        "min_cluster_size",
    ]
    measures = ["noise_pct", "n_noise", "n_clustered", "n_clusters"]
    baseline = run_table[run_table["min_samples"] == BASELINE_MIN_SAMPLES]
    tested = run_table[run_table["min_samples"] != BASELINE_MIN_SAMPLES]
    if baseline.duplicated(keys).any():
        raise ValueError("min_samples baseline has duplicate matched configurations")

    baseline = baseline[keys + ["row", "min_samples", *measures]].rename(
        columns={
            "row": "baseline_row",
            "min_samples": "baseline_min_samples",
            **{measure: f"baseline_{measure}" for measure in measures},
        }
    )
    tested = tested[keys + ["row", "min_samples", *measures]].rename(
        columns={
            "row": "tested_row",
            "min_samples": "tested_min_samples",
            **{measure: f"tested_{measure}" for measure in measures},
        }
    )
    paired = tested.merge(baseline, on=keys, how="left", validate="many_to_one")
    if paired["baseline_row"].isna().any():
        raise ValueError("a non-baseline run has no min_samples=5 match")
    for measure in measures:
        paired[f"{measure}_change"] = (
            paired[f"tested_{measure}"] - paired[f"baseline_{measure}"]
        )

    columns = [
        "baseline_row",
        "tested_row",
        *keys,
        "baseline_min_samples",
        "tested_min_samples",
    ]
    for measure in measures:
        columns.extend([
            f"baseline_{measure}",
            f"tested_{measure}",
            f"{measure}_change",
        ])
    paired = paired[columns]
    expected = len(BASE_GRID) * len(MIN_CLUSTER_SIZES) * (len(MIN_SAMPLES_VALUES) - 1)
    if len(paired) != expected or paired.duplicated(["tested_row", "baseline_row"]).any():
        raise ValueError(f"paired min_samples table expected {expected} unique rows")
    return paired


def _require_columns(table: pd.DataFrame, table_name: str, columns: set[str]) -> None:
    missing = sorted(columns - set(table.columns))
    if missing:
        raise ValueError(f"{table_name} is missing columns: {', '.join(missing)}")


def _min_samples_markdown_blocks(
    run_table: pd.DataFrame,
    paired_table: pd.DataFrame,
) -> dict[str, str]:
    run_columns = {
        "row",
        "dataset",
        "records",
        "n",
        "subset_hash",
        "space",
        "metric",
        "min_cluster_size",
        "min_samples",
        "n_clusters",
        "n_clustered",
        "n_noise",
        "noise_pct",
    }
    paired_columns = {
        "baseline_row",
        "tested_row",
        "baseline_min_samples",
        "tested_min_samples",
        "baseline_noise_pct",
        "tested_noise_pct",
        "noise_pct_change",
        "baseline_n_noise",
        "tested_n_noise",
        "n_noise_change",
        "baseline_n_clustered",
        "tested_n_clustered",
        "n_clustered_change",
        "baseline_n_clusters",
        "tested_n_clusters",
        "n_clusters_change",
    }
    _require_columns(run_table, RUNS_PATH.name, run_columns)
    _require_columns(paired_table, PAIRED_PATH.name, paired_columns)

    expected_runs_per_value = len(BASE_GRID) * len(MIN_CLUSTER_SIZES)
    expected_run_counts = pd.Series(
        expected_runs_per_value,
        index=pd.Index(MIN_SAMPLES_VALUES, name="min_samples"),
        dtype="int64",
    )
    actual_run_counts = run_table.groupby("min_samples").size()
    if not actual_run_counts.equals(expected_run_counts):
        raise ValueError(
            "run CSV does not contain the expected number of rows for each min_samples value"
        )
    if not run_table["row"].is_unique:
        raise ValueError("run CSV contains duplicate row IDs")
    calculated_noise = 100 * run_table["n_noise"] / run_table["n"]
    if not np.allclose(
        calculated_noise,
        run_table["noise_pct"],
        rtol=0,
        atol=1e-12,
        equal_nan=False,
    ):
        raise ValueError("run CSV noise_pct does not equal 100 * n_noise / n")

    tested_values = tuple(value for value in MIN_SAMPLES_VALUES if value != BASELINE_MIN_SAMPLES)
    expected_pairs_per_value = expected_runs_per_value
    expected_pair_counts = pd.Series(
        expected_pairs_per_value,
        index=pd.Index(tested_values, name="tested_min_samples"),
        dtype="int64",
    )
    actual_pair_counts = paired_table.groupby("tested_min_samples").size()
    if not actual_pair_counts.equals(expected_pair_counts):
        raise ValueError(
            "paired CSV does not contain the expected number of rows for each tested value"
        )
    if paired_table.duplicated(["baseline_row", "tested_row"]).any():
        raise ValueError("paired CSV contains duplicate run pairs")
    if not paired_table["baseline_min_samples"].eq(BASELINE_MIN_SAMPLES).all():
        raise ValueError("paired CSV contains a non-baseline comparison row")

    run_by_row = run_table.set_index("row")
    for side in ("baseline", "tested"):
        row_ids = paired_table[f"{side}_row"]
        if not row_ids.isin(run_by_row.index).all():
            raise ValueError(f"paired CSV contains an unknown {side} row")
        for measure in ("noise_pct", "n_noise", "n_clustered", "n_clusters"):
            from_runs = row_ids.map(run_by_row[measure]).to_numpy()
            if not np.allclose(
                from_runs,
                paired_table[f"{side}_{measure}"],
                rtol=0,
                atol=1e-12,
                equal_nan=False,
            ):
                raise ValueError(
                    f"paired CSV {side}_{measure} does not match the run CSV"
                )
    for measure in ("noise_pct", "n_noise", "n_clustered", "n_clusters"):
        calculated_change = (
            paired_table[f"tested_{measure}"] - paired_table[f"baseline_{measure}"]
        )
        if not np.allclose(
            calculated_change,
            paired_table[f"{measure}_change"],
            rtol=0,
            atol=1e-12,
            equal_nan=False,
        ):
            raise ValueError(f"paired CSV {measure}_change has incorrect arithmetic")

    absolute_lines = [
        "| `min_samples` | Minimum noise | Median noise | Maximum noise |",
        "|---:|---:|---:|---:|",
    ]
    for min_samples in MIN_SAMPLES_VALUES:
        noise = run_table.loc[run_table["min_samples"] == min_samples, "noise_pct"]
        absolute_lines.append(
            f"| {min_samples} | {noise.min():.2f}% | {noise.median():.2f}% | "
            f"{noise.max():.2f}% |"
        )

    paired_lines = [
        "| Tested `min_samples` | Noise change, min / median / max | "
        "Increased / decreased / unchanged | Median change in noise records | "
        "Median change in clustered records | Median change in clusters |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for min_samples in tested_values:
        rows = paired_table[paired_table["tested_min_samples"] == min_samples]
        noise_change = rows["noise_pct_change"]
        direction = (
            f"{int((noise_change > 0).sum())} / "
            f"{int((noise_change < 0).sum())} / "
            f"{int((noise_change == 0).sum())}"
        )
        noise_range = " / ".join(
            f"{value:+.2f}"
            for value in (noise_change.min(), noise_change.median(), noise_change.max())
        )
        paired_lines.append(
            f"| {min_samples} | {noise_range} | {direction} | "
            f"{rows['n_noise_change'].median():+.1f} | "
            f"{rows['n_clustered_change'].median():+.1f} | "
            f"{rows['n_clusters_change'].median():+.1f} |"
        )

    baseline_minimum = run_table.loc[
        run_table["min_samples"] == BASELINE_MIN_SAMPLES, "noise_pct"
    ].min()
    expanded_minimum_row = run_table.loc[run_table["noise_pct"].idxmin()]
    if not isinstance(expanded_minimum_row, pd.Series):
        raise ValueError("run table minimum must identify exactly one row")
    if int(expanded_minimum_row["min_samples"]) == BASELINE_MIN_SAMPLES:
        minimum_example = (
            f"**The literal {baseline_minimum:.2f}% floor survives this sweep.** The expanded "
            f"minimum remains {expanded_minimum_row['noise_pct']}% at "
            f"`{expanded_minimum_row['row']}`."
        )
    else:
        matched = paired_table[
            paired_table["tested_row"] == expanded_minimum_row["row"]
        ]
        if len(matched) != 1:
            raise ValueError("expanded minimum does not have exactly one baseline match")
        matched = matched.iloc[0]
        cluster_change = int(matched["n_clusters_change"])
        minimum_example = (
            f"**The literal {baseline_minimum:.2f}% floor is withdrawn.** The pre-declared "
            f"comparison used the exact initial minimum, {baseline_minimum}%, from the run CSV. "
            f"The expanded minimum is {expanded_minimum_row['noise_pct']}% at "
            f"`{expanded_minimum_row['row']}`. That run clusters "
            f"{int(expanded_minimum_row['n_clustered']):,} records into "
            f"{int(expanded_minimum_row['n_clusters'])} clusters. Its "
            f"`min_samples={BASELINE_MIN_SAMPLES}` match, `{matched['baseline_row']}`, has "
            f"{matched['baseline_noise_pct']:.2f}% noise and clusters "
            f"{int(matched['baseline_n_clustered']):,} records into "
            f"{int(matched['baseline_n_clusters'])} clusters. The lower noise therefore comes "
            f"with {cluster_change} more clusters. It is a coverage and granularity trade-off, "
            "not automatic evidence of better structure."
        )

    return {
        "min_samples absolute noise": "\n".join(absolute_lines),
        "min_samples paired changes": "\n".join(paired_lines),
        "min_samples minimum example": textwrap.fill(minimum_example, width=100),
    }


def _replace_generated_markdown_block(text: str, name: str, body: str) -> str:
    begin = f"<!-- BEGIN GENERATED: {name} -->"
    end = f"<!-- END GENERATED: {name} -->"
    if text.count(begin) != 1 or text.count(end) != 1:
        raise ValueError(f"{WRITEUP_PATH.name} must contain one generated block named {name!r}")
    start = text.index(begin) + len(begin)
    finish = text.index(end, start)
    newline = "\r\n" if "\r\n" in text else "\n"
    replacement = newline + body.replace("\n", newline) + newline
    return text[:start] + replacement + text[finish:]


def _update_min_samples_markdown_from_csv() -> None:
    if not WRITEUP_PATH.is_file():
        return
    run_table = pd.read_csv(RUNS_PATH)
    paired_table = pd.read_csv(PAIRED_PATH)
    blocks = _min_samples_markdown_blocks(run_table, paired_table)
    with WRITEUP_PATH.open("r", encoding="utf-8", newline="") as handle:
        original = handle.read()
    updated = original
    for name, body in blocks.items():
        updated = _replace_generated_markdown_block(updated, name, body)
    if updated != original:
        with WRITEUP_PATH.open("w", encoding="utf-8", newline="") as handle:
            handle.write(updated)
    print(f"updated generated min_samples summaries in {WRITEUP_PATH.relative_to(REPO)}")


def _draw_deduplicated_raw1024_condensed_trees(
    models: dict[int, hdbscan.HDBSCAN],
) -> None:
    if set(models) != set(MIN_CLUSTER_SIZES):
        raise ValueError("condensed-tree figure requires one model per minimum cluster size")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"HDBSCAN condensed trees: deduplicated raw 1,024-dimensional {DATASET} embeddings",
        fontsize=14,
        fontweight="bold",
    )
    for axis, min_cluster_size in zip(axes.flat, MIN_CLUSTER_SIZES):
        model = models[min_cluster_size]
        if model.min_cluster_size != min_cluster_size:
            raise ValueError("condensed-tree model does not match its minimum cluster size")
        labels = np.asarray(model.labels_)
        n_clusters = len(_cluster_ids(labels))
        noise_pct = 100 * np.count_nonzero(labels == -1) / len(labels)
        model.condensed_tree_.plot(axis=axis, select_clusters=False, colorbar=False)
        axis.set_title(
            f"min_cluster_size={min_cluster_size}, min_samples={BASELINE_MIN_SAMPLES}: "
            f"{n_clusters} clusters, {noise_pct:.1f}% noise",
            fontsize=10,
        )

    fig.text(
        0.5,
        0.02,
        "Each panel is a separate HDBSCAN fit. Bar colour is local to its panel and cannot be "
        "compared across settings. Selection ovals are omitted here; separate images show local "
        "overlays for sizes 10, 20, and 40.",
        ha="center",
        fontsize=8,
        style="italic",
        color="grey",
    )
    fig.tight_layout(rect=(0, 0.07, 1, 0.94))
    CONDENSED_TREE_FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(CONDENSED_TREE_FIGURE_PATH, dpi=150)
    plt.close(fig)


def _draw_condensed_tree_with_ovals(tree, axis: Axes) -> int:
    tree.plot(axis=axis, select_clusters=True, colorbar=False)
    ovals = [child for child in axis.get_children() if isinstance(child, Ellipse)]
    if not ovals:
        raise ValueError("hdbscan drew no EOM selection ovals")
    cluster_bounds = tree.get_plot_data()["cluster_bounds"]
    x_span = float(
        max(bounds[1] for bounds in cluster_bounds.values())
        - min(bounds[0] for bounds in cluster_bounds.values())
    )
    min_width = OVAL_MIN_WIDTH_FRACTION * x_span
    if not np.isfinite(min_width) or min_width <= 0:
        raise ValueError("condensed tree has no positive horizontal range for selection ovals")
    for oval in ovals:
        oval.set_width(max(float(np.ravel(oval.get_width())[0]), min_width))
        oval.set_height(float(np.ravel(oval.get_height())[0]))
        oval.set_linewidth(1.25)

    return len(ovals)


def _draw_deduplicated_raw1024_condensed_tree_panels(
    models: dict[int, hdbscan.HDBSCAN],
) -> list[Path]:
    if set(models) != set(MIN_CLUSTER_SIZES):
        raise ValueError("condensed-tree panels require one model per minimum cluster size")

    output_paths = []
    CONDENSED_TREE_PANEL_DIR.mkdir(parents=True, exist_ok=True)
    for min_cluster_size in MIN_CLUSTER_SIZES:
        model = models[min_cluster_size]
        if model.min_cluster_size != min_cluster_size:
            raise ValueError("condensed-tree model does not match its minimum cluster size")
        labels = np.asarray(model.labels_)
        n_clusters = len(_cluster_ids(labels))
        noise_pct = 100 * np.count_nonzero(labels == -1) / len(labels)
        fig, axis = plt.subplots(figsize=(12, 8))
        tree = model.condensed_tree_

        if min_cluster_size == 5:
            tree.plot(axis=axis, select_clusters=False, colorbar=False)
            footer = (
                f"EOM selection ovals omitted: {n_clusters} selected clusters would obscure "
                "the hierarchy."
            )
        else:
            n_ovals = _draw_condensed_tree_with_ovals(tree, axis)
            if n_ovals != n_clusters:
                raise ValueError("EOM oval count does not match HDBSCAN cluster count")
            footer = (
                f"Red ovals mark all {n_ovals} EOM-selected clusters. Minimum oval dimensions are "
                "display-only and do not change the clustering."
            )

        fig.suptitle(
            f"HDBSCAN condensed tree: deduplicated raw 1,024-dimensional {DATASET} embeddings\n"
            f"min_cluster_size={min_cluster_size}, min_samples={BASELINE_MIN_SAMPLES}: "
            f"{n_clusters} clusters, {noise_pct:.1f}% noise",
            fontsize=14,
            fontweight="bold",
        )
        fig.text(0.5, 0.02, footer, ha="center", fontsize=9, style="italic", color="grey")
        fig.subplots_adjust(left=0.07, right=0.98, top=0.82, bottom=0.10)
        output_path = CONDENSED_TREE_PANEL_DIR / (
            f"min_cluster_size_{min_cluster_size:02d}_"
            f"min_samples_{BASELINE_MIN_SAMPLES:02d}.png"
        )
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        output_paths.append(output_path)

    return output_paths


def _excluded_silhouette_note(run_table: pd.DataFrame, width: int | None = None) -> str:
    expected_configurations = {
        (records, space, metric)
        for _base, records, space, _dimensions, metric in BASE_GRID
        if metric == "cosine"
    }
    observed_configurations = {
        (str(records), str(space), str(metric))
        for records, space, metric in run_table.loc[
            run_table["metric"] == "cosine", ["records", "space", "metric"]
        ].itertuples(index=False, name=None)
    }
    if observed_configurations != expected_configurations:
        raise ValueError("silhouette note does not cover exactly the cosine grid configurations")

    geometries = tuple(dict.fromkeys(
        (space, metric)
        for _base, _records, space, _dimensions, metric in BASE_GRID
        if metric == "cosine"
    ))
    assigned_configurations: list[tuple[str, str, str]] = []
    parts = []
    for space, metric in geometries:
        configurations = [
            configuration
            for configuration in expected_configurations
            if configuration[1:] == (space, metric)
        ]
        assigned_configurations.extend(configurations)
        rows = run_table[
            (run_table["space"] == space) & (run_table["metric"] == metric)
        ]
        finite_values = rows["silhouette"].dropna()
        label = f"{SPACE_STYLES[space][0]} {METRIC_LABELS[metric]}"
        if finite_values.empty:
            parts.append(
                f"{label} silhouette unavailable: 0 finite of {len(rows)} plotted runs "
                "(full and deduplicated pooled)."
            )
        else:
            parts.append(
                f"{label} silhouette {finite_values.min():.3f} to {finite_values.max():.3f} "
                f"over {len(finite_values)} finite of {len(rows)} plotted runs "
                "(full and deduplicated pooled)."
            )
    if (
        set(assigned_configurations) != expected_configurations
        or len(assigned_configurations) != len(expected_configurations)
    ):
        raise ValueError("a cosine grid configuration is missing from or repeated in the silhouette note")
    note = (
        "Cosine is omitted from the Euclidean-only silhouette panel because the two distances "
        "are not on one scale. " + " ".join(parts)
    )
    return textwrap.fill(note, width) if width else note


def _dbcv_availability_note(run_table: pd.DataFrame, width: int | None = None) -> str:
    unavailable = run_table[run_table["dbcv_status"] != "ok"]
    if unavailable.empty:
        note = f"Scaled DBCV is available for all {len(run_table)} runs."
    else:
        configurations = tuple(dict.fromkeys(
            (records, space, metric)
            for _base, records, space, _dimensions, metric in BASE_GRID
        ))
        unavailable_configurations = {
            (str(records), str(space), str(metric))
            for records, space, metric in unavailable[
                ["records", "space", "metric"]
            ].itertuples(index=False, name=None)
        }
        named_configurations = [
            configuration
            for configuration in configurations
            if configuration in unavailable_configurations
        ]
        if set(named_configurations) != unavailable_configurations:
            raise ValueError("DBCV note has an unavailable configuration outside BASE_GRID")
        show_metric = run_table["metric"].nunique() > 1
        named = [
            f"{records} {SPACE_STYLES[space][0]}"
            + (f" {METRIC_LABELS[metric]}" if show_metric else "")
            for records, space, metric in named_configurations
        ]
        suffix = (
            ", where hdbscan raised a numerical warning."
            if run_table["metric"].nunique() == 1
            else "; the run CSV records each status."
        )
        note = (
            f"Scaled DBCV is unavailable in {len(unavailable)} of {len(run_table)} runs "
            f"({', '.join(named)}){suffix}"
        )
    return textwrap.fill(note, width) if width else note


def _draw_results(run_table: pd.DataFrame, cluster_table: pd.DataFrame) -> None:
    run_table = run_table[run_table["metric"] == "euclidean"]
    cluster_table = cluster_table[cluster_table["metric"] == "euclidean"]
    fig, axes = plt.subplots(3, 2, figsize=(13, 13), sharex=True)
    fig.suptitle(
        f"HDBSCAN sensitivity grid: {DATASET}, Euclidean, min_samples={BASELINE_MIN_SAMPLES}",
        fontsize=14,
        fontweight="bold",
    )

    persistence_rng = np.random.default_rng(42)
    persistence_offsets = 2.0 ** np.linspace(-0.25, 0.25, len(EUCLIDEAN_SPACES))
    for space_index, space in enumerate(EUCLIDEAN_SPACES):
        colour = SPACE_STYLES[space][1]
        for records, line_style in RECORD_STYLES.items():
            rows = run_table[
                (run_table["space"] == space) & (run_table["records"] == records)
            ].sort_values("min_cluster_size")
            x = rows["min_cluster_size"]
            axes[0, 0].plot(x, rows["n_clusters"], marker="o", color=colour, ls=line_style)
            axes[0, 1].plot(x, rows["noise_pct"], marker="o", color=colour, ls=line_style)
            axes[1, 0].plot(
                x,
                rows["dbcv_scaled_by_clustered_fraction"],
                marker="o",
                color=colour,
                ls=line_style,
            )
            axes[1, 1].plot(x, rows["silhouette"], marker="o", color=colour, ls=line_style)
            axes[2, 0].plot(
                x,
                rows["mean_clustered_probability"],
                marker="o",
                color=colour,
                ls=line_style,
            )

        rows = cluster_table[
            (cluster_table["space"] == space)
            & (cluster_table["records"] == "deduplicated")
        ]
        persistence_values = rows["persistence"].to_numpy()
        if rows.empty or not np.isfinite(persistence_values).all():
            raise ValueError(f"missing or non-finite deduplicated persistence for {space}")
        if (persistence_values == 1.0).any():
            raise ValueError(f"degenerate deduplicated persistence for {space}")
        for min_cluster_size in MIN_CLUSTER_SIZES:
            values = rows.loc[
                rows["min_cluster_size"] == min_cluster_size, "persistence"
            ].to_numpy()
            if values.size == 0:
                raise ValueError(
                    f"missing deduplicated persistence for {space}, size {min_cluster_size}"
                )
            x = min_cluster_size * persistence_offsets[space_index]
            jitter = 2.0 ** persistence_rng.uniform(-0.03, 0.03, size=len(values))
            axes[2, 1].scatter(
                x * jitter,
                values,
                s=14,
                alpha=0.6,
                color=colour,
                edgecolors="none",
            )
            axes[2, 1].scatter(
                x,
                np.median(values),
                marker="_",
                s=100,
                linewidths=2,
                color="black",
                zorder=3,
            )

    axes[0, 0].set(title="1. Number of clusters", ylabel="clusters")
    axes[0, 1].set(title="2. Records labelled as noise", ylabel="noise (%)")
    axes[1, 0].set(
        title=(
            "3. DBCV scaled by clustered fraction\n"
            "(compare settings within one space only; missing means a library numerical warning)"
        ),
        ylabel="scaled DBCV (-1 to 1)",
    )
    axes[1, 1].set(
        title=(
            "4. Silhouette on clustered records\n"
            "(scored set changes with minimum cluster size)\n"
            "(compare settings within one space only)"
        ),
        ylabel="silhouette (-1 to 1)",
    )
    axes[2, 0].set(
        title=(
            "5. Mean membership probability\n"
            "(rises mechanically with cluster count)\n"
            "(probabilities are per-cluster normalised)"
        ),
        xlabel="minimum cluster size (log scale)",
        ylabel="mean probability, clustered records",
    )
    axes[2, 1].set(
        title=(
            "6. Cluster persistence, deduplicated runs\n"
            "(each dot is one cluster; black bar is the median)\n"
            "(each run has its own scale, so no two runs are comparable)"
        ),
        xlabel="minimum cluster size (log scale)",
        ylabel="cluster persistence",
    )

    for ax in axes.flat:
        ax.set_xscale("log", base=2)
        ax.set_xticks(MIN_CLUSTER_SIZES, [str(size) for size in MIN_CLUSTER_SIZES])
        ax.grid(alpha=0.3)

    space_legend = [
        Line2D(
            [0],
            [0],
            color=SPACE_STYLES[space][1],
            marker="o",
            label=SPACE_STYLES[space][0],
        )
        for space in EUCLIDEAN_SPACES
    ]
    record_legend = [
        Line2D([0], [0], color="black", ls=line_style, label=records)
        for records, line_style in RECORD_STYLES.items()
    ]
    fig.legend(
        handles=space_legend + record_legend,
        loc="lower center",
        ncol=6,
        frameon=False,
    )
    fig.text(
        0.5,
        0.035,
        "Panels 2 and 3 are not independent: scaled DBCV falls partly as noise rises. "
        "Blank scaled DBCV marks an unavailable run.\n"
        f"{_dbcv_availability_note(run_table)}\n"
        "Duplicate rows bias any full-set scaled DBCV toward sparser. "
        "Membership covers clustered records only.",
        ha="center",
        fontsize=8,
        style="italic",
        color="grey",
    )
    fig.tight_layout(rect=(0, 0.09, 1, 0.95))
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_PATH, dpi=150)
    plt.close(fig)


def _grid_values(
    table: pd.DataFrame,
    row_field: str,
    row_keys: list[str],
    value_field: str,
    context: str,
) -> np.ndarray:
    wanted = table[
        table[row_field].isin(row_keys)
        & table["min_cluster_size"].isin(MIN_CLUSTER_SIZES)
    ]
    if len(wanted) != len(row_keys) * len(MIN_CLUSTER_SIZES):
        raise ValueError(f"expected one {context} row per row key and minimum cluster size")
    grid = wanted.pivot(index=row_field, columns="min_cluster_size", values=value_field)
    return grid.reindex(
        index=row_keys, columns=list(MIN_CLUSTER_SIZES)
    ).to_numpy(dtype=float)


def _annotate_heatmap(
    ax: Axes,
    image: AxesImage,
    values: np.ndarray,
    value_format: str,
    secondary_values: np.ndarray | None = None,
) -> None:
    for row_index in range(values.shape[0]):
        for column_index in range(values.shape[1]):
            value = values[row_index, column_index]
            if not np.isfinite(value):
                label = "NA"
                text_colour = "#555555"
            else:
                label = format(value, value_format)
                luminance = relative_luminance(image.cmap(image.norm(value)))
                text_colour = "black" if luminance > TEXT_CONTRAST_CROSSOVER else "white"
            if secondary_values is not None and np.isfinite(secondary_values[row_index, column_index]):
                label = f"{label}\nn={int(secondary_values[row_index, column_index])}"
            ax.text(
                column_index,
                row_index,
                label,
                ha="center",
                va="center",
                fontsize=8,
                color=text_colour,
            )


def _draw_metric_heatmaps(run_table: pd.DataFrame) -> None:
    configurations = [
        (records, space, metric)
        for _, records, space, _dims, metric in BASE_GRID
    ]
    table = run_table.copy()
    table["metric_heatmap_key"] = table.apply(
        lambda run: _run_key(run["records"], run["space"], run["metric"]), axis=1
    )
    panel_specs = (
        ("n_clusters", "1. Number of clusters", ".0f", None),
        ("noise_pct", "2. Records labelled as noise", ".1f", None),
        (
            "silhouette",
            "3. Silhouette on clustered records, Euclidean only\n"
            "(value; n scored)",
            ".3f",
            "euclidean",
        ),
        (
            "mean_clustered_probability",
            "4. Mean membership probability\n"
            "(clustered records; per-cluster normalised)",
            ".3f",
            None,
        ),
        (
            "dbcv_scaled_by_clustered_fraction",
            "5. DBCV scaled by clustered fraction",
            ".3f",
            None,
        ),
    )
    if len(panel_specs) != 5:
        raise ValueError("metric heatmap requires exactly five panels")

    fig, axes = plt.subplots(3, 2, figsize=(13, 15))
    fig.suptitle(
        f"HDBSCAN run metrics: {DATASET}, min_samples={BASELINE_MIN_SAMPLES}",
        fontsize=14,
        fontweight="bold",
    )
    for ax, (column, title, value_format, only_metric) in zip(axes.flat, panel_specs):
        panel_rows = [
            configuration
            for configuration in configurations
            if only_metric is None or configuration[2] == only_metric
        ]
        panel_keys = [_run_key(*configuration) for configuration in panel_rows]
        panel_labels = [_run_label(*configuration) for configuration in panel_rows]
        values = _grid_values(
            table,
            "metric_heatmap_key",
            panel_keys,
            column,
            "metric heatmap",
        )
        colour_map = plt.get_cmap("viridis").copy()
        colour_map.set_bad("#d9d9d9")
        image = ax.imshow(
            np.ma.masked_invalid(values),
            aspect="auto",
            cmap=colour_map,
        )
        secondary_values = (
            _grid_values(
                table,
                "metric_heatmap_key",
                panel_keys,
                "silhouette_n_scored",
                "silhouette sample-count heatmap",
            )
            if column == "silhouette"
            else None
        )
        _annotate_heatmap(ax, image, values, value_format, secondary_values)
        for boundary in range(1, len(panel_rows)):
            if panel_rows[boundary][0] != panel_rows[boundary - 1][0]:
                ax.axhline(boundary - 0.5, color="white", linewidth=2)
        ax.set_title(title)
        ax.set_xticks(np.arange(len(MIN_CLUSTER_SIZES)), MIN_CLUSTER_SIZES)
        ax.set_yticks(np.arange(len(panel_labels)), panel_labels)
        ax.set_xlabel("minimum cluster size")
        fig.colorbar(image, ax=ax, shrink=0.8)

    guide_axis = axes[2, 1]
    guide_axis.axis("off")
    guide_axis.text(
        0.05,
        0.82,
        "Each cell is one HDBSCAN run.\n"
        "Rows combine the record set, embedding space\n"
        "and distance metric.\n"
        "Columns show the tested minimum cluster size.\n\n"
        "Colours compare values within a panel only.\n"
        "A higher value does not automatically mean a better run.\n"
        "Mean membership probability rises with cluster count\n"
        "because probabilities are normalised per cluster.\n"
        "Silhouette cells show n scored; that set changes with\n"
        "minimum cluster size.\n"
        "Compare silhouette only within one embedding space\n"
        "and one distance metric.\n"
        "Compare scaled DBCV only within one embedding space.\n"
        "Full-set scaled DBCV is biased toward sparser\n"
        "because of duplicate rows.\n"
        f"{_dbcv_availability_note(run_table, width=46)}\n\n"
        f"{_excluded_silhouette_note(run_table, width=46)}",
        fontsize=8,
        linespacing=1.25,
        va="top",
        transform=guide_axis.transAxes,
    )
    fig.tight_layout(rect=(0, 0.02, 1, 0.96))
    METRIC_FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(METRIC_FIGURE_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _draw_min_samples_grid(
    run_table: pd.DataFrame,
    column: str,
    value_format: str,
    vmax: float,
    title: str,
    colour_bar_label: str,
    scale_note: str,
    output_path: Path,
    vmin: float = 0,
    colour_map: str = "viridis",
) -> None:
    configurations = [
        (records, space, metric)
        for _base, records, space, _dimensions, metric in BASE_GRID
    ]
    configuration_keys = [_run_key(*configuration) for configuration in configurations]
    configuration_labels = [_run_label(*configuration) for configuration in configurations]
    table = run_table.copy()
    table["min_samples_heatmap_key"] = table.apply(
        lambda run: _run_key(run["records"], run["space"], run["metric"]), axis=1
    )

    fig, axes = plt.subplots(2, 2, figsize=(15, 13), sharex=True, sharey=True)
    fig.suptitle(title, fontsize=14, fontweight="bold")
    image = None
    for axis, min_cluster_size in zip(axes.flat, MIN_CLUSTER_SIZES):
        wanted = table[table["min_cluster_size"] == min_cluster_size]
        expected = len(configuration_keys) * len(MIN_SAMPLES_VALUES)
        if len(wanted) != expected:
            raise ValueError(
                f"min_samples {column} heatmap expected {expected} rows "
                f"at size {min_cluster_size}"
            )
        values = wanted.pivot(
            index="min_samples_heatmap_key",
            columns="min_samples",
            values=column,
        ).reindex(index=configuration_keys, columns=list(MIN_SAMPLES_VALUES)).to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"min_samples {column} heatmap contains a missing value")
        image = axis.imshow(values, aspect="auto", cmap=colour_map, vmin=vmin, vmax=vmax)
        _annotate_heatmap(axis, image, values, value_format)
        axis.set_title(f"min_cluster_size={min_cluster_size}")
        axis.set_xticks(np.arange(len(MIN_SAMPLES_VALUES)), MIN_SAMPLES_VALUES)
        axis.tick_params(axis="x", labelbottom=True)
        axis.set_yticks(np.arange(len(configuration_labels)), configuration_labels)
        axis.set_xlabel("min_samples")
        for boundary in range(1, len(configurations)):
            if configurations[boundary][0] != configurations[boundary - 1][0]:
                axis.axhline(boundary - 0.5, color="white", linewidth=2)
    if image is None:
        raise ValueError(f"min_samples {column} heatmap has no panels")
    colour_axis = fig.add_axes((0.92, 0.15, 0.018, 0.7))
    fig.colorbar(image, cax=colour_axis, label=colour_bar_label)
    fig.text(
        0.5,
        0.025,
        scale_note,
        ha="center",
        fontsize=9,
        style="italic",
        color="grey",
    )
    fig.subplots_adjust(left=0.22, right=0.88, top=0.92, bottom=0.08, hspace=0.24, wspace=0.10)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _draw_min_samples_noise(run_table: pd.DataFrame) -> None:
    _draw_min_samples_grid(
        run_table,
        column="noise_pct",
        value_format=".2f",
        vmax=100,
        title=f"HDBSCAN records labelled as noise: {DATASET}",
        colour_bar_label="records labelled as noise (%)",
        scale_note="Each cell is one run. All panels use the same 0% to 100% colour scale.",
        output_path=MIN_SAMPLES_FIGURE_PATH,
    )


def _draw_min_samples_clusters(run_table: pd.DataFrame) -> None:
    vmax = float(run_table["n_clusters"].max())
    _draw_min_samples_grid(
        run_table,
        column="n_clusters",
        value_format=".0f",
        vmax=vmax,
        title=f"HDBSCAN clusters recovered: {DATASET}",
        colour_bar_label="clusters",
        scale_note=(
            f"Each cell is one run. All panels use the same 0 to {int(vmax)} colour scale, so "
            "panels darken together as the count collapses."
        ),
        output_path=MIN_SAMPLES_CLUSTERS_FIGURE_PATH,
    )


def _draw_min_samples_dbcv(
    run_table: pd.DataFrame,
    output_path: Path = MIN_SAMPLES_DBCV_FIGURE_PATH,
) -> None:
    _draw_min_samples_grid(
        run_table,
        column="dbcv_scaled_by_clustered_fraction",
        value_format=".3f",
        vmin=-1,
        vmax=1,
        colour_map="RdYlGn",
        title=f"HDBSCAN DBCV: {DATASET}",
        colour_bar_label="DBCV scaled by clustered fraction (-1 to 1)",
        scale_note=(
            "Each cell is one run. Colours use DBCV's full -1 to 1 range. Compare settings only "
            "within one row: it fixes records, embedding space and metric. DBCV includes the "
            "clustered fraction, so it and noise are not independent."
        ),
        output_path=output_path,
    )


def _agreement_figure_rows(
    agreement_table: pd.DataFrame,
    comparison: str,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    table = agreement_table[agreement_table["comparison"] == comparison].copy()
    if table.empty:
        raise ValueError(f"agreement figure has no {comparison} rows")
    if comparison == "full_vs_deduplicated":
        table["agreement_figure_key"] = [
            _run_key(str(row.right_space), str(row.right_metric))
            for row in table.itertuples(index=False)
        ]
        table["agreement_figure_label"] = [
            f"{SPACE_STYLES[str(row.right_space)][0]}, "
            f"{METRIC_LABELS[str(row.right_metric)]}"
            for row in table.itertuples(index=False)
        ]
    elif comparison in {"raw_vs_pca", "centred_vs_pca_cosine"}:
        table["agreement_figure_key"] = [
            _run_key(str(row.left_records), str(row.right_space))
            for row in table.itertuples(index=False)
        ]
        table["agreement_figure_label"] = [
            f"{str(row.left_records)}: {SPACE_STYLES[str(row.right_space)][0]}"
            for row in table.itertuples(index=False)
        ]
    elif comparison == "euclidean_vs_cosine":
        table["agreement_figure_key"] = [
            _run_key(str(row.left_records), str(row.left_space))
            for row in table.itertuples(index=False)
        ]
        table["agreement_figure_label"] = [
            f"{str(row.left_records)}: {SPACE_STYLES[str(row.left_space)][0]}"
            for row in table.itertuples(index=False)
        ]
    elif comparison == "uncentred_vs_centred_cosine":
        table["agreement_figure_key"] = table["left_records"].astype(str)
        table["agreement_figure_label"] = table["left_records"].astype(str)
    else:
        raise ValueError(f"unknown agreement figure comparison: {comparison}")

    label_counts = table.groupby("agreement_figure_key")["agreement_figure_label"].nunique()
    if not label_counts.eq(1).all():
        raise ValueError(f"agreement figure has inconsistent labels for {comparison}")
    row_specs = table[["agreement_figure_key", "agreement_figure_label"]].drop_duplicates()
    return (
        table,
        row_specs["agreement_figure_key"].tolist(),
        row_specs["agreement_figure_label"].tolist(),
    )


def _euclidean_cosine_clustered_coverage_rows(
    agreement_table: pd.DataFrame,
) -> pd.DataFrame:
    required_columns = {
        "comparison",
        "left_records",
        "right_records",
        "left_space",
        "right_space",
        "left_metric",
        "right_metric",
        "min_cluster_size",
        "min_samples",
        "n_common",
        "n_clustered_both",
    }
    missing_columns = required_columns - set(agreement_table.columns)
    if missing_columns:
        raise ValueError(
            "Euclidean-cosine coverage chart is missing columns: "
            f"{sorted(missing_columns)}"
        )

    table = agreement_table.loc[
        (agreement_table["comparison"] == "euclidean_vs_cosine")
        & (agreement_table["min_samples"] == BASELINE_MIN_SAMPLES)
    ].copy()
    numeric_columns = (
        "min_cluster_size",
        "min_samples",
        "n_common",
        "n_clustered_both",
    )
    for column in numeric_columns:
        table[column] = pd.to_numeric(table[column], errors="coerce")
        values = table[column].to_numpy(dtype=float)
        if not np.isfinite(values).all() or not np.equal(values, np.floor(values)).all():
            raise ValueError(f"Euclidean-cosine coverage chart has invalid {column} values")
        table[column] = table[column].astype(int)

    cosine_configurations = {
        (records, space)
        for _base, records, space, _dimensions, metric in BASE_GRID
        if metric == "cosine"
    }
    expected_keys = {
        (records, space, min_cluster_size)
        for _base, records, space, _dimensions, metric in BASE_GRID
        if metric == "euclidean" and (records, space) in cosine_configurations
        for min_cluster_size in MIN_CLUSTER_SIZES
    }
    observed_keys = set(
        table[["left_records", "left_space", "min_cluster_size"]].itertuples(
            index=False, name=None
        )
    )
    if len(table) != len(expected_keys) or observed_keys != expected_keys:
        raise ValueError(
            "Euclidean-cosine coverage chart expected one baseline row per configured "
            "record set, space and minimum cluster size"
        )
    if not (
        table["left_records"].eq(table["right_records"]).all()
        and table["left_space"].eq(table["right_space"]).all()
        and table["left_metric"].eq("euclidean").all()
        and table["right_metric"].eq("cosine").all()
    ):
        raise ValueError("Euclidean-cosine coverage chart has mismatched comparison metadata")

    expected_n_common = table["left_records"].map({
        "full": EXPECTED_FULL_N,
        "deduplicated": EXPECTED_DEDUP_N,
    })
    if not table["n_common"].eq(expected_n_common).all():
        raise ValueError("Euclidean-cosine coverage chart has an unexpected common-record count")
    if (table["n_clustered_both"] < 0).any() or (
        table["n_clustered_both"] > table["n_common"]
    ).any():
        raise ValueError("Euclidean-cosine coverage chart has an invalid clustered-both count")

    table["clustered_both_pct"] = 100 * table["n_clustered_both"] / table["n_common"]
    return table


def _draw_euclidean_cosine_clustered_coverage(
    agreement_table: pd.DataFrame,
    output_path: Path,
) -> None:
    table = _euclidean_cosine_clustered_coverage_rows(agreement_table)
    fig, axes = plt.subplots(1, 2, figsize=(15, 7), sharey=True)
    fig.suptitle(
        f"HDBSCAN Euclidean vs cosine coverage: {DATASET}, min_samples={BASELINE_MIN_SAMPLES}",
        fontsize=14,
        fontweight="bold",
    )
    positions = np.arange(len(EUCLIDEAN_SPACES), dtype=float)
    bar_width = 0.18
    offsets = (np.arange(len(MIN_CLUSTER_SIZES)) - (len(MIN_CLUSTER_SIZES) - 1) / 2) * bar_width
    colours = plt.get_cmap("viridis")(np.linspace(0.2, 0.8, len(MIN_CLUSTER_SIZES)))
    handles = []
    for axis, records in zip(axes, ("full", "deduplicated")):
        record_rows = table[table["left_records"] == records]
        n_common = EXPECTED_FULL_N if records == "full" else EXPECTED_DEDUP_N
        for index, min_cluster_size in enumerate(MIN_CLUSTER_SIZES):
            rows = record_rows[
                record_rows["min_cluster_size"] == min_cluster_size
            ].set_index("left_space")
            rows = rows.reindex(EUCLIDEAN_SPACES)
            if rows["clustered_both_pct"].isna().any():
                raise ValueError("Euclidean-cosine coverage chart is missing a bar value")
            bars = axis.bar(
                positions + offsets[index],
                rows["clustered_both_pct"],
                width=bar_width,
                color=colours[index],
                label=f"min_cluster_size={min_cluster_size}",
            )
            axis.bar_label(
                bars,
                labels=[f"n={count:,}" for count in rows["n_clustered_both"]],
                padding=2,
                rotation=90,
                fontsize=7,
            )
            if records == "full":
                handles.append(bars[0])
        axis.set(
            title=f"{records.capitalize()} records (n_common={n_common:,})",
            xlabel="embedding space",
            xticks=positions,
            xticklabels=[SPACE_STYLES[space][0] for space in EUCLIDEAN_SPACES],
            ylim=(0, 108),
        )
        axis.grid(axis="y", alpha=0.3)
        axis.set_axisbelow(True)
    axes[0].set_ylabel("records clustered by both runs (%)")
    fig.legend(
        handles=handles,
        labels=[f"min_cluster_size={size}" for size in MIN_CLUSTER_SIZES],
        loc="lower center",
        ncol=len(MIN_CLUSTER_SIZES),
        frameon=False,
    )
    fig.text(
        0.5,
        0.07,
        "Bar height is 100 x n_clustered_both / n_common. Labels give n_clustered_both. "
        "These are records assigned a non-noise label by both runs, not HDBSCAN core points.\n"
        "Euclidean and cosine use different HDBSCAN backends in this grid, so the comparison is "
        "metric plus backend rather than distance only.",
        ha="center",
        fontsize=8,
        style="italic",
        color="grey",
    )
    fig.subplots_adjust(left=0.08, right=0.98, top=0.86, bottom=0.24, wspace=0.12)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _draw_euclidean_cosine_clustered_coverage_from_csv(
    agreement_path: Path = AGREEMENT_PATH,
    output_path: Path = EUCLIDEAN_COSINE_COVERAGE_FIGURE_PATH,
) -> None:
    _draw_euclidean_cosine_clustered_coverage(pd.read_csv(agreement_path), output_path)


def _draw_partition_agreement(agreement_table: pd.DataFrame) -> None:
    table = agreement_table.copy()

    fig, axes = plt.subplots(5, 2, figsize=(14, 20), sharex=True)
    fig.text(
        0.5,
        0.97,
        f"HDBSCAN partition agreement: {DATASET}, min_samples={BASELINE_MIN_SAMPLES}",
        ha="center",
        fontsize=14,
        fontweight="bold",
    )
    has_negative_ari = bool(
        (table[["ari_all", "ari_clustered_both"]].to_numpy(dtype=float) < 0).any()
    )
    agreement_cmap = "coolwarm" if has_negative_ari else "viridis"
    agreement_vmin = -1 if has_negative_ari else 0

    def draw_panel(
        ax: Axes,
        comparison: str,
        score: str,
        title: str,
    ) -> AxesImage:
        subset, row_keys, row_labels = _agreement_figure_rows(table, comparison)
        values = _grid_values(
            subset, "agreement_figure_key", row_keys, score, f"{comparison} ARI"
        )
        secondary_values = (
            _grid_values(
                subset,
                "agreement_figure_key",
                row_keys,
                "n_clustered_both",
                f"{comparison} clustered-both count",
            )
            if score == "ari_clustered_both"
            else None
        )
        colour_map = plt.get_cmap(agreement_cmap).copy()
        colour_map.set_bad("#d9d9d9")
        image = ax.imshow(
            np.ma.masked_invalid(values),
            cmap=colour_map,
            vmin=agreement_vmin,
            vmax=1,
            aspect="auto",
        )
        _annotate_heatmap(ax, image, values, ".2f", secondary_values)
        ax.set_title(title)
        ax.set_yticks(np.arange(len(row_labels)), row_labels)
        ax.set_xticks(np.arange(len(MIN_CLUSTER_SIZES)), [str(size) for size in MIN_CLUSTER_SIZES])
        return image

    panel_specs = (
        (
            "full_vs_deduplicated",
            "1. Full vs row-removed clustering\nSame space and metric",
            "2. Full vs row-removed shared cluster cores\nOnly proteins clustered in both runs",
        ),
        (
            "raw_vs_pca",
            "3. Does Euclidean PCA change the overall clustering?\nRaw 1,024 vs PCA, including noise",
            "4. Does Euclidean PCA change shared cluster cores?\nOnly proteins clustered in both runs",
        ),
        (
            "euclidean_vs_cosine",
            "5. Euclidean vs cosine overall clustering\nSame coordinates; metric plus backend until parity passes",
            "6. Euclidean vs cosine shared cluster cores\nOnly proteins clustered in both runs",
        ),
        (
            "uncentred_vs_centred_cosine",
            "7. Does centring change cosine overall clustering?\nRaw 1,024 vs centred raw 1,024",
            "8. Does centring change cosine shared cluster cores?\nOnly proteins clustered in both runs",
        ),
        (
            "centred_vs_pca_cosine",
            "9. Does cosine dimension reduction\nchange the overall clustering?\nCentred raw 1,024 vs PCA, including noise",
            "10. Does cosine dimension reduction\nchange shared cluster cores?\nOnly proteins clustered in both runs",
        ),
    )
    image = None
    for row_index, (comparison, all_title, clustered_title) in enumerate(panel_specs):
        image = draw_panel(axes[row_index, 0], comparison, "ari_all", all_title)
        draw_panel(
            axes[row_index, 1], comparison, "ari_clustered_both", clustered_title
        )
    for ax in axes[-1, :]:
        ax.set_xlabel("minimum cluster size")
    if image is None:
        raise ValueError("agreement figure has no panels")
    colour_axis = fig.add_axes((0.92, 0.15, 0.018, 0.7))
    fig.colorbar(image, cax=colour_axis, label="adjusted Rand index")
    fig.text(
        0.5,
        0.025,
        "All-record ARI treats noise as one category. "
        "Clustered-both ARI removes noise from either run; its subset is selected by the clustering outcome.\n"
        "Panels 2, 4, 6, 8 and 10 show n_clustered_both in each cell.\n"
        f"Colour scale: {agreement_cmap}, {agreement_vmin} to 1.",
        ha="center",
        fontsize=8,
        style="italic",
        color="grey",
    )
    fig.subplots_adjust(left=0.25, right=0.86, top=0.94, bottom=0.10, hspace=0.56, wspace=0.62)
    AGREEMENT_FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(AGREEMENT_FIGURE_PATH, dpi=150)
    plt.close(fig)


def _reset_outputs() -> None:
    for path in (
        RUNS_PATH,
        CLUSTERS_PATH,
        AGREEMENT_PATH,
        PAIRED_PATH,
        FIGURE_PATH,
        METRIC_FIGURE_PATH,
        AGREEMENT_FIGURE_PATH,
        EUCLIDEAN_COSINE_COVERAGE_FIGURE_PATH,
        MIN_SAMPLES_FIGURE_PATH,
        MIN_SAMPLES_CLUSTERS_FIGURE_PATH,
        MIN_SAMPLES_DBCV_FIGURE_PATH,
        CONDENSED_TREE_FIGURE_PATH,
        *LEGACY_FIGURE_PATHS,
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.unlink(missing_ok=True)
    for directory in (ASSIGNMENTS_DIR, CONDENSED_TREE_PANEL_DIR):
        if directory.exists():
            shutil.rmtree(directory)
        directory.mkdir(parents=True)


def main() -> None:
    _reset_outputs()

    full = load_dataset(DATASET)
    dedup = deduplicate(full)
    if len(full) != EXPECTED_FULL_N or len(dedup) != EXPECTED_DEDUP_N:
        raise ValueError(
            f"grid expects {EXPECTED_FULL_N} full and {EXPECTED_DEDUP_N} deduplicated records; "
            f"loaded {len(full)} and {len(dedup)}"
        )

    full_spaces, retained = _spaces(full.X)
    full_maximum_exact_multiplicity = _maximum_exact_multiplicity(full.X)
    dedup_maximum_exact_multiplicity = _maximum_exact_multiplicity(dedup.X)
    if dedup_maximum_exact_multiplicity != 1:
        raise ValueError("deduplicated matrix still contains exact duplicate coordinates")
    maximum_exact_multiplicity_by_records = {
        "full": full_maximum_exact_multiplicity,
        "deduplicated": dedup_maximum_exact_multiplicity,
    }
    dedup_idx = _deduplicated_indices(full.ids, dedup.ids)
    if not np.array_equal(full.X[dedup_idx], dedup.X):
        raise ValueError("deduplicated rows do not match the full embedding matrix")
    dedup_spaces = {name: X[dedup_idx] for name, X in full_spaces.items()}
    datasets = {"full": full, "deduplicated": dedup}
    spaces_by_records = {"full": full_spaces, "deduplicated": dedup_spaces}
    _require_cosine_input_norms(spaces_by_records)

    run_rows: list[dict] = []
    cluster_rows: list[dict] = []
    assignments: dict[str, tuple[list[str], np.ndarray]] = {}
    condensed_tree_models: dict[int, hdbscan.HDBSCAN] = {}
    for (
        base_row,
        records,
        space,
        expected_dims,
        metric,
        min_cluster_size,
        min_samples,
    ) in GRID:
        row = _row_id(base_row, min_cluster_size, min_samples)
        ds = datasets[records]
        spaces = spaces_by_records[records]
        X = spaces[space]
        if X.shape != (len(ds), expected_dims):
            raise ValueError(f"row {row} expected {(len(ds), expected_dims)}, got {X.shape}")

        variance_retained = _variance_retained(records, space, expected_dims, retained)
        print(
            f"{row}: {records}, {space}, {metric}, {len(ds):,} x {X.shape[1]:,}, "
            f"min_cluster_size={min_cluster_size}, min_samples={min_samples}"
        )
        run_row, per_cluster, labels, model = _run_one(
            row=row,
            records=records,
            space=space,
            metric=metric,
            X=X,
            ids=ds.ids,
            subset_hash=ds.subset_hash,
            variance_retained=variance_retained,
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            maximum_exact_multiplicity=maximum_exact_multiplicity_by_records[records],
        )
        run_rows.append(run_row)
        cluster_rows.extend(per_cluster)
        assignments[row] = (ds.ids, labels)
        if (
            records == "deduplicated"
            and space == "raw1024"
            and metric == "euclidean"
            and min_samples == BASELINE_MIN_SAMPLES
        ):
            condensed_tree_models[min_cluster_size] = model
        print(
            f"   {run_row['n_clusters']} clusters, {run_row['n_noise']:,} noise "
            f"({run_row['noise_pct']:.2f}%)"
        )

    run_table = pd.DataFrame(run_rows)
    cluster_table = pd.DataFrame(cluster_rows)
    agreement_table = _partition_agreement_table(assignments, run_table)
    paired_table = _min_samples_paired_table(run_table)
    expected_agreements = len(_partition_agreement_pairs())
    if len(run_table) != len(GRID) or not run_table["row"].is_unique:
        raise ValueError("run table does not contain exactly one row per grid setting")
    expected_row_ids = run_table.apply(
        lambda run: _expected_row_id(
            run["records"],
            run["space"],
            run["metric"],
            int(run["min_cluster_size"]),
            int(run["min_samples"]),
        ),
        axis=1,
    )
    if not expected_row_ids.equals(run_table["row"]):
        raise ValueError("run table row ID does not match its recorded configuration")
    if cluster_table.duplicated(["row", "cluster"]).any():
        raise ValueError("cluster table contains duplicate row/cluster pairs")
    if len(agreement_table) != expected_agreements:
        raise ValueError(
            f"agreement table expected {expected_agreements} rows, got {len(agreement_table)}"
        )
    RUNS_PATH.parent.mkdir(parents=True, exist_ok=True)
    run_table.to_csv(RUNS_PATH, index=False)
    cluster_table.to_csv(CLUSTERS_PATH, index=False)
    agreement_table.to_csv(AGREEMENT_PATH, index=False)
    paired_table.to_csv(PAIRED_PATH, index=False)
    _update_min_samples_markdown_from_csv()
    baseline_run_table = run_table[run_table["min_samples"] == BASELINE_MIN_SAMPLES]
    baseline_cluster_table = cluster_table[
        cluster_table["min_samples"] == BASELINE_MIN_SAMPLES
    ]
    baseline_agreement_table = agreement_table[
        agreement_table["min_samples"] == BASELINE_MIN_SAMPLES
    ]
    expected_baseline_runs = len(BASE_GRID) * len(MIN_CLUSTER_SIZES)
    expected_baseline_agreements = len(_partition_agreement_pairs(
        min_samples_values=(BASELINE_MIN_SAMPLES,)
    ))
    if (
        len(baseline_run_table) != expected_baseline_runs
        or len(baseline_agreement_table) != expected_baseline_agreements
    ):
        raise ValueError("baseline figures do not have the expected run or agreement rows")
    _draw_results(baseline_run_table, baseline_cluster_table)
    _draw_metric_heatmaps(baseline_run_table)
    _draw_partition_agreement(baseline_agreement_table)
    _draw_euclidean_cosine_clustered_coverage_from_csv()
    _draw_min_samples_noise(run_table)
    _draw_min_samples_clusters(run_table)
    _draw_min_samples_dbcv(run_table)
    _draw_deduplicated_raw1024_condensed_trees(condensed_tree_models)
    condensed_tree_panel_paths = _draw_deduplicated_raw1024_condensed_tree_panels(
        condensed_tree_models
    )
    print(f"wrote {RUNS_PATH.relative_to(REPO)}")
    print(f"wrote {CLUSTERS_PATH.relative_to(REPO)}")
    print(f"wrote {AGREEMENT_PATH.relative_to(REPO)}")
    print(f"wrote {PAIRED_PATH.relative_to(REPO)}")
    print(f"wrote assignments under {ASSIGNMENTS_DIR.relative_to(REPO)}")
    print(f"wrote {FIGURE_PATH.relative_to(REPO)}")
    print(f"wrote {METRIC_FIGURE_PATH.relative_to(REPO)}")
    print(f"wrote {AGREEMENT_FIGURE_PATH.relative_to(REPO)}")
    print(f"wrote {EUCLIDEAN_COSINE_COVERAGE_FIGURE_PATH.relative_to(REPO)}")
    print(f"wrote {MIN_SAMPLES_FIGURE_PATH.relative_to(REPO)}")
    print(f"wrote {MIN_SAMPLES_CLUSTERS_FIGURE_PATH.relative_to(REPO)}")
    print(f"wrote {MIN_SAMPLES_DBCV_FIGURE_PATH.relative_to(REPO)}")
    print(f"wrote {CONDENSED_TREE_FIGURE_PATH.relative_to(REPO)}")
    for path in condensed_tree_panel_paths:
        print(f"wrote {path.relative_to(REPO)}")


if __name__ == "__main__":
    if sys.argv[1:] == ["--update-min-samples-summary"]:
        _update_min_samples_markdown_from_csv()
    elif sys.argv[1:] == ["--draw-euclidean-cosine-coverage"]:
        _draw_euclidean_cosine_clustered_coverage_from_csv()
        print(f"wrote {EUCLIDEAN_COSINE_COVERAGE_FIGURE_PATH.relative_to(REPO)}")
    elif sys.argv[1:]:
        raise SystemExit(
            "usage: hdbscan_standalone.py "
            "[--update-min-samples-summary | --draw-euclidean-cosine-coverage]"
        )
    else:
        main()
