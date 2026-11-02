from __future__ import annotations

import argparse
import importlib.metadata
import shutil
import subprocess
import sys
import warnings
from collections import Counter
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    pairwise_distances,
    silhouette_score,
)
from sklearn.preprocessing import normalize

from data import REPO, Dataset, deduplicate, load_dataset


DATASET = "phosphatase"
EXPECTED_SKLEARN_VERSION = "1.9.0"
EXPECTED_FULL_N = 1_587
EXPECTED_DEDUP_N = 1_480
PCA_DIMS = (113, 67, 34)
GEOMETRIES = (
    "raw1024", "pca113", "pca67", "pca34", "l2_raw1024",
)
SEEDS = tuple(range(10))
K_VALUES = tuple(range(2, 51))
ENDPOINTS = (30, 40, 50)
PRIMARY_ENDPOINT = 40
KNEE_MIN_DEVIATION = 0.05
CONTROL_PERMUTATIONS = 5
GEOMETRY_CODES = {
    "raw1024": "R1024",
    "pca113": "P113",
    "pca67": "P067",
    "pca34": "P034",
    "l2_raw1024": "L2R1024",
}
RECORD_CODES = {"full": "F", "deduplicated": "D"}
ROLE_CODES = {"procedure_selected": "P", "fixed_k_stability": "K"}
PROTSPACE_SRC = REPO / "external" / "protspace" / "apps" / "protspace" / "src"
DIMENSIONS = {
    "raw1024": 1_024,
    "pca113": 113,
    "pca67": 67,
    "pca34": 34,
    "l2_raw1024": 1_024,
}
LIBRARY_VERSIONS = {
    "numpy_version": importlib.metadata.version("numpy"),
    "pandas_version": importlib.metadata.version("pandas"),
    "sklearn_version": importlib.metadata.version("scikit-learn"),
    "matplotlib_version": importlib.metadata.version("matplotlib"),
}

KMEANS_ARGUMENTS = {
    "init": "k-means++",
    "n_init": 10,
    "max_iter": 300,
    "tol": 1e-4,
    "algorithm": "lloyd",
    "copy_x": True,
    "verbose": 0,
}


@dataclass(frozen=True)
class OutputPaths:
    root: Path
    sweep: Path
    selection: Path
    runs: Path
    clusters: Path
    agreement: Path
    crosscheck: Path
    assignments: Path
    elbow_figure: Path
    frequency_figure: Path
    stability_figure: Path


def _output_paths(root: Path | None = None) -> OutputPaths:
    result_root = root if root is not None else REPO / "results" / "kmeans"
    metrics = result_root / "metrics"
    figures = result_root / "figures"
    return OutputPaths(
        root=result_root,
        sweep=metrics / "kmeans_standalone_elbow_sweep.csv",
        selection=metrics / "kmeans_standalone_elbow_selection.csv",
        runs=metrics / "kmeans_standalone_runs.csv",
        clusters=metrics / "kmeans_standalone_clusters.csv",
        agreement=metrics / "kmeans_standalone_partition_agreement.csv",
        crosscheck=metrics / "kmeans_protspace_crosscheck.csv",
        assignments=result_root / "assignments" / "kmeans_standalone",
        elbow_figure=figures / "kmeans_standalone_elbow.png",
        frequency_figure=figures / "kmeans_standalone_selected_k_frequency.png",
        stability_figure=figures / "kmeans_standalone_stability.png",
    )


def _git_sha() -> str:
    try:
        revision = subprocess.run(
            ["git", "describe", "--always"],
            cwd=REPO,
            capture_output=True,
            text=True,
            check=True,
        )
        status = subprocess.run(
            ["git", "status", "--porcelain", "--", ".", ":(exclude)results/kmeans/**"],
            cwd=REPO,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    git_sha = revision.stdout.strip() or "unknown"
    return f"{git_sha}-dirty" if status.stdout.strip() else git_sha


def _require_sklearn_version() -> None:
    installed = importlib.metadata.version("scikit-learn")
    if installed != EXPECTED_SKLEARN_VERSION:
        raise RuntimeError(
            f"k-means requires scikit-learn {EXPECTED_SKLEARN_VERSION}, found {installed}"
        )


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
    if raw.shape != (EXPECTED_FULL_N, 1_024) or not np.isfinite(raw).all():
        raise ValueError(
            f"expected finite full coordinates shaped {(EXPECTED_FULL_N, 1_024)}, got {raw.shape}"
        )

    pca = PCA(n_components=max(PCA_DIMS), svd_solver="full", whiten=False)
    transformed = pca.fit_transform(raw)
    full_spaces = {
        "raw1024": raw,
        **{f"pca{dimensions}": transformed[:, :dimensions] for dimensions in PCA_DIMS},
        "l2_raw1024": normalize(raw, norm="l2", axis=1, copy=True),
    }
    retained = {
        dimensions: float(pca.explained_variance_ratio_[:dimensions].sum())
        for dimensions in PCA_DIMS
    }

    positions = _deduplicated_indices(full.ids, deduplicated.ids)
    if len(deduplicated) != EXPECTED_DEDUP_N:
        raise ValueError(f"expected {EXPECTED_DEDUP_N} deduplicated records, got {len(deduplicated)}")
    if not np.array_equal(full.X[positions], deduplicated.X):
        raise ValueError("deduplicated coordinates do not match the full matrix")
    spaces_by_records = {
        "full": full_spaces,
        "deduplicated": {
            geometry: np.ascontiguousarray(coordinates[positions], dtype=np.float64)
            for geometry, coordinates in full_spaces.items()
        },
    }
    _validate_spaces(spaces_by_records)
    return spaces_by_records, retained


def _validate_spaces(spaces_by_records: dict[str, dict[str, np.ndarray]]) -> None:
    expected_n = {"full": EXPECTED_FULL_N, "deduplicated": EXPECTED_DEDUP_N}
    expected_dims = {
        "raw1024": 1_024,
        "pca113": 113,
        "pca67": 67,
        "pca34": 34,
        "l2_raw1024": 1_024,
    }
    for records, spaces in spaces_by_records.items():
        if tuple(spaces) != GEOMETRIES:
            raise ValueError(f"{records} geometries do not match the fixed grid")
        for geometry, dimensions in expected_dims.items():
            coordinates = spaces[geometry]
            if coordinates.shape != (expected_n[records], dimensions):
                raise ValueError(f"unexpected shape for {records}/{geometry}: {coordinates.shape}")
            if not np.isfinite(coordinates).all():
                raise ValueError(f"non-finite coordinates in {records}/{geometry}")
        if not np.array_equal(spaces["pca113"][:, :67], spaces["pca67"]):
            raise ValueError(f"{records} PCA 67 is not nested in PCA 113")
        if not np.array_equal(spaces["pca113"][:, :34], spaces["pca34"]):
            raise ValueError(f"{records} PCA 34 is not nested in PCA 113")
        if not np.allclose(np.linalg.norm(spaces["l2_raw1024"], axis=1), 1.0):
            raise ValueError(f"{records} L2 coordinates are not unit norm")


def _canonical_coordinates(X: np.ndarray, ids: list[str]) -> tuple[np.ndarray, np.ndarray]:
    if len(ids) != len(X) or len(ids) != len(set(ids)):
        raise ValueError("KMeans requires one unique protein ID per coordinate row")
    canonical = np.argsort(np.asarray(ids), kind="stable")
    inverse = np.empty(len(ids), dtype=int)
    inverse[canonical] = np.arange(len(ids))
    return np.ascontiguousarray(X[canonical], dtype=np.float64), inverse


def _fit_kmeans(
    X: np.ndarray,
    ids: list[str],
    k: int,
    seed: int,
) -> tuple[KMeans, np.ndarray]:
    canonical_X, inverse = _canonical_coordinates(X, ids)
    model = KMeans(
        n_clusters=k,
        random_state=seed,
        **KMEANS_ARGUMENTS,
    ).fit(canonical_X)
    return model, np.asarray(model.labels_[inverse], dtype=np.int32)


def chord_deviation(inertia: np.ndarray) -> np.ndarray:
    values = np.asarray(inertia, dtype=float)
    if values.ndim != 1:
        raise ValueError("chord deviation requires a one-dimensional inertia curve")
    if len(values) < 3:
        return np.zeros(len(values))
    x = np.linspace(0.0, 1.0, len(values))
    span = max(float(values.max() - values.min()), 1e-12)
    normalised = (values - values.min()) / span
    denominator = max(float(np.hypot(1.0, normalised[-1] - normalised[0])), 1e-12)
    return np.abs(
        (normalised[-1] - normalised[0]) * x
        - normalised
        + normalised[0]
    ) / denominator


def _selection(inertia: np.ndarray, k_values: tuple[int, ...]) -> dict[str, int | float | str]:
    deviations = chord_deviation(inertia)
    order = np.lexsort((np.asarray(k_values), -deviations))
    winner, runner_up = int(order[0]), int(order[1])
    maximum = float(deviations[winner])
    runner_up_deviation = float(deviations[runner_up])
    return {
        "selected_k": int(k_values[winner]),
        "maximum_chord_deviation": maximum,
        "runner_up_k": int(k_values[runner_up]),
        "runner_up_chord_deviation": runner_up_deviation,
        "deviation_margin": maximum - runner_up_deviation,
        "knee_confidence": "high" if maximum >= KNEE_MIN_DEVIATION else "low",
    }


def _sweep_table(
    full_spaces: dict[str, np.ndarray],
    ids: list[str],
    seeds: tuple[int, ...] = SEEDS,
    k_values: tuple[int, ...] = K_VALUES,
    endpoints: tuple[int, ...] = ENDPOINTS,
    progress: bool = False,
) -> pd.DataFrame:
    rows = []
    for geometry, X in full_spaces.items():
        for seed in seeds:
            started = perf_counter()
            inertia = []
            for k in k_values:
                model, _labels = _fit_kmeans(X, ids, k, seed)
                inertia.append(float(model.inertia_))
                rows.append({"geometry": geometry, "seed": seed, "k": k, "inertia": inertia[-1]})
            seed_start = len(rows) - len(k_values)
            for endpoint in endpoints:
                eligible = [index for index, k in enumerate(k_values) if k <= endpoint]
                if len(eligible) < 3:
                    raise ValueError(f"endpoint {endpoint} leaves fewer than three K values")
                deviations = chord_deviation(np.asarray(inertia)[eligible])
                column = f"chord_deviation_kmax_{endpoint}"
                for index, deviation in zip(eligible, deviations, strict=True):
                    rows[seed_start + index][column] = float(deviation)
            if progress:
                print(
                    f"sweep {geometry} seed {seed}: {len(k_values)} fits in "
                    f"{perf_counter() - started:.3f}s",
                    flush=True,
                )
    return pd.DataFrame(rows)


def _selection_table(
    sweep: pd.DataFrame,
    endpoints: tuple[int, ...] = ENDPOINTS,
) -> pd.DataFrame:
    rows = []
    for (geometry, seed), group in sweep.groupby(["geometry", "seed"], sort=False):
        seed_number = int(str(seed))
        ordered = group.sort_values("k", kind="stable")
        for endpoint in endpoints:
            selected = ordered[ordered["k"] <= endpoint]
            result = _selection(
                selected["inertia"].to_numpy(dtype=float),
                tuple(selected["k"].astype(int)),
            )
            rows.append({
                "selection_id": f"KS-{geometry}-S{seed_number:02d}-E{endpoint:02d}",
                "geometry": geometry,
                "seed": seed_number,
                "endpoint": endpoint,
                "is_primary": endpoint == PRIMARY_ENDPOINT,
                **result,
            })
    return pd.DataFrame(rows)


def _modal_k(selection: pd.DataFrame) -> dict[str, int]:
    primary = selection[selection["is_primary"]]
    result = {}
    for geometry, group in primary.groupby("geometry", sort=False):
        counts = Counter(group["selected_k"].astype(int))
        highest = max(counts.values())
        result[str(geometry)] = min(k for k, count in counts.items() if count == highest)
    return result


def _fit_phase_estimate(timings: dict[str, float]) -> dict[str, float]:
    seconds_per_matched_set = sum(timings.values())
    return {
        "sweep_seconds": len(SEEDS) * len(K_VALUES) * seconds_per_matched_set,
        "procedure_seconds": 2 * len(SEEDS) * seconds_per_matched_set,
        "maximum_fixed_k_seconds": 2 * (len(SEEDS) - 1) * seconds_per_matched_set,
    }


def _fit_partition(
    X: np.ndarray,
    ids: list[str],
    k: int,
    seed: int,
) -> tuple[KMeans, np.ndarray, str]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model, labels = _fit_kmeans(X, ids, k, seed)
    unexpected = [warning for warning in caught if warning.category is not ConvergenceWarning]
    if unexpected:
        messages = " | ".join(str(warning.message) for warning in unexpected)
        raise RuntimeError(f"unexpected warning while fitting KMeans: {messages}")
    warning_text = " | ".join(
        str(warning.message) for warning in caught if warning.category is ConvergenceWarning
    )
    return model, labels, warning_text


def _validate_partition(model: KMeans, labels: np.ndarray, n: int) -> int:
    labels = np.asarray(labels)
    if labels.shape != (n,) or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("KMeans labels must be a one-dimensional integer array of dataset length")
    cluster_ids = np.unique(labels)
    realised_k = len(cluster_ids)
    if not np.array_equal(cluster_ids, np.arange(realised_k)):
        raise ValueError("KMeans labels must be contiguous from zero")
    requested_k = int(model.get_params(deep=False)["n_clusters"])
    if model.cluster_centers_.shape != (requested_k, model.n_features_in_):
        raise ValueError("KMeans centres do not match the requested model shape")
    if not np.isfinite(model.cluster_centers_).all() or not np.isfinite(model.inertia_):
        raise ValueError("KMeans centres and inertia must be finite")
    return realised_k


def _internal_scores(X: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    scores = {
        "silhouette": float(silhouette_score(X, labels, metric="euclidean")),
        "calinski_harabasz": float(calinski_harabasz_score(X, labels)),
        "davies_bouldin": float(davies_bouldin_score(X, labels)),
    }
    if not all(np.isfinite(value) for value in scores.values()):
        raise ValueError("all internal scores must be finite")
    return scores


def _control_summaries(X: np.ndarray, labels: np.ndarray, seed: int) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    controls = [
        _internal_scores(X, rng.permutation(labels))
        for _ in range(CONTROL_PERMUTATIONS)
    ]
    summaries = {}
    for metric in ("silhouette", "calinski_harabasz", "davies_bouldin"):
        values = np.asarray([control[metric] for control in controls], dtype=float)
        summaries.update({
            f"control_{metric}_minimum": float(values.min()),
            f"control_{metric}_median": float(np.median(values)),
            f"control_{metric}_maximum": float(values.max()),
            f"control_{metric}_range": float(np.ptp(values)),
        })
    return summaries


def _davies_bouldin_terms(X: np.ndarray, labels: np.ndarray) -> np.ndarray:
    cluster_ids = np.unique(labels)
    clusters = [X[labels == cluster_id] for cluster_id in cluster_ids]
    centroids = np.asarray([cluster.mean(axis=0) for cluster in clusters])
    dispersions = np.asarray([
        pairwise_distances(cluster, centroid[None, :], metric="euclidean").mean()
        for cluster, centroid in zip(clusters, centroids, strict=True)
    ])
    centroid_distances = pairwise_distances(centroids, metric="euclidean")
    if np.allclose(dispersions, 0.0) or np.allclose(centroid_distances, 0.0):
        terms = np.zeros(len(cluster_ids))
    else:
        centroid_distances[centroid_distances == 0] = np.inf
        terms = np.max(
            (dispersions[:, None] + dispersions[None, :]) / centroid_distances,
            axis=1,
        )
    expected = davies_bouldin_score(X, labels)
    if not np.isclose(terms.mean(), expected, rtol=1e-12, atol=1e-12):
        raise ValueError("per-cluster Davies-Bouldin terms do not reproduce sklearn's score")
    return terms


def _cluster_table(
    X: np.ndarray,
    labels: np.ndarray,
    model: KMeans,
    partition: dict[str, object],
) -> pd.DataFrame:
    terms = _davies_bouldin_terms(X, labels)
    rows = []
    for cluster_id in np.unique(labels):
        members = X[labels == cluster_id]
        centroid = model.cluster_centers_[cluster_id]
        radii = np.linalg.norm(members - centroid, axis=1)
        rows.append({
            **partition,
            "cluster": int(cluster_id),
            "size": len(members),
            "fraction": len(members) / len(X),
            "squared_error_sum": float(np.square(radii).sum()),
            "mean_radius": float(radii.mean()),
            "maximum_radius": float(radii.max()),
            "centroid_norm": float(np.linalg.norm(centroid)),
            "davies_bouldin_worst_pair_ratio": float(terms[cluster_id]),
        })
    table = pd.DataFrame(rows)
    if int(table["size"].sum()) != len(X):
        raise ValueError("cluster sizes do not reproduce the partition size")
    if not np.isclose(
        table["squared_error_sum"].sum(),
        model.inertia_,
        rtol=1e-10,
        atol=1e-8,
    ):
        raise ValueError("cluster squared errors do not reproduce KMeans inertia")
    numeric = table.select_dtypes(include=np.number)
    if not np.isfinite(numeric.to_numpy()).all():
        raise ValueError("all cluster diagnostics must be finite")
    return table


def _fit_and_score(
    X: np.ndarray,
    ids: list[str],
    geometry: str,
    records: str,
    seed: int,
    k: int,
    role: str,
    selection_id: str,
) -> tuple[dict[str, object], pd.DataFrame, dict[str, np.ndarray]]:
    model, labels, warning_text = _fit_partition(X, ids, k, seed)
    realised_k = _validate_partition(model, labels, len(X))
    if not 2 <= realised_k < len(X):
        raise ValueError(
            f"internal scores are undefined: requested_k={k}, realised_k={realised_k}, "
            f"n={len(X)}, iterations={model.n_iter_}, inertia={model.inertia_}, "
            f"convergence_warning={warning_text!r}"
        )
    scores = _internal_scores(X, labels)
    row_id = _partition_id(geometry, records, seed, k, role)
    partition = {
        "row": row_id,
        "geometry": geometry,
        "records": records,
        "seed": seed,
        "requested_k": k,
        "role": role,
    }
    run = {
        **partition,
        "selection_id": selection_id,
        "n": len(X),
        "realised_k": realised_k,
        "cluster_count_shortfall": realised_k != k,
        "n_iter": int(model.n_iter_),
        "convergence_warning": warning_text,
        "inertia": float(model.inertia_),
        "silhouette_n_scored": len(X),
        "calinski_harabasz_n_scored": len(X),
        "davies_bouldin_n_scored": len(X),
        **scores,
        **_control_summaries(X, labels, seed),
    }
    clusters = _cluster_table(X, labels, model, partition)
    payload = {
        "protein_id": np.asarray(ids, dtype=str),
        "label": np.asarray(labels, dtype=np.int32),
    }
    return run, clusters, payload


def _partition_id(geometry: str, records: str, seed: int, k: int, role: str) -> str:
    try:
        geometry_code = GEOMETRY_CODES[geometry]
        record_code = RECORD_CODES[records]
        role_code = ROLE_CODES[role]
    except KeyError as exc:
        raise ValueError(f"unsupported partition axis: {exc.args[0]}") from exc
    if seed < 0 or k < 2:
        raise ValueError("partition IDs require a non-negative seed and K of at least two")
    return f"KM-{geometry_code}-{record_code}-S{seed:02d}-K{k:02d}-{role_code}"


def _partition_tables(
    spaces_by_records: dict[str, dict[str, np.ndarray]],
    ids_by_records: dict[str, list[str]],
    selection: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict[str, np.ndarray]], int]:
    primary = selection[selection["is_primary"]].copy()
    if primary.duplicated(["geometry", "seed"]).any():
        raise ValueError("primary selections must contain one row per geometry and seed")
    fixed_k = _modal_k(primary)
    runs = []
    clusters = []
    payloads = {}

    def add_partition(row: pd.Series, records: str, k: int, role: str) -> None:
        geometry = str(row["geometry"])
        seed = int(row["seed"])
        run, cluster_table, payload = _fit_and_score(
            spaces_by_records[records][geometry],
            ids_by_records[records],
            geometry,
            records,
            seed,
            k,
            role,
            str(row["selection_id"]),
        )
        key = str(run["row"])
        if key in payloads:
            raise ValueError(f"duplicate in-memory partition key: {key}")
        runs.append(run)
        clusters.append(cluster_table)
        payloads[key] = payload

    for _, row in primary.iterrows():
        selected_k = int(row["selected_k"])
        for records in ("full", "deduplicated"):
            add_partition(row, records, selected_k, "procedure_selected")
    varying = primary[primary.apply(
        lambda row: int(row["selected_k"]) != fixed_k[str(row["geometry"])],
        axis=1,
    )]
    for _, row in varying.iterrows():
        for records in ("full", "deduplicated"):
            add_partition(
                row,
                records,
                fixed_k[str(row["geometry"])],
                "fixed_k_stability",
            )
    q = len(varying)
    if not 0 <= q <= 45 or len(runs) != 2 * (len(primary) + q):
        raise ValueError("partition counts do not match the selected-K table")
    return pd.DataFrame(runs), pd.concat(clusters, ignore_index=True), payloads, q


def _aligned_ari(
    left: dict[str, np.ndarray],
    right: dict[str, np.ndarray],
) -> tuple[int, float]:
    frames = []
    for side, payload in (("left", left), ("right", right)):
        ids = np.asarray(payload["protein_id"], dtype=str)
        labels = np.asarray(payload["label"])
        if ids.ndim != 1 or labels.shape != ids.shape or len(np.unique(ids)) != len(ids):
            raise ValueError(f"{side} partition requires unique aligned protein IDs and labels")
        frames.append(pd.DataFrame({"protein_id": ids, side: labels}))
    aligned = frames[0].merge(frames[1], on="protein_id", how="inner", validate="one_to_one")
    if aligned.empty:
        raise ValueError("partitions have no protein IDs in common")
    return len(aligned), float(adjusted_rand_score(aligned["left"], aligned["right"]))


def _agreement_table(
    runs: pd.DataFrame,
    payloads: dict[str, dict[str, np.ndarray]],
) -> pd.DataFrame:
    if runs["row"].duplicated().any() or set(runs["row"]) != set(payloads):
        raise ValueError("run rows and in-memory partition payloads must match exactly")
    rows = []

    def add(comparison: str, left: pd.Series, right: pd.Series, geometry: str, records: str) -> None:
        n_common, adjusted_rand = _aligned_ari(
            payloads[str(left["row"])],
            payloads[str(right["row"])],
        )
        rows.append({
            "comparison": comparison,
            "geometry": geometry,
            "records": records,
            "left_row": str(left["row"]),
            "right_row": str(right["row"]),
            "n_common": n_common,
            "adjusted_rand": adjusted_rand,
        })

    procedure = runs[runs["role"] == "procedure_selected"]
    for (geometry, records), group in procedure.groupby(["geometry", "records"], sort=False):
        ordered = [row for _, row in group.sort_values("seed", kind="stable").iterrows()]
        for left, right in combinations(ordered, 2):
            add("procedure_selected_stability", left, right, str(geometry), str(records))

    full_procedure = procedure[procedure["records"] == "full"]
    for geometry, group in full_procedure.groupby("geometry", sort=False):
        selected_counts = Counter(group["requested_k"].astype(int))
        if len(selected_counts) == 1:
            continue
        highest = max(selected_counts.values())
        fixed_k = min(k for k, count in selected_counts.items() if count == highest)
        for records in ("full", "deduplicated"):
            candidates = runs[
                (runs["geometry"] == geometry)
                & (runs["records"] == records)
                & (runs["requested_k"] == fixed_k)
            ]
            chosen = []
            for seed in sorted(group["seed"].astype(int)):
                seed_rows = candidates[candidates["seed"] == seed]
                if len(seed_rows) != 1:
                    raise ValueError(
                        f"fixed-K assembly expected one {geometry}/{records}/seed-{seed} partition"
                    )
                chosen.append(seed_rows.iloc[0])
            for left, right in combinations(chosen, 2):
                add("fixed_k_stability", left, right, str(geometry), records)

    for _, left in procedure[procedure["records"] == "full"].iterrows():
        matched = procedure[
            (procedure["geometry"] == left["geometry"])
            & (procedure["records"] == "deduplicated")
            & (procedure["seed"] == left["seed"])
        ]
        if len(matched) != 1:
            raise ValueError("full-versus-deduplicated comparison requires one matched partition")
        add(
            "full_vs_deduplicated",
            left,
            matched.iloc[0],
            str(left["geometry"]),
            "full_vs_deduplicated",
        )

    for records in ("full", "deduplicated"):
        raw = procedure[(procedure["geometry"] == "raw1024") & (procedure["records"] == records)]
        l2 = procedure[
            (procedure["geometry"] == "l2_raw1024") & (procedure["records"] == records)
        ]
        if len(raw) != len(l2):
            raise ValueError("raw-versus-L2 comparison requires matching seed coverage")
        l2_by_seed = {int(row["seed"]): row for _, row in l2.iterrows()}
        for _, left in raw.iterrows():
            seed = int(left["seed"])
            if seed not in l2_by_seed:
                raise ValueError(f"raw-versus-L2 comparison is missing seed {seed}")
            add("raw_vs_l2", left, l2_by_seed[seed], "raw_vs_l2", records)

    agreement = pd.DataFrame(rows)
    if agreement["adjusted_rand"].isna().any() or not agreement["adjusted_rand"].between(-1, 1).all():
        raise ValueError("all adjusted Rand scores must be finite and lie between -1 and one")
    return agreement


def _agreement_summary(agreement: pd.DataFrame) -> pd.DataFrame:
    summary = (
        agreement.groupby(["comparison", "geometry", "records"], sort=False)["adjusted_rand"]
        .agg(["count", "median", "min", "max"])
        .reset_index()
        .rename(columns={"min": "minimum", "max": "maximum"})
    )
    summary["range"] = summary["maximum"] - summary["minimum"]
    return summary.drop(columns="maximum")


def _tables_with_provenance(
    sweep: pd.DataFrame,
    selection: pd.DataFrame,
    runs: pd.DataFrame,
    clusters: pd.DataFrame,
    agreement: pd.DataFrame,
    crosscheck: pd.DataFrame,
    datasets: dict[str, Dataset],
    variance_retained: dict[int, float],
) -> dict[str, pd.DataFrame]:
    git_sha = _git_sha()
    common = {"git_sha": git_sha, **LIBRARY_VERSIONS}
    full = datasets["full"]

    sweep_output = sweep.copy()
    sweep_output["dataset"] = DATASET
    sweep_output["records"] = "full"
    sweep_output["n"] = len(full)
    sweep_output["subset_hash"] = full.subset_hash
    sweep_output["space"] = sweep_output["geometry"]
    sweep_output["n_dims"] = sweep_output["geometry"].map(DIMENSIONS)
    sweep_output["input_normalisation"] = sweep_output["geometry"].map(
        lambda geometry: "l2" if geometry == "l2_raw1024" else "none"
    )
    sweep_output["endpoints"] = "|".join(str(endpoint) for endpoint in ENDPOINTS)
    sweep_output = sweep_output.assign(**common)

    selection_output = selection.copy()
    selection_output["dataset"] = DATASET
    selection_output["records"] = "full"
    selection_output["n"] = len(full)
    selection_output["subset_hash"] = full.subset_hash
    selection_output["space"] = selection_output["geometry"]
    selection_output["n_dims"] = selection_output["geometry"].map(DIMENSIONS)
    selection_output = selection_output.assign(**common)

    runs_output = runs.copy()
    runs_output["dataset"] = DATASET
    runs_output["subset_hash"] = runs_output["records"].map(
        {records: dataset.subset_hash for records, dataset in datasets.items()}
    )
    runs_output["space"] = runs_output["geometry"]
    runs_output["n_dims"] = runs_output["geometry"].map(DIMENSIONS)
    runs_output["variance_retained"] = runs_output["geometry"].map({
        f"pca{dimensions}": retained for dimensions, retained in variance_retained.items()
    })
    runs_output["input_normalisation"] = runs_output["geometry"].map(
        lambda geometry: "l2" if geometry == "l2_raw1024" else "none"
    )
    runs_output["clustering_metric"] = "euclidean"
    runs_output["scoring_metric"] = "euclidean"
    runs_output["cluster_seed"] = runs_output["seed"]
    runs_output["control_permutations"] = CONTROL_PERMUTATIONS
    runs_output["kmeans_n_clusters"] = runs_output["requested_k"]
    runs_output["kmeans_random_state"] = runs_output["seed"]
    for argument, value in KMEANS_ARGUMENTS.items():
        runs_output[f"kmeans_{argument}"] = value
    runs_output["archive"] = runs_output["row"].map(lambda row: f"{row}.npz")
    runs_output = runs_output.assign(**common)

    run_metadata = runs_output.set_index("row")[["dataset", "subset_hash"]]
    clusters_output = clusters.copy().join(run_metadata, on="row")
    clusters_output = clusters_output.assign(**common)
    agreement_output = agreement.copy().assign(dataset=DATASET, **common)
    crosscheck_output = crosscheck.copy().assign(dataset=DATASET, **common)
    return {
        "sweep": sweep_output,
        "selection": selection_output,
        "runs": runs_output,
        "clusters": clusters_output,
        "agreement": agreement_output,
        "crosscheck": crosscheck_output,
    }


def _reset_outputs(paths: OutputPaths) -> None:
    for directory in (paths.sweep.parent, paths.elbow_figure.parent, paths.assignments):
        if directory.exists():
            shutil.rmtree(directory)
        directory.mkdir(parents=True)


def _plot_elbow(sweep: pd.DataFrame, selection: pd.DataFrame, path: Path) -> None:
    geometries = list(dict.fromkeys(sweep["geometry"].astype(str)))
    fig, axes = plt.subplots(
        1,
        len(geometries),
        figsize=(4 * len(geometries), 3.6),
        squeeze=False,
        sharey=False,
    )
    for axis, geometry in zip(axes[0], geometries, strict=True):
        subset = sweep[sweep["geometry"] == geometry]
        for _seed, curve in subset.groupby("seed", sort=True):
            ordered = curve.sort_values("k", kind="stable")
            axis.plot(ordered["k"], ordered["inertia"], color="#4472a3", alpha=0.45, linewidth=1)
        primary = selection[(selection["geometry"] == geometry) & selection["is_primary"]]
        selected_points = primary.merge(
            subset[["seed", "k", "inertia"]],
            left_on=["seed", "selected_k"],
            right_on=["seed", "k"],
            how="left",
            validate="one_to_one",
        )
        if selected_points["inertia"].isna().any():
            raise ValueError(f"primary selections do not join the {geometry} inertia curves")
        axis.scatter(selected_points["selected_k"], selected_points["inertia"], color="#b33a3a", s=20)
        axis.set_title(geometry)
        axis.set_xlabel("K")
        axis.set_ylabel("Inertia")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_selection_frequency(selection: pd.DataFrame, path: Path) -> None:
    frequencies = (
        selection.groupby(["geometry", "endpoint", "selected_k"], sort=True)
        .size()
        .rename("count")
        .reset_index()
    )
    geometries = list(dict.fromkeys(frequencies["geometry"].astype(str)))
    fig, axes = plt.subplots(
        1,
        len(geometries),
        figsize=(4 * len(geometries), 3.6),
        squeeze=False,
        sharey=True,
    )
    colours = {30: "#4472a3", 40: "#b33a3a", 50: "#4d7f52"}
    for axis, geometry in zip(axes[0], geometries, strict=True):
        subset = frequencies[frequencies["geometry"] == geometry]
        endpoints = sorted(subset["endpoint"].astype(int).unique())
        width = 0.8 / max(len(endpoints), 1)
        for index, endpoint in enumerate(endpoints):
            rows = subset[subset["endpoint"] == endpoint]
            offset = (index - (len(endpoints) - 1) / 2) * width
            axis.bar(
                rows["selected_k"] + offset,
                rows["count"],
                width=width,
                label=f"endpoint {endpoint}",
                color=colours.get(endpoint, "#777777"),
            )
        axis.set_title(geometry)
        axis.set_xlabel("Selected K")
        axis.set_ylabel("Seeds")
        axis.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_stability(agreement: pd.DataFrame, path: Path) -> None:
    comparisons = list(dict.fromkeys(agreement["comparison"].astype(str)))
    values = [
        agreement.loc[agreement["comparison"] == comparison, "adjusted_rand"].to_numpy()
        for comparison in comparisons
    ]
    fig, axis = plt.subplots(figsize=(max(7, 1.8 * len(comparisons)), 4.2))
    axis.boxplot(values, tick_labels=comparisons, showfliers=True)
    axis.axhline(0.0, color="#777777", linewidth=0.8)
    axis.set_ylabel("Adjusted Rand index")
    axis.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _validate_reopened_outputs(
    tables: dict[str, pd.DataFrame],
    payloads: dict[str, dict[str, np.ndarray]],
    paths: OutputPaths,
) -> None:
    required = {
        "sweep": {"geometry", "seed", "k", "inertia", "dataset", "subset_hash", "git_sha"},
        "selection": {
            "selection_id", "geometry", "seed", "endpoint", "is_primary", "selected_k",
            "maximum_chord_deviation", "runner_up_k", "runner_up_chord_deviation",
            "deviation_margin", "knee_confidence", "git_sha",
        },
        "runs": {
            "row", "dataset", "records", "n", "subset_hash", "space", "n_dims",
            "cluster_seed", "role", "selection_id", "requested_k", "realised_k",
            "inertia", "n_iter", "silhouette", "calinski_harabasz", "davies_bouldin",
            "silhouette_n_scored", "calinski_harabasz_n_scored", "davies_bouldin_n_scored",
            "control_permutations", "archive", "git_sha", "numpy_version", "pandas_version",
            "sklearn_version", "matplotlib_version", "kmeans_n_clusters",
            "kmeans_random_state", "kmeans_init", "kmeans_n_init", "kmeans_max_iter",
            "kmeans_tol", "kmeans_algorithm", "kmeans_copy_x", "kmeans_verbose",
        },
        "clusters": {
            "row", "cluster", "size", "fraction", "squared_error_sum", "mean_radius",
            "maximum_radius", "centroid_norm", "davies_bouldin_worst_pair_ratio",
        },
        "agreement": {"comparison", "left_row", "right_row", "n_common", "adjusted_rand"},
        "crosscheck": {
            "k", "standalone_inertia", "protspace_inertia", "standalone_chord_deviation",
            "protspace_chord_deviation", "standalone_selected_k", "protspace_selected_k",
            "standalone_confidence", "protspace_confidence", "git_sha",
        },
    }
    for metric in ("silhouette", "calinski_harabasz", "davies_bouldin"):
        for statistic in ("minimum", "median", "maximum", "range"):
            required["runs"].add(f"control_{metric}_{statistic}")
    for name, columns in required.items():
        missing = columns - set(tables[name].columns)
        if missing:
            raise ValueError(f"reopened {name} table is missing columns: {sorted(missing)}")
        if tables[name].empty:
            raise ValueError(f"reopened {name} table is empty")
    if tables["sweep"].duplicated(["geometry", "seed", "k"]).any():
        raise ValueError("reopened sweep rows are not unique")
    if tables["selection"]["selection_id"].duplicated().any():
        raise ValueError("reopened selection IDs are not unique")
    expected_selections = len(
        tables["sweep"][["geometry", "seed"]].drop_duplicates()
    ) * tables["selection"]["endpoint"].nunique()
    if len(tables["selection"]) != expected_selections:
        raise ValueError("reopened selection count does not match sweep axes and endpoints")
    if tables["runs"]["row"].duplicated().any() or set(tables["runs"]["row"]) != set(payloads):
        raise ValueError("reopened run IDs do not match partition payloads")
    expected_archive_names = tables["runs"]["row"].astype(str) + ".npz"
    if not tables["runs"]["archive"].astype(str).equals(expected_archive_names):
        raise ValueError("reopened archive names do not equal their run IDs plus .npz")
    run_ids = set(tables["runs"]["row"])
    if not set(tables["clusters"]["row"]).issubset(run_ids):
        raise ValueError("reopened cluster rows reference missing runs")
    cluster_counts = tables["clusters"].groupby("row")["size"].sum().astype(int)
    run_counts = tables["runs"].set_index("row")["n"].astype(int)
    if set(cluster_counts.index) != run_ids or not cluster_counts.equals(run_counts.loc[cluster_counts.index]):
        raise ValueError("reopened cluster sizes do not reproduce every run's record count")
    agreement_ids = set(tables["agreement"]["left_row"]) | set(tables["agreement"]["right_row"])
    if not agreement_ids.issubset(run_ids):
        raise ValueError("reopened agreement rows reference missing runs")
    archive_paths = list(paths.assignments.glob("*.npz"))
    if len(archive_paths) != len(payloads):
        raise ValueError("archive count does not match the run table")
    n_by_row = tables["runs"].set_index("row")["n"].astype(int).to_dict()
    for row, expected in payloads.items():
        archive = paths.assignments / f"{row}.npz"
        with np.load(archive, allow_pickle=False) as loaded:
            if set(loaded.files) != {"protein_id", "label"}:
                raise ValueError(f"archive {archive.name} has unexpected arrays")
            ids = loaded["protein_id"]
            labels = loaded["label"]
        if labels.dtype != np.int32 or ids.dtype.kind not in {"U", "S"}:
            raise ValueError(f"archive {archive.name} has incorrect array types")
        if len(ids) != n_by_row[row] or not np.array_equal(ids, expected["protein_id"]):
            raise ValueError(f"archive {archive.name} does not preserve dataset ID order")
        if not np.array_equal(labels, expected["label"]):
            raise ValueError(f"archive {archive.name} labels do not match the in-memory partition")


def _validate_scientific_outputs(
    tables: dict[str, pd.DataFrame],
    paths: OutputPaths,
) -> dict[str, int]:
    sweep = tables["sweep"]
    selection = tables["selection"]
    runs = tables["runs"]
    agreement = tables["agreement"]
    crosscheck = tables["crosscheck"]

    expected_sweep = len(GEOMETRIES) * len(SEEDS) * len(K_VALUES)
    if len(sweep) != expected_sweep or sweep.duplicated(["geometry", "seed", "k"]).any():
        raise ValueError(f"scientific sweep must contain {expected_sweep} unique rows")
    if (
        set(sweep["geometry"]) != set(GEOMETRIES)
        or set(sweep["seed"].astype(int)) != set(SEEDS)
        or set(sweep["k"].astype(int)) != set(K_VALUES)
    ):
        raise ValueError("scientific sweep axes do not match the fixed geometries, seeds and K values")

    expected_selection = len(GEOMETRIES) * len(SEEDS) * len(ENDPOINTS)
    if len(selection) != expected_selection or selection["selection_id"].duplicated().any():
        raise ValueError(f"scientific selection table must contain {expected_selection} unique rows")
    primary = selection[selection["endpoint"] == PRIMARY_ENDPOINT]
    if len(primary) != len(GEOMETRIES) * len(SEEDS) or not primary["is_primary"].all():
        raise ValueError("scientific primary selections must be exactly the endpoint-40 rows")
    fixed_k = _modal_k(primary)
    varying = primary[primary.apply(
        lambda row: int(row["selected_k"]) != fixed_k[str(row["geometry"])],
        axis=1,
    )]
    q = len(varying)
    if not 0 <= q <= 45:
        raise ValueError(f"derived q is outside its contract: {q}")

    procedure = runs[runs["role"] == "procedure_selected"]
    fixed = runs[runs["role"] == "fixed_k_stability"]
    if len(procedure) != 100 or len(fixed) != 2 * q:
        raise ValueError(f"scientific partitions require 100 procedure and {2 * q} fixed-K rows")
    if runs.duplicated(["geometry", "records", "seed", "requested_k"]).any():
        raise ValueError("scientific run table contains a duplicate partition configuration")
    archives = list(paths.assignments.glob("*.npz"))
    if len(archives) != 100 + 2 * q:
        raise ValueError(f"scientific output requires {100 + 2 * q} partition archives")

    comparison_counts = agreement["comparison"].value_counts().to_dict()
    expected_comparisons = {
        "procedure_selected_stability": 450,
        "full_vs_deduplicated": 50,
        "raw_vs_l2": 20,
        "fixed_k_stability": 90 * primary.groupby("geometry")["selected_k"].nunique().gt(1).sum(),
    }
    for comparison, expected in expected_comparisons.items():
        if comparison_counts.get(comparison, 0) != expected:
            raise ValueError(f"{comparison} requires {expected} rows")
    if set(comparison_counts) != {
        comparison for comparison, count in expected_comparisons.items() if count > 0
    }:
        raise ValueError("scientific agreement table contains an unexpected comparison type")
    varying_geometries = set(primary.groupby("geometry")["selected_k"].nunique().loc[lambda x: x > 1].index)
    fixed_geometries = set(
        agreement.loc[agreement["comparison"] == "fixed_k_stability", "geometry"]
    )
    if fixed_geometries != varying_geometries:
        raise ValueError("fixed-K agreement rows do not match the varying-K geometries")
    run_ids = set(runs["row"])
    if not (
        set(agreement["left_row"]).issubset(run_ids)
        and set(agreement["right_row"]).issubset(run_ids)
    ):
        raise ValueError("scientific agreement table references a missing partition")
    expected_common = agreement["records"].map(
        {"full": EXPECTED_FULL_N, "deduplicated": EXPECTED_DEDUP_N}
    )
    expected_common = expected_common.mask(
        agreement["comparison"] == "full_vs_deduplicated",
        EXPECTED_DEDUP_N,
    )
    if expected_common.isna().any() or not np.array_equal(
        agreement["n_common"].astype(int), expected_common.astype(int)
    ):
        raise ValueError("scientific agreement rows have incorrect common-record counts")

    if not (runs["control_permutations"].astype(int) == CONTROL_PERMUTATIONS).all():
        raise ValueError("every scientific run must record five control permutations")
    for metric in ("silhouette", "calinski_harabasz", "davies_bouldin"):
        minimum = runs[f"control_{metric}_minimum"]
        median = runs[f"control_{metric}_median"]
        maximum = runs[f"control_{metric}_maximum"]
        recorded_range = runs[f"control_{metric}_range"]
        if not ((minimum <= median) & (median <= maximum)).all():
            raise ValueError(f"{metric} control summaries are not ordered")
        if not np.allclose(recorded_range, maximum - minimum, rtol=1e-12, atol=1e-12):
            raise ValueError(f"{metric} control ranges do not equal maximum minus minimum")

    matching_sweep = sweep[
        (sweep["geometry"] == "raw1024")
        & (sweep["seed"] == 0)
        & (sweep["k"] <= PRIMARY_ENDPOINT)
    ].sort_values("k", kind="stable")
    ordered_crosscheck = crosscheck.sort_values("k", kind="stable")
    if len(ordered_crosscheck) != 39 or tuple(ordered_crosscheck["k"].astype(int)) != tuple(range(2, 41)):
        raise ValueError("ProtSpace cross-check must contain K=2 through 40 exactly once")
    if not np.allclose(
        ordered_crosscheck["standalone_inertia"],
        matching_sweep["inertia"],
        rtol=1e-12,
        atol=1e-9,
    ) or not np.allclose(
        ordered_crosscheck["protspace_inertia"],
        ordered_crosscheck["standalone_inertia"],
        rtol=1e-12,
        atol=1e-9,
    ):
        raise ValueError("ProtSpace and standalone inertia curves do not match")
    if not np.allclose(
        ordered_crosscheck["standalone_chord_deviation"],
        matching_sweep["chord_deviation_kmax_40"],
        rtol=1e-12,
        atol=1e-9,
    ) or not np.allclose(
        ordered_crosscheck["protspace_chord_deviation"],
        ordered_crosscheck["standalone_chord_deviation"],
        rtol=1e-12,
        atol=1e-9,
    ):
        raise ValueError("ProtSpace and standalone chord-deviation curves do not match")
    primary_raw_seed_zero = primary[
        (primary["geometry"] == "raw1024") & (primary["seed"] == 0)
    ].iloc[0]
    for column, expected in (
        ("standalone_selected_k", int(primary_raw_seed_zero["selected_k"])),
        ("protspace_selected_k", int(primary_raw_seed_zero["selected_k"])),
        ("standalone_confidence", str(primary_raw_seed_zero["knee_confidence"])),
        ("protspace_confidence", str(primary_raw_seed_zero["knee_confidence"])),
    ):
        if ordered_crosscheck[column].nunique() != 1 or ordered_crosscheck[column].iloc[0] != expected:
            raise ValueError(f"ProtSpace cross-check {column} does not match the primary selection")

    for table_path in (
        paths.sweep,
        paths.selection,
        paths.runs,
        paths.clusters,
        paths.agreement,
        paths.crosscheck,
    ):
        if not table_path.exists():
            raise ValueError(f"scientific metric file is missing: {table_path.name}")
    for figure in (paths.elbow_figure, paths.frequency_figure, paths.stability_figure):
        pixels = plt.imread(figure)
        if not np.isfinite(pixels).all() or float(np.ptp(pixels)) == 0.0:
            raise ValueError(f"scientific figure is blank or non-finite: {figure.name}")

    core_fits = len(sweep) + len(runs)
    return {
        "q": q,
        "archives": len(archives),
        "core_fits": core_fits,
        "core_initialisations": core_fits * int(KMEANS_ARGUMENTS["n_init"]),
        "protspace_fits": len(crosscheck),
        "complete_fits": core_fits + len(crosscheck),
        "complete_initialisations": (core_fits + len(crosscheck))
        * int(KMEANS_ARGUMENTS["n_init"]),
    }


def _persist_outputs(
    tables: dict[str, pd.DataFrame],
    payloads: dict[str, dict[str, np.ndarray]],
    paths: OutputPaths,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    _reset_outputs(paths)
    table_paths = {
        "sweep": paths.sweep,
        "selection": paths.selection,
        "runs": paths.runs,
        "clusters": paths.clusters,
        "agreement": paths.agreement,
        "crosscheck": paths.crosscheck,
    }
    for name, path in table_paths.items():
        tables[name].to_csv(path, index=False)
    for row, payload in payloads.items():
        np.savez_compressed(
            paths.assignments / f"{row}.npz",
            protein_id=np.asarray(payload["protein_id"], dtype=str),
            label=np.asarray(payload["label"], dtype=np.int32),
        )
    reopened = {name: pd.read_csv(path) for name, path in table_paths.items()}
    _validate_reopened_outputs(reopened, payloads, paths)
    _plot_elbow(reopened["sweep"], reopened["selection"], paths.elbow_figure)
    _plot_selection_frequency(reopened["selection"], paths.frequency_figure)
    _plot_stability(reopened["agreement"], paths.stability_figure)
    return reopened, _agreement_summary(reopened["agreement"])


def _round_trip_preflight() -> None:
    rng = np.random.default_rng(11)
    X = np.vstack([rng.normal(loc=location, scale=0.2, size=(12, 3)) for location in (-1.0, 1.0)])
    full_ids = [f"p{index:03d}" for index in range(len(X))]
    deduplicated_ids = full_ids[:-2]
    spaces = {
        "full": {"raw1024": X, "l2_raw1024": normalize(X)},
        "deduplicated": {
            "raw1024": X[:-2],
            "l2_raw1024": normalize(X[:-2]),
        },
    }
    datasets = {
        "full": Dataset(DATASET, full_ids, X, pd.DataFrame(index=range(len(X)))),
        "deduplicated": Dataset(
            DATASET,
            deduplicated_ids,
            X[:-2],
            pd.DataFrame(index=range(len(X) - 2)),
        ),
    }
    sweep = _sweep_table(
        spaces["full"],
        full_ids,
        seeds=(0, 1),
        k_values=(2, 3, 4, 5),
        endpoints=ENDPOINTS,
    )
    selection = _selection_table(sweep, endpoints=ENDPOINTS)
    raw_primary = (
        (selection["geometry"] == "raw1024")
        & (selection["seed"] == 1)
        & selection["is_primary"]
    )
    raw_seed_zero_k = int(selection[
        (selection["geometry"] == "raw1024")
        & (selection["seed"] == 0)
        & selection["is_primary"]
    ].iloc[0]["selected_k"])
    selection.loc[raw_primary, "selected_k"] = 3 if raw_seed_zero_k != 3 else 4
    runs, clusters, payloads, q = _partition_tables(
        spaces,
        {records: dataset.ids for records, dataset in datasets.items()},
        selection,
    )
    agreement = _agreement_table(runs, payloads)
    crosscheck = pd.DataFrame([{
        "k": 2,
        "standalone_inertia": float(sweep.iloc[0]["inertia"]),
        "protspace_inertia": float(sweep.iloc[0]["inertia"]),
        "standalone_chord_deviation": float(sweep.iloc[0]["chord_deviation_kmax_40"]),
        "protspace_chord_deviation": float(sweep.iloc[0]["chord_deviation_kmax_40"]),
        "standalone_selected_k": int(selection[selection["is_primary"]].iloc[0]["selected_k"]),
        "protspace_selected_k": int(selection[selection["is_primary"]].iloc[0]["selected_k"]),
        "standalone_confidence": str(selection[selection["is_primary"]].iloc[0]["knee_confidence"]),
        "protspace_confidence": str(selection[selection["is_primary"]].iloc[0]["knee_confidence"]),
    }])
    tables = _tables_with_provenance(
        sweep,
        selection,
        runs,
        clusters,
        agreement,
        crosscheck,
        datasets,
        {113: 0.95, 67: 0.90, 34: 0.80},
    )
    with TemporaryDirectory() as directory:
        paths = _output_paths(Path(directory) / "kmeans")
        paths.root.mkdir(parents=True)
        writeup = paths.root / "kmeans.md"
        writeup.write_text("manual evidence\n", encoding="utf-8")
        paths.runs.parent.mkdir(parents=True)
        paths.runs.write_text("stale\n", encoding="utf-8")
        paths.assignments.mkdir(parents=True)
        (paths.assignments / "stale.npz").write_bytes(b"stale")
        paths.elbow_figure.parent.mkdir(parents=True)
        stale_figure = paths.elbow_figure.parent / "stale.png"
        stale_figure.write_bytes(b"stale")
        reopened, summary = _persist_outputs(tables, payloads, paths)
        if writeup.read_text(encoding="utf-8") != "manual evidence\n":
            raise ValueError("Reset changed the manual write-up")
        if (
            (paths.assignments / "stale.npz").exists()
            or stale_figure.exists()
            or len(reopened["runs"]) != len(runs)
        ):
            raise ValueError("Reduced round trip retained stale output or lost run rows")
        if q != 1 or set(reopened["runs"]["role"]) != {
            "procedure_selected",
            "fixed_k_stability",
        }:
            raise ValueError("Reduced round trip did not exercise fixed-K persistence")
        figures = (paths.elbow_figure, paths.frequency_figure, paths.stability_figure)
        if not all(path.exists() and path.stat().st_size > 0 for path in figures):
            raise ValueError("Reduced round trip did not produce all three figures")
        for figure in figures:
            pixels = plt.imread(figure)
            if not np.isfinite(pixels).all() or float(np.ptp(pixels)) == 0.0:
                raise ValueError(f"Figure is blank or non-finite: {figure.name}")
    print(
        f"Temporary round trip passed: runs={len(runs)}, q={q}, archives={len(payloads)}, "
        f"summary_rows={len(summary)}"
    )


def _comparison_preflight() -> None:
    full_ids = np.asarray([f"p{index}" for index in range(8)])
    deduplicated_ids = full_ids[:6]
    runs = []
    payloads = {}
    for geometry_index, geometry in enumerate(("raw1024", "l2_raw1024")):
        for records, ids in (("full", full_ids), ("deduplicated", deduplicated_ids)):
            for seed, selected_k in ((0, 2), (1, 3)):
                for role, k in (("procedure_selected", selected_k),):
                    row_id = _partition_id(geometry, records, seed, k, role)
                    runs.append({
                        "row": row_id,
                        "geometry": geometry,
                        "records": records,
                        "seed": seed,
                        "requested_k": k,
                        "role": role,
                    })
                    order = np.arange(len(ids))[::-1] if seed else np.roll(np.arange(len(ids)), 1)
                    payloads[row_id] = {
                        "protein_id": ids[order],
                        "label": ((np.arange(len(ids))[order] + geometry_index + seed) % k).astype(np.int32),
                    }
            row_id = _partition_id(geometry, records, 1, 2, "fixed_k_stability")
            runs.append({
                "row": row_id,
                "geometry": geometry,
                "records": records,
                "seed": 1,
                "requested_k": 2,
                "role": "fixed_k_stability",
            })
            order = np.roll(np.arange(len(ids)), 2)
            payloads[row_id] = {
                "protein_id": ids[order],
                "label": ((np.arange(len(ids))[order] + geometry_index + 1) % 2).astype(np.int32),
            }
    agreement = _agreement_table(pd.DataFrame(runs), payloads)
    counts = agreement["comparison"].value_counts().to_dict()
    expected = {
        "procedure_selected_stability": 4,
        "fixed_k_stability": 4,
        "full_vs_deduplicated": 4,
        "raw_vs_l2": 4,
    }
    if counts != expected or len(_agreement_summary(agreement)) != 12:
        raise ValueError("reduced comparison preflight produced unexpected comparison or summary counts")
    if set(agreement["n_common"]) != {6, 8}:
        raise ValueError("reduced comparisons did not align the expected shared IDs")
    print(f"Comparison checks passed: comparisons={len(agreement)}, summary_rows=12")


def _partition_preflight() -> None:
    rng = np.random.default_rng(7)
    X = np.vstack([
        rng.normal(loc=location, scale=0.2, size=(20, 3))
        for location in (-2.0, 0.0, 2.0, 4.0)
    ])
    ids = [f"protein_{index:03d}" for index in range(len(X))]
    spaces = {
        "full": {"raw1024": X},
        "deduplicated": {"raw1024": X[:-4]},
    }
    ids_by_records = {"full": ids, "deduplicated": ids[:-4]}
    selection = pd.DataFrame([
        {
            "selection_id": "KS-raw1024-S00-E40",
            "geometry": "raw1024",
            "seed": 0,
            "endpoint": 40,
            "is_primary": True,
            "selected_k": 4,
        },
        {
            "selection_id": "KS-raw1024-S01-E40",
            "geometry": "raw1024",
            "seed": 1,
            "endpoint": 40,
            "is_primary": True,
            "selected_k": 3,
        },
    ])
    runs, clusters, payloads, q = _partition_tables(spaces, ids_by_records, selection)
    if q != 1 or len(runs) != 6 or len(payloads) != 6:
        raise ValueError("reduced partition preflight produced unexpected partition counts")
    if clusters.groupby(["geometry", "records", "seed", "role"]).ngroups != 6:
        raise ValueError("reduced cluster rows do not cover every partition")
    print(f"Partition checks passed: runs={len(runs)}, q={q}, clusters={len(clusters)}")


def _timing_preflight(
    full: Dataset,
    deduplicated: Dataset,
    spaces_by_records: dict[str, dict[str, np.ndarray]],
    fit_estimate: dict[str, float],
) -> None:
    datasets = {"full": full, "deduplicated": deduplicated}
    fit_seconds = {}
    scoring_seconds = {}
    for records, dataset in datasets.items():
        for geometry, X in spaces_by_records[records].items():
            start = perf_counter()
            model, labels, _warning_text = _fit_partition(X, dataset.ids, 40, 0)
            fit_seconds[(records, geometry)] = perf_counter() - start

            start = perf_counter()
            _validate_partition(model, labels, len(X))
            _internal_scores(X, labels)
            _control_summaries(X, labels, 0)
            _cluster_table(
                X,
                labels,
                model,
                {
                    "geometry": geometry,
                    "records": records,
                    "seed": 0,
                    "requested_k": 40,
                    "role": "timing_preflight",
                },
            )
            scoring_seconds[(records, geometry)] = perf_counter() - start

    matched_scoring_seconds = sum(scoring_seconds.values())
    procedure_scoring = len(SEEDS) * matched_scoring_seconds
    maximum_fixed_scoring = (len(SEEDS) - 1) * matched_scoring_seconds
    minimum_total = (
        fit_estimate["sweep_seconds"]
        + fit_estimate["procedure_seconds"]
        + procedure_scoring
    )
    maximum_total = (
        minimum_total
        + fit_estimate["maximum_fixed_k_seconds"]
        + maximum_fixed_scoring
    )
    print("K=40 seed-0 timing probe:")
    for records in ("full", "deduplicated"):
        for geometry in GEOMETRIES:
            key = (records, geometry)
            print(
                f"  {records}/{geometry}: fit={fit_seconds[key]:.6f}s, "
                f"scoring={scoring_seconds[key]:.6f}s"
            )
    print(
        f"combined prelaunch estimate: {minimum_total / 60:.3f}min at q=0, "
        f"at most {maximum_total / 60:.3f}min"
    )


def _protspace_crosscheck(
    X: np.ndarray,
    ids: list[str],
    sweep: pd.DataFrame,
    selection: pd.DataFrame,
) -> pd.DataFrame:
    protspace_path = str(PROTSPACE_SRC)
    if protspace_path not in sys.path:
        sys.path.insert(0, protspace_path)
    from protspace.stats.cluster.kmeans_elbow import (
        chord_deviation as protspace_chord_deviation,
        kmeans_elbow,
    )

    result = kmeans_elbow(
        X,
        ids=ids,
        rng_seed=0,
        k_max=PRIMARY_ENDPOINT,
        n_init=int(KMEANS_ARGUMENTS["n_init"]),
        knee_min_deviation=KNEE_MIN_DEVIATION,
        silhouette_selection=False,
    )
    if result is None or tuple(result.k_range) != tuple(range(2, PRIMARY_ENDPOINT + 1)):
        raise ValueError("ProtSpace did not return the required K=2 through 40 sweep")
    reference_deviation = protspace_chord_deviation(np.asarray(result.inertia, dtype=float))
    standalone_curve = sweep[
        (sweep["geometry"] == "raw1024")
        & (sweep["seed"] == 0)
        & (sweep["k"] <= PRIMARY_ENDPOINT)
    ].sort_values("k", kind="stable")
    standalone_selection = selection[
        (selection["geometry"] == "raw1024")
        & (selection["seed"] == 0)
        & (selection["endpoint"] == PRIMARY_ENDPOINT)
    ]
    if len(standalone_curve) != 39 or len(standalone_selection) != 1:
        raise ValueError("standalone raw seed-0 primary evidence is incomplete")
    selected = standalone_selection.iloc[0]
    return pd.DataFrame({
        "k": result.k_range,
        "standalone_inertia": standalone_curve["inertia"].to_numpy(dtype=float),
        "protspace_inertia": np.asarray(result.inertia, dtype=float),
        "standalone_chord_deviation": standalone_curve[
            "chord_deviation_kmax_40"
        ].to_numpy(dtype=float),
        "protspace_chord_deviation": reference_deviation,
        "standalone_selected_k": int(selected["selected_k"]),
        "protspace_selected_k": int(result.k),
        "standalone_confidence": str(selected["knee_confidence"]),
        "protspace_confidence": str(result.knee_confidence),
    })


def _run_scientific() -> tuple[dict[str, pd.DataFrame], pd.DataFrame, dict[str, int]]:
    _require_sklearn_version()
    paths = _output_paths()
    _reset_outputs(paths)
    print("Cleared owned KMeans outputs; loading phosphatase coordinates.", flush=True)
    full = load_dataset(DATASET)
    deduplicated = deduplicate(full)
    spaces_by_records, retained = _spaces(full, deduplicated)
    datasets = {"full": full, "deduplicated": deduplicated}

    print("Starting 2,450-fit inertia sweep.", flush=True)
    sweep = _sweep_table(
        spaces_by_records["full"],
        full.ids,
        progress=True,
    )
    selection = _selection_table(sweep)
    print("Fitting and scoring selected and required fixed-K partitions.", flush=True)
    runs, clusters, payloads, q = _partition_tables(
        spaces_by_records,
        {records: dataset.ids for records, dataset in datasets.items()},
        selection,
    )
    print(f"Built {len(runs)} partitions with q={q}; calculating ARI comparisons.", flush=True)
    agreement = _agreement_table(runs, payloads)

    print("Running the independent 39-fit ProtSpace cross-check.", flush=True)
    crosscheck = _protspace_crosscheck(
        spaces_by_records["full"]["raw1024"],
        full.ids,
        sweep,
        selection,
    )
    tables = _tables_with_provenance(
        sweep,
        selection,
        runs,
        clusters,
        agreement,
        crosscheck,
        datasets,
        retained,
    )
    reopened, summary = _persist_outputs(tables, payloads, paths)
    counts = _validate_scientific_outputs(reopened, paths)
    print(f"Scientific output validation passed: {counts}", flush=True)
    return reopened, summary, counts


def _coordinate_preflight() -> tuple[Dataset, Dataset, dict[str, dict[str, np.ndarray]], dict[str, float]]:
    _require_sklearn_version()
    full = load_dataset(DATASET)
    deduplicated = deduplicate(full)
    spaces_by_records, retained = _spaces(full, deduplicated)

    timings = {}
    for geometry, X in spaces_by_records["full"].items():
        start = perf_counter()
        _fit_kmeans(X, full.ids, k=40, seed=0)
        timings[geometry] = perf_counter() - start

    reduced_k = (2, 3, 4, 5)
    reduced_endpoints = (4, 5)
    reduced = _sweep_table(
        {"raw1024": spaces_by_records["full"]["raw1024"][:120]},
        full.ids[:120],
        seeds=(0, 1),
        k_values=reduced_k,
        endpoints=reduced_endpoints,
    )
    selected = _selection_table(reduced, endpoints=reduced_endpoints)
    expected_rows = 2 * len(reduced_k)
    if len(reduced) != expected_rows or len(selected) != 2 * len(reduced_endpoints):
        raise ValueError("reduced sweep produced unexpected table counts")
    estimate = _fit_phase_estimate(timings)

    print("Coordinate checks passed")
    print("variance retained: " + ", ".join(
        f"PCA {dimensions}={retained[dimensions]:.12f}" for dimensions in PCA_DIMS
    ))
    print("K=40 seed-0 timings: " + ", ".join(
        f"{geometry}={seconds:.6f}s" for geometry, seconds in timings.items()
    ))
    print("fit-phase estimate: " + ", ".join(
        f"{phase}={seconds / 60:.3f}min" for phase, seconds in estimate.items()
    ))
    print(f"reduced sweep rows={len(reduced)}, selection rows={len(selected)}")
    return full, deduplicated, spaces_by_records, estimate


def _run_preflights() -> None:
    full_dataset, deduplicated_dataset, coordinate_spaces, sweep_fit_estimate = _coordinate_preflight()
    _partition_preflight()
    _comparison_preflight()
    _timing_preflight(
        full_dataset,
        deduplicated_dataset,
        coordinate_spaces,
        sweep_fit_estimate,
    )
    _round_trip_preflight()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-scientific",
        action="store_true",
        help="run the full phosphatase experiment and replace owned KMeans outputs",
    )
    arguments = parser.parse_args(argv)
    if arguments.run_scientific:
        _run_scientific()
    else:
        _run_preflights()


if __name__ == "__main__":
    main()
