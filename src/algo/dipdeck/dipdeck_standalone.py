from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import inspect
import io
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import threading
import warnings
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from dataclasses import dataclass
from itertools import combinations, product
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace
from typing import Any, Callable, Iterator

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import psutil
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.linalg import subspace_angles
from scipy.spatial.distance import pdist

from data import (
    DATASETS, EC_LEVELS, REPO, Dataset, deduplicate, ec_table, external_scores, load_dataset,
    load_embeddings, read_bundle_part,
)


DATASET = "phosphatase"
RUN_ROLE = "engineering_smoke"
RECORDS = "full"
INPUT_SPACE = "raw1024"
EXPECTED_FULL_N = 1_587
EXPECTED_DIMS = 1_024
EXPECTED_SUBSET_HASH = "1906375a02bc"
SCIENTIFIC_FOLD_ASSIGNMENT = (
    REPO
    / "results"
    / "graphpart"
    / "explore_20260928T005618Z"
    / "unlabelled_k5"
    / "folds_t0p40_all_records.csv"
)
SCIENTIFIC_RETAINED_N = 1_577
SCIENTIFIC_REMOVED_N = 10
SCIENTIFIC_DEDUP_RETAINED_N = 1_470
SCIENTIFIC_FOLDS = ("0", "1", "2", "3", "4")
SCIENTIFIC_INITIAL_CLUSTERS = (15, 25, 35, 50)
SCIENTIFIC_MERGE_THRESHOLDS = (0.60, 0.70, 0.80, 0.90, 0.95)
SCIENTIFIC_SEEDS = tuple(range(10))
SCIENTIFIC_REFERENCE_INITIAL_CLUSTERS = 35
SCIENTIFIC_REFERENCE_MERGE_THRESHOLD = 0.90
SCIENTIFIC_PCA_DIMS = (113, 67, 34)


@dataclass(frozen=True)
class EmbeddingSpec:
    key: str
    h5: Path | None
    dims: int
    raw_space: str
    pca_dims: tuple[int, ...] | None
    pca_prefix: str
    pipeline: str


EMBEDDINGS = {
    "prott5": EmbeddingSpec(
        key="prott5", h5=None, dims=1_024, raw_space="raw1024", pca_dims=None,
        pca_prefix="", pipeline="raw_mean_pooled_unnormalised_prott5_float32",
    ),
    "esm2_3b": EmbeddingSpec(
        key="esm2_3b",
        h5=REPO / "data/phosphatase/phosphatase/esm2_t36_3B_UR50D_full_length.h5",
        dims=2_560, raw_space="esm2_3b_raw2560", pca_dims=(45,), pca_prefix="esm2_3b_",
        pipeline="raw_mean_pooled_unnormalised_esm2_t36_3b_float32",
    ),
}
ALL_PCA_DIMS = tuple(sorted(
    {*SCIENTIFIC_PCA_DIMS, *(d for spec in EMBEDDINGS.values() for d in spec.pca_dims or ())}
))


def _pca_dims(embedding: str) -> tuple[int, ...]:
    spec = EMBEDDINGS[embedding]
    return SCIENTIFIC_PCA_DIMS if spec.pca_dims is None else spec.pca_dims


def _input_space(embedding: str, pca_dimensions: int | None) -> str:
    spec = EMBEDDINGS[embedding]
    if pca_dimensions is None:
        return spec.raw_space
    if pca_dimensions not in _pca_dims(embedding):
        raise ValueError(f"{embedding} PCA dimensions must be one of {_pca_dims(embedding)}")
    return f"{spec.pca_prefix}pca{pca_dimensions}"


def _load_embedding_dataset(embedding: str) -> Dataset:
    dataset = load_dataset(DATASET)
    spec = EMBEDDINGS[embedding]
    if spec.h5 is None:
        return dataset
    ids, X = load_embeddings(spec.h5)
    if sorted(ids) != sorted(dataset.ids) or len(set(ids)) != len(ids):
        raise ValueError(f"{embedding} protein IDs do not match the phosphatase records")
    position = {protein_id: row for row, protein_id in enumerate(ids)}
    aligned = np.ascontiguousarray(X[[position[i] for i in dataset.ids]], dtype=np.float32)
    if aligned.shape != (len(dataset), spec.dims):
        raise ValueError(f"{embedding} vectors have shape {aligned.shape}, not {(len(dataset), spec.dims)}")
    return Dataset(name=dataset.name, ids=list(dataset.ids), X=aligned, ann=dataset.ann)


def _parse_seeds(text: str) -> tuple[int, ...]:
    seeds: list[int] = []
    for part in text.split(","):
        if "-" in part:
            first, last = (int(value) for value in part.split("-"))
            seeds.extend(range(first, last + 1))
        else:
            seeds.append(int(part))
    if not seeds or len(set(seeds)) != len(seeds) or not set(seeds) <= set(SCIENTIFIC_SEEDS):
        raise ValueError(f"seeds must be distinct values from {SCIENTIFIC_SEEDS}")
    return tuple(sorted(seeds))
PCA_DISTANCE_RTOL = 1e-4
CLUSTPY_VERSION = "0.0.2"
CLUSTPY_COMMIT = "82548dc35a01d221474fbb44908b3621d4b97616"
UPSTREAM_DIPDECK_SOURCE_SHA256 = "c567d24c1722087a75c3eb56f17fe91152b6f818148083d908086b9fc78d438a"
DIPDECK_SOURCE_SHA256 = "cfbf629ab341bdbb5967af232c24cfc3f2e5fb09cd082fb28b317c45738b7420"
LEGACY_CLUSTPY_COMMIT = "6de210e1bc4395176a4299aab96e4156541295b5"
LEGACY_DIPDECK_SOURCE_SHA256 = "3c73c0b8af424ad4023b1304d2ec193a11feeaa745a5f2d876d080761fdd1506"
N_CLUSTERS_INIT = 35
DIP_MERGE_THRESHOLD = 0.9
EMBEDDING_SIZE = 5
BATCH_SIZE = 256
PRETRAIN_EPOCHS = 3
CLUSTERING_EPOCHS = 3
SEED = 42
MERGE_RECORD = re.compile(r"^Merge in iteration")
PRETRAINING_LOSS_RECORD = re.compile(r"Training Loss=([0-9.eE+-]+)")
RSS_SAMPLE_INTERVAL_SECONDS = 0.05
SCORE_RECOMPUTE_RTOL = {
    "silhouette": 100 * float(np.finfo(np.float32).eps),
    "calinski_harabasz": 100 * float(np.finfo(np.float32).eps),
    "davies_bouldin": 1e-12,
}

RUN_COLUMNS = (
    "row", "run_role", "dataset", "records", "n", "subset_hash", "input_space", "n_dims",
    "input_pipeline", "transform_fit_records", "transform_sha256", "variance_retained", "seed",
    "pca_random_state", "dipdeck_random_state", "device", "device_name", "torch_intraop_threads",
    "torch_interop_threads", "numpy_blas_threads", "scipy_blas_threads", "sklearn_openmp_threads",
    "torch_openmp_threads", "omp_num_threads", "mkl_num_threads", "openblas_num_threads",
    "deterministic_algorithms", "n_clusters_init", "dip_merge_threshold", "embedding_size",
    "max_cluster_size_diff_factor", "pval_strategy", "min_n_clusters", "max_n_clusters",
    "augmentation_invariance", "network_layers", "activation", "batch_size", "pretrain_epochs",
    "clustering_epochs", "pretrain_learning_rate", "clustering_learning_rate", "optimizer",
    "ssl_loss", "ssl_loss_reduction", "ssl_loss_weight", "clustering_loss_weight", "status",
    "n_clusters", "cluster_size_minimum", "cluster_size_median", "cluster_size_maximum",
    "warning_count", "merge_count", "effective_clustering_epochs", "effective_epochs_status",
    "clustering_space", "clustering_distance", "scoring_space", "scoring_distance",
    "elapsed_seconds", "peak_cuda_bytes", "peak_cuda_status", "peak_resident_bytes",
    "peak_resident_status", "model_parameter_count", "model_parameter_bytes", "git_sha",
    "clustpy_version", "clustpy_commit", "dipdeck_source_sha256", "python_version",
    "torch_version", "torchvision_version", "numpy_version", "scipy_version", "sklearn_version",
    "pandas_version", "h5py_version", "pyarrow_version", "tqdm_version", "threadpoolctl_version",
)
CLUSTER_COLUMNS = (
    "row", "cluster", "size", "proportion", "representative_protein_id",
    "representative_match_count", "representative_matching_label_count",
)
SCALING_COLUMNS = (
    "input_space", "run_row", "input_pipeline", "transform_fit_records",
    "transform_sha256", "pretraining_loss_first_observed",
    "pretraining_loss_final_observed", "pretraining_loss_status", "finite_outputs",
    "n_clusters", "elapsed_seconds", "partition_ari_to_raw",
)
TIMING_COLUMNS = (
    "run_row", "input_space", "input_selection_basis", "split_policy",
    "pretrain_epochs", "nominal_clustering_epochs", "elapsed_seconds",
    "pretraining_seconds", "clustering_seconds", "phase_timing_status",
    "peak_cuda_bytes", "peak_cuda_status", "peak_resident_bytes", "peak_resident_status",
    "rss_sample_interval_seconds", "effective_clustering_epochs", "effective_epochs_status",
    "psutil_version",
)
SCIENTIFIC_RUN_COLUMNS = (
    "row", "run_role", "held_out_fold", "train_n", "train_subset_hash", "test_n",
    "test_subset_hash", "removed_n", "fold_assignment_sha256", "input_space", "n_dims", "seed",
    "n_clusters_init", "dip_merge_threshold", "embedding_size", "batch_size", "pretrain_epochs",
    "clustering_epochs", "clustering_epoch_updates", "clustering_epoch_status",
    "n_training_clusters", "n_test_clusters", "warning_count", "merge_count",
    "elapsed_seconds", "peak_cuda_bytes", "peak_cuda_status", "peak_resident_bytes",
    "peak_resident_status", "network_layers", "model_parameter_count", "model_parameter_bytes",
    "status", "git_sha", "clustpy_version", "clustpy_commit", "dipdeck_source_sha256",
    "python_version", "score_status", "silhouette", "calinski_harabasz", "davies_bouldin",
    "torch_version", "torchvision_version", "numpy_version", "scipy_version", "sklearn_version",
    "pandas_version", "h5py_version", "pyarrow_version", "tqdm_version", "threadpoolctl_version",
)
SCIENTIFIC_CLUSTER_COLUMNS = ("row", "held_out_fold", "cluster", "size", "proportion")


@dataclass(frozen=True)
class SmokeConfig:
    n_clusters_init: int = N_CLUSTERS_INIT
    dip_merge_threshold: float = DIP_MERGE_THRESHOLD
    embedding_size: int = EMBEDDING_SIZE
    batch_size: int = BATCH_SIZE
    pretrain_epochs: int = PRETRAIN_EPOCHS
    clustering_epochs: int = CLUSTERING_EPOCHS
    seed: int = SEED


SMOKE_CONFIG = SmokeConfig()
FULL_SETTING_CONFIG = SmokeConfig(pretrain_epochs=100, clustering_epochs=50)


def _scientific_configs(
    *,
    initial_clusters: tuple[int, ...] = SCIENTIFIC_INITIAL_CLUSTERS,
    merge_thresholds: tuple[float, ...] = SCIENTIFIC_MERGE_THRESHOLDS,
    seeds: tuple[int, ...] = SCIENTIFIC_SEEDS,
) -> tuple[SmokeConfig, ...]:
    return tuple(
        SmokeConfig(
            n_clusters_init=n_clusters_init,
            dip_merge_threshold=dip_merge_threshold,
            pretrain_epochs=FULL_SETTING_CONFIG.pretrain_epochs,
            clustering_epochs=FULL_SETTING_CONFIG.clustering_epochs,
            seed=seed,
        )
        for n_clusters_init, dip_merge_threshold, seed in product(
            initial_clusters, merge_thresholds, seeds
        )
    )


def _scientific_reference_configs(
    seeds: tuple[int, ...] = SCIENTIFIC_SEEDS,
) -> tuple[SmokeConfig, ...]:
    return tuple(
        SmokeConfig(
            n_clusters_init=SCIENTIFIC_REFERENCE_INITIAL_CLUSTERS,
            dip_merge_threshold=SCIENTIFIC_REFERENCE_MERGE_THRESHOLD,
            pretrain_epochs=FULL_SETTING_CONFIG.pretrain_epochs,
            clustering_epochs=FULL_SETTING_CONFIG.clustering_epochs,
            seed=seed,
        )
        for seed in seeds
    )


@dataclass(frozen=True)
class RunIdentity:
    run_role: str = RUN_ROLE
    records: str = RECORDS
    input_space: str = INPUT_SPACE
    input_pipeline: str = "raw_mean_pooled_unnormalised_prott5_float32"
    transform_fit_records: str = ""
    transform_sha256: str = ""
    variance_retained: float | str = ""


SMOKE_IDENTITY = RunIdentity()
SCIENTIFIC_REFERENCE_IDENTITY = RunIdentity(
    run_role="scientific_reference",
    records="graphpart_retained",
)
SCIENTIFIC_REFERENCE_DEDUP_IDENTITY = RunIdentity(
    run_role="scientific_reference",
    records="graphpart_retained_dedup",
)


def _raw_reference_identity(embedding: str, *, dedup: bool) -> RunIdentity:
    spec = EMBEDDINGS[embedding]
    return RunIdentity(
        run_role="scientific_reference",
        records="graphpart_retained_dedup" if dedup else "graphpart_retained",
        input_space=spec.raw_space,
        input_pipeline=spec.pipeline,
    )


@dataclass(frozen=True)
class OutputPaths:
    root: Path
    runs: Path
    clusters: Path
    assignments: Path
    logs: Path


@dataclass(frozen=True)
class ScalingPaths:
    root: Path
    scaling_summary: Path
    full_timing: Path
    standardiser: Path
    raw: OutputPaths
    standardised: OutputPaths
    full_setting: OutputPaths


@dataclass(frozen=True)
class ScientificPaths:
    root: Path
    runs: Path
    clusters: Path
    assignments: Path
    logs: Path
    manifest: Path


@dataclass(frozen=True)
class FoldSplit:
    held_out_fold: str
    train: Dataset
    test: Dataset
    removed_ids: tuple[str, ...]
    assignment_sha256: str


def _fold_assignment_sha256(path: Path) -> str:
    contents = path.read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(contents).hexdigest()


def _load_fold_assignment(
    dataset: Dataset,
    path: Path,
    *,
    expected_retained: int | None = None,
    expected_removed: int | None = None,
    expected_folds: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    assignment = pd.read_csv(path, dtype=str, keep_default_na=False)
    expected_columns = ("protein_id", "representative", "fold", "status")
    if tuple(assignment.columns) != expected_columns:
        raise ValueError("GraphPart assignment columns do not match the all-record schema")
    if assignment["protein_id"].duplicated().any():
        raise ValueError("GraphPart assignment has duplicate protein IDs")
    if set(assignment["protein_id"]) != set(dataset.ids):
        raise ValueError("GraphPart assignment IDs do not match the DipDECK dataset")
    if assignment[["protein_id", "representative"]].eq("").any().any():
        raise ValueError("GraphPart assignment has blank protein or representative IDs")

    statuses = set(assignment["status"])
    if statuses != {"retained", "removed"}:
        raise ValueError("GraphPart assignment statuses must be retained or removed")
    retained = assignment.loc[assignment["status"] == "retained"]
    removed = assignment.loc[assignment["status"] == "removed"]
    if retained["fold"].eq("").any() or not removed["fold"].eq("").all():
        raise ValueError("GraphPart retained and removed records have invalid fold values")
    observed_folds = tuple(sorted(retained["fold"].unique()))
    if expected_folds is not None and observed_folds != expected_folds:
        raise ValueError("GraphPart retained folds do not match the scientific five-fold design")
    if expected_retained is not None and len(retained) != expected_retained:
        raise ValueError("GraphPart retained record count does not match the scientific design")
    if expected_removed is not None and len(removed) != expected_removed:
        raise ValueError("GraphPart removed record count does not match the scientific design")

    assignments_per_representative = assignment.groupby("representative", sort=False)[
        ["fold", "status"]
    ].nunique(dropna=False)
    if (assignments_per_representative > 1).any(axis=None):
        raise ValueError("Records with one GraphPart representative have inconsistent assignments")
    return assignment


def _fold_dataset(dataset: Dataset, protein_ids: list[str]) -> Dataset:
    positions = {protein_id: position for position, protein_id in enumerate(dataset.ids)}
    indices = np.asarray([positions[protein_id] for protein_id in protein_ids], dtype=int)
    return Dataset(
        name=dataset.name,
        ids=protein_ids,
        X=np.ascontiguousarray(dataset.X[indices], dtype=np.float32),
        ann=pd.DataFrame(index=np.arange(len(indices))),
    )


def _build_fold_splits(
    dataset: Dataset,
    assignment: pd.DataFrame,
    assignment_sha256: str,
    *,
    folds: tuple[str, ...],
) -> tuple[FoldSplit, ...]:
    retained = assignment.loc[assignment["status"] == "retained"]
    removed_ids = tuple(sorted(assignment.loc[assignment["status"] == "removed", "protein_id"]))
    splits: list[FoldSplit] = []
    for held_out_fold in folds:
        test_ids = retained.loc[retained["fold"] == held_out_fold, "protein_id"].tolist()
        train_ids = retained.loc[retained["fold"] != held_out_fold, "protein_id"].tolist()
        if not test_ids or not train_ids:
            raise ValueError(f"GraphPart fold {held_out_fold} cannot define train and test records")
        if set(train_ids) & set(test_ids):
            raise ValueError(f"GraphPart fold {held_out_fold} overlaps train and test records")
        splits.append(
            FoldSplit(
                held_out_fold=held_out_fold,
                train=_fold_dataset(dataset, train_ids),
                test=_fold_dataset(dataset, test_ids),
                removed_ids=removed_ids,
                assignment_sha256=assignment_sha256,
            )
        )
    return tuple(splits)


def _deduplicated_assignment(dataset: Dataset, assignment: pd.DataFrame) -> pd.DataFrame:
    survivors = set(deduplicate(dataset).ids)
    kept = assignment.loc[assignment["protein_id"].isin(survivors)]
    if not kept["representative"].is_unique or (
        set(kept["representative"]) != set(assignment["representative"])
    ):
        raise ValueError("deduplicated records do not leave one record per GraphPart representative")
    if (kept["status"] == "retained").sum() != SCIENTIFIC_DEDUP_RETAINED_N:
        raise ValueError("deduplicated GraphPart retained count does not match the recorded design")
    return kept


def _scientific_fold_splits(
    dataset: Dataset, path: Path = SCIENTIFIC_FOLD_ASSIGNMENT, *, dedup: bool = False
) -> tuple[FoldSplit, ...]:
    if len(dataset) != EXPECTED_FULL_N:
        raise ValueError("scientific DipDECK requires the full phosphatase record set")
    assignment = _load_fold_assignment(
        dataset,
        path,
        expected_retained=SCIENTIFIC_RETAINED_N,
        expected_removed=SCIENTIFIC_REMOVED_N,
        expected_folds=SCIENTIFIC_FOLDS,
    )
    if dedup:
        assignment = _deduplicated_assignment(dataset, assignment)
    return _build_fold_splits(
        dataset, assignment, _fold_assignment_sha256(path), folds=SCIENTIFIC_FOLDS
    )


def _pca_fold_split(
    split: FoldSplit, dimensions: int, *, embedding: str = "prott5"
) -> tuple[FoldSplit, dict[str, Any]]:
    input_space = _input_space(embedding, dimensions)
    train_X, train_ids = _canonical_input(split.train.X, split.train.ids)
    test_X, test_ids = _canonical_input(split.test.X, split.test.ids)
    pca = PCA(n_components=dimensions, svd_solver="full", whiten=False)
    transformed_split = FoldSplit(
        held_out_fold=split.held_out_fold,
        train=Dataset(
            name=split.train.name,
            ids=train_ids.tolist(),
            X=np.ascontiguousarray(pca.fit_transform(train_X), dtype=np.float32),
            ann=pd.DataFrame(index=np.arange(len(train_X))),
        ),
        test=Dataset(
            name=split.test.name,
            ids=test_ids.tolist(),
            X=np.ascontiguousarray(pca.transform(test_X), dtype=np.float32),
            ann=pd.DataFrame(index=np.arange(len(test_X))),
        ),
        removed_ids=split.removed_ids,
        assignment_sha256=split.assignment_sha256,
    )
    return transformed_split, {
        "held_out_fold": split.held_out_fold,
        "input_space": input_space,
        "pca_dimensions": dimensions,
        "pca_fit_records": len(train_X),
        "pca_fit_subset_hash": split.train.subset_hash,
        "pca_components_sha256": hashlib.sha256(pca.components_.tobytes()).hexdigest(),
        "variance_retained": float(pca.explained_variance_ratio_.sum()),
    }


def _pca_scientific_splits(
    dataset: Dataset, dimensions: int, *, dedup: bool = False, embedding: str = "prott5"
) -> tuple[tuple[FoldSplit, ...], list[dict[str, Any]]]:
    transformed = [
        _pca_fold_split(split, dimensions, embedding=embedding)
        for split in _scientific_fold_splits(dataset, dedup=dedup)
    ]
    return tuple(split for split, _metadata in transformed), [
        metadata for _split, metadata in transformed
    ]


def _check_same_pca_coordinates(stored: np.ndarray, recomputed: np.ndarray, label: str) -> None:
    if stored.shape != recomputed.shape:
        raise ValueError(f"{label}: stored PCA input has shape {stored.shape}, expected {recomputed.shape}")
    stored_distances = pdist(np.asarray(stored, dtype=np.float64))
    recomputed_distances = pdist(np.asarray(recomputed, dtype=np.float64))
    difference = float(np.max(np.abs(stored_distances - recomputed_distances)))
    if difference > PCA_DISTANCE_RTOL * float(np.median(recomputed_distances)):
        raise ValueError(f"{label}: stored PCA input does not match the recomputed basis")


def _scientific_reference_dataset(
    dataset: Dataset, path: Path = SCIENTIFIC_FOLD_ASSIGNMENT, *, dedup: bool = False
) -> Dataset:
    assignment = _load_fold_assignment(
        dataset,
        path,
        expected_retained=SCIENTIFIC_RETAINED_N,
        expected_removed=SCIENTIFIC_REMOVED_N,
        expected_folds=SCIENTIFIC_FOLDS,
    )
    if dedup:
        assignment = _deduplicated_assignment(dataset, assignment)
    retained_ids = assignment.loc[assignment["status"] == "retained", "protein_id"].tolist()
    return _fold_dataset(dataset, retained_ids)


def _pca_scientific_reference_dataset(
    dataset: Dataset, dimensions: int, *, dedup: bool = False, embedding: str = "prott5"
) -> tuple[Dataset, dict[str, Any]]:
    input_space = _input_space(embedding, dimensions)
    reference = _scientific_reference_dataset(dataset, dedup=dedup)
    X, protein_ids = _canonical_input(reference.X, reference.ids)
    pca = PCA(n_components=dimensions, svd_solver="full", whiten=False)
    transformed = np.ascontiguousarray(pca.fit_transform(X), dtype=np.float32)
    metadata = {
        "input_space": input_space,
        "pca_dimensions": dimensions,
        "pca_fit_records": len(X),
        "pca_fit_subset_hash": reference.subset_hash,
        "pca_components_sha256": hashlib.sha256(pca.components_.tobytes()).hexdigest(),
        "variance_retained": float(pca.explained_variance_ratio_.sum()),
    }
    return Dataset(
        name=reference.name,
        ids=protein_ids.tolist(),
        X=transformed,
        ann=pd.DataFrame(index=np.arange(len(transformed))),
    ), metadata


def _pca_scientific_reference_identity(
    metadata: dict[str, Any], *, dedup: bool, embedding: str = "prott5"
) -> RunIdentity:
    return RunIdentity(
        run_role="scientific_reference",
        records="graphpart_retained_dedup" if dedup else "graphpart_retained",
        input_space=str(metadata["input_space"]),
        input_pipeline=f"graphpart_retained_{EMBEDDINGS[embedding].raw_space}_then_pca_unwhitened_float32",
        transform_fit_records=f"graphpart_retained_{metadata['pca_fit_records']}_records",
        transform_sha256=str(metadata["pca_components_sha256"]),
        variance_retained=float(metadata["variance_retained"]),
    )


def _row_id(dataset: Dataset, config: SmokeConfig, identity: RunIdentity = SMOKE_IDENTITY) -> str:
    threshold = str(config.dip_merge_threshold).replace(".", "p")
    return (
        f"dipdeck_{identity.run_role}_{identity.records}_{identity.input_space}_"
        f"k{config.n_clusters_init:03d}_"
        f"t{threshold}_s{config.seed:03d}"
    )


class _PeakResidentMemory:
    def __init__(self, interval_seconds: float = RSS_SAMPLE_INTERVAL_SECONDS):
        self.interval_seconds = interval_seconds
        self.peak_bytes = 0
        self._process = psutil.Process()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample_until_stopped, daemon=True)

    def _sample(self) -> None:
        self.peak_bytes = max(self.peak_bytes, int(self._process.memory_info().rss))

    def _sample_until_stopped(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            self._sample()

    def __enter__(self) -> _PeakResidentMemory:
        self._sample()
        self._thread.start()
        return self

    def __exit__(self, *_error: object) -> None:
        self._sample()
        self._stop.set()
        self._thread.join()


@contextmanager
def _count_dipdeck_clustering_epochs(estimator: Any) -> Iterator[list[int] | None]:
    module = sys.modules.get(type(estimator).__module__)
    if module is None:
        yield None
        return
    tqdm_module = getattr(module, "tqdm", None)
    original = getattr(tqdm_module, "tqdm", None)
    if not callable(original):
        yield None
        return
    updates = [0]

    class CountedProgress:
        def __init__(self, progress: Any) -> None:
            self._progress = progress

        def update(self, count: int = 1) -> Any:
            updates[0] += count
            return self._progress.update(count)

        def __getattr__(self, name: str) -> Any:
            return getattr(self._progress, name)

    def counted_tqdm(*args: Any, **kwargs: Any) -> Any:
        progress = original(*args, **kwargs)
        if kwargs.get("desc") == "DipDECK training":
            return CountedProgress(progress)
        return progress

    module.tqdm = SimpleNamespace(tqdm=counted_tqdm)
    try:
        yield updates
    finally:
        module.tqdm = tqdm_module


def _output_paths(root: Path | None = None) -> OutputPaths:
    result_root = root if root is not None else REPO / "results" / "dipdeck"
    return OutputPaths(
        root=result_root,
        runs=result_root / "metrics" / "dipdeck_standalone_runs.csv",
        clusters=result_root / "metrics" / "dipdeck_standalone_clusters.csv",
        assignments=result_root / "assignments" / "dipdeck_standalone_grid",
        logs=result_root / "logs" / "dipdeck_standalone_grid",
    )


def _scientific_reference_paths(
    root: Path | None, seed: int, *, dedup: bool = False, input_space: str = "raw1024"
) -> OutputPaths:
    result_root = root if root is not None else REPO / "results" / "dipdeck"
    directory = "scientific_reference" if input_space == "raw1024" else f"scientific_reference_{input_space}"
    if dedup:
        directory += "_dedup"
    return _output_paths(result_root / directory / f"seed_{seed:03d}")


def _scaling_paths(root: Path | None = None) -> ScalingPaths:
    result_root = root if root is not None else REPO / "results" / "dipdeck"
    scaling_root = result_root / "scaling_timing"
    return ScalingPaths(
        root=scaling_root,
        scaling_summary=scaling_root / "metrics" / "dipdeck_scaling.csv",
        full_timing=scaling_root / "metrics" / "dipdeck_full_setting_timing.csv",
        standardiser=scaling_root / "transforms" / "zraw1024_standard_scaler.npz",
        raw=_output_paths(scaling_root / "scaling" / "raw1024"),
        standardised=_output_paths(scaling_root / "scaling" / "zraw1024"),
        full_setting=_output_paths(scaling_root / "full_setting" / "zraw1024"),
    )


def _scientific_paths(
    root: Path | None = None, *, dedup: bool = False, input_space: str = "raw1024"
) -> ScientificPaths:
    result_root = root if root is not None else REPO / "results" / "dipdeck"
    suffix = "_dedup" if dedup else ""
    directory = "scientific_grid" if input_space == "raw1024" else f"scientific_grid_{input_space}"
    scientific_root = result_root / f"{directory}{suffix}"
    return ScientificPaths(
        root=scientific_root,
        runs=scientific_root / "metrics" / "dipdeck_scientific_runs.csv",
        clusters=scientific_root / "metrics" / "dipdeck_scientific_clusters.csv",
        assignments=scientific_root / "assignments",
        logs=scientific_root / "logs",
        manifest=scientific_root / "manifest.json",
    )


def _reset_outputs(paths: OutputPaths) -> None:
    for table in (paths.runs, paths.clusters):
        table.parent.mkdir(parents=True, exist_ok=True)
        table.unlink(missing_ok=True)
    for directory in (paths.assignments, paths.logs):
        if directory.exists():
            shutil.rmtree(directory)
        directory.mkdir(parents=True)


def _reset_scientific_outputs(paths: ScientificPaths) -> None:
    if paths.root.exists():
        shutil.rmtree(paths.root)
    paths.assignments.mkdir(parents=True)
    paths.logs.mkdir(parents=True)
    paths.runs.parent.mkdir(parents=True)


def _git_sha() -> str:
    try:
        revision = subprocess.run(
            ["git", "describe", "--always"], cwd=REPO, capture_output=True, text=True, check=True
        )
        status = subprocess.run(
            [
                "git", "status", "--porcelain", "--", ".",
                ":(exclude)results/dipdeck/metrics/dipdeck_standalone_runs.csv",
                ":(exclude)results/dipdeck/metrics/dipdeck_standalone_clusters.csv",
                ":(exclude)results/dipdeck/assignments/dipdeck_standalone_grid/**",
                ":(exclude)results/dipdeck/logs/dipdeck_standalone_grid/**",
                ":(exclude)results/dipdeck/scaling_timing/**",
            ],
            cwd=REPO, capture_output=True, text=True, check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    git_sha = revision.stdout.strip() or "unknown"
    return f"{git_sha}-dirty" if status.stdout.strip() else git_sha


def _canonical_input(
    X: np.ndarray,
    protein_ids: list[str] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    coordinates = np.asarray(X)
    if coordinates.ndim != 2 or not np.issubdtype(coordinates.dtype, np.number):
        raise ValueError("DipDECK input must be a two-dimensional real numeric array")
    if np.issubdtype(coordinates.dtype, np.complexfloating):
        raise ValueError("DipDECK input must contain real values")
    if coordinates.shape[0] == 0 or coordinates.shape[1] == 0:
        raise ValueError("DipDECK input must contain rows and features")
    canonical_coordinates = np.ascontiguousarray(coordinates, dtype=np.float32)
    if not np.isfinite(canonical_coordinates).all():
        raise ValueError("DipDECK input contains non-finite values after float32 conversion")

    identifiers = np.asarray(protein_ids, dtype=object)
    if identifiers.ndim != 1 or len(identifiers) != len(canonical_coordinates):
        raise ValueError("DipDECK requires one protein ID per coordinate row")
    values = identifiers.tolist()
    if any(not isinstance(value, str) or not value for value in values):
        raise ValueError("DipDECK protein IDs must be non-empty strings")
    if len(values) != len(set(values)):
        raise ValueError("DipDECK protein IDs must be unique")
    order = np.argsort(identifiers.astype(str), kind="stable")
    return canonical_coordinates[order], identifiers.astype(str)[order]


def _standardise_features(
    X: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    scaler = StandardScaler(copy=True)
    transformed = np.ascontiguousarray(scaler.fit_transform(X), dtype=np.float32)
    feature_mean = np.asarray(scaler.mean_, dtype=np.float64)
    feature_scale = np.asarray(scaler.scale_, dtype=np.float64)
    if transformed.shape != X.shape or not np.isfinite(transformed).all():
        raise ValueError("Z-standardised input has invalid shape or non-finite values")
    if (feature_scale <= 0).any() or not np.isfinite(feature_mean).all() or not np.isfinite(feature_scale).all():
        raise ValueError("StandardScaler returned invalid feature parameters")
    return transformed, feature_mean, feature_scale, _standardiser_digest(feature_mean, feature_scale)


def _standardiser_digest(feature_mean: np.ndarray, feature_scale: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in (feature_mean, feature_scale):
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _save_standardiser(
    path: Path, protein_ids: np.ndarray, feature_mean: np.ndarray, feature_scale: np.ndarray,
) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = _standardiser_digest(feature_mean, feature_scale)
    np.savez_compressed(
        path,
        protein_id=np.asarray(protein_ids, dtype=str),
        feature_mean=np.asarray(feature_mean, dtype=np.float64),
        feature_scale=np.asarray(feature_scale, dtype=np.float64),
    )
    reopened_digest, _reopened_ids = _validate_standardiser(path)
    if reopened_digest != digest:
        raise ValueError("reopened standardiser digest changed")
    return digest


def _validate_standardiser(path: Path) -> tuple[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as loaded:
        if set(loaded.files) != {"protein_id", "feature_mean", "feature_scale"}:
            raise ValueError("Standardiser archive does not match its schema")
        protein_ids = loaded["protein_id"]
        feature_mean = loaded["feature_mean"]
        feature_scale = loaded["feature_scale"]
        if (
            protein_ids.dtype.kind != "U"
            or protein_ids.shape != (EXPECTED_FULL_N,)
            or protein_ids.tolist() != sorted(protein_ids.tolist())
            or len(set(protein_ids.tolist())) != EXPECTED_FULL_N
        ):
            raise ValueError("Standardiser IDs do not identify the canonical full population")
        expected_hash = hashlib.sha1("\n".join(protein_ids.tolist()).encode()).hexdigest()[:12]
        if expected_hash != EXPECTED_SUBSET_HASH:
            raise ValueError("Standardiser IDs do not match the full phosphatase subset")
        for name, array in (("mean", feature_mean), ("scale", feature_scale)):
            if array.dtype != np.float64 or array.shape != (EXPECTED_DIMS,) or not np.isfinite(array).all():
                raise ValueError(f"Standardiser {name} has invalid dtype, shape, or values")
        if (feature_scale <= 0).any():
            raise ValueError("Standardiser scales must be positive")
        return _standardiser_digest(feature_mean, feature_scale), protein_ids.copy()


def _validate_configuration(
    X: np.ndarray,
    *,
    n_clusters_init: int,
    dip_merge_threshold: float,
    embedding_size: int,
    batch_size: int,
    pretrain_epochs: int,
    clustering_epochs: int,
    seed: int,
) -> None:
    integers = {
        "n_clusters_init": n_clusters_init,
        "embedding_size": embedding_size,
        "batch_size": batch_size,
        "pretrain_epochs": pretrain_epochs,
        "clustering_epochs": clustering_epochs,
        "seed": seed,
    }
    if any(isinstance(value, bool) or not isinstance(value, (int, np.integer)) for value in integers.values()):
        raise ValueError("DipDECK integer settings must be integers, not booleans")
    unique_vectors = len(np.unique(X, axis=0))
    if not 2 <= n_clusters_init <= min(len(X), unique_vectors):
        raise ValueError("n_clusters_init must be between 2 and the available unique rows")
    if not 1 <= embedding_size <= X.shape[1]:
        raise ValueError("embedding_size must be between 1 and the input width")
    if batch_size <= 0 or pretrain_epochs <= 0 or clustering_epochs <= 0:
        raise ValueError("batch size and epoch counts must be positive")
    if seed < 0:
        raise ValueError("seed must be non-negative")
    if not np.isfinite(dip_merge_threshold) or not 0 <= dip_merge_threshold <= 1:
        raise ValueError("dip_merge_threshold must lie in [0, 1]")


def _validate_predicted_labels(
    labels: np.ndarray, *, n_records: int, n_clusters: int
) -> np.ndarray:
    predicted = np.asarray(labels)
    if (
        predicted.ndim != 1
        or predicted.shape != (n_records,)
        or not np.issubdtype(predicted.dtype, np.integer)
    ):
        raise ValueError("DipDECK held-out labels must be a one-dimensional integer array")
    predicted = predicted.astype(np.int32, copy=False)
    if (predicted < 0).any() or (predicted >= n_clusters).any():
        raise ValueError("DipDECK held-out labels must refer to a learned training cluster")
    return predicted


def _held_out_internal_scores(X: np.ndarray, labels: np.ndarray) -> dict[str, float | str]:
    n_clusters = len(np.unique(labels))
    if n_clusters < 2:
        return {
            "score_status": "undefined_one_predicted_cluster",
            "silhouette": "",
            "calinski_harabasz": "",
            "davies_bouldin": "",
        }
    if n_clusters >= len(X):
        return {
            "score_status": "undefined_one_record_per_predicted_cluster",
            "silhouette": "",
            "calinski_harabasz": "",
            "davies_bouldin": "",
        }
    return {
        "score_status": "defined",
        "silhouette": float(silhouette_score(X, labels, metric="euclidean")),
        "calinski_harabasz": float(calinski_harabasz_score(X, labels)),
        "davies_bouldin": float(davies_bouldin_score(X, labels)),
    }


def _stored_score_matches(field: str, observed: Any, expected: Any) -> bool:
    return bool(
        np.isclose(float(observed), float(expected), rtol=SCORE_RECOMPUTE_RTOL[field], atol=1e-12)
    )


def _validate_live_dataset(dataset: Dataset, X: np.ndarray, protein_ids: np.ndarray) -> None:
    if dataset.name != DATASET or X.shape != (EXPECTED_FULL_N, EXPECTED_DIMS):
        raise ValueError(
            f"Smoke run requires {DATASET} shape {(EXPECTED_FULL_N, EXPECTED_DIMS)}, "
            f"found {dataset.name} {X.shape}"
        )
    if dataset.subset_hash != EXPECTED_SUBSET_HASH or len(protein_ids) != EXPECTED_FULL_N:
        raise ValueError("Smoke run phosphatase identity does not match the recorded full set")


def _discover_clustpy_source(explicit: Path | None) -> Path:
    configured = explicit or (
        Path(os.environ["DIPDECK_CLUSTPY_SOURCE_ROOT"])
        if os.environ.get("DIPDECK_CLUSTPY_SOURCE_ROOT") else None
    )
    for candidate in (configured, REPO.parent / "ClustPy", REPO / "external" / "ClustPy"):
        if candidate is not None and (candidate / ".git").exists():
            return candidate.resolve()
    raise FileNotFoundError(
        "A ClustPy checkout is required; pass --clustpy-source-root or set "
        "DIPDECK_CLUSTPY_SOURCE_ROOT"
    )


def _verify_clustpy_source(source_root: Path) -> None:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=source_root, capture_output=True, text=True, check=True
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=source_root, capture_output=True, text=True, check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError(f"Cannot identify ClustPy source at {source_root}") from error
    source = source_root / "clustpy" / "deep" / "dipdeck.py"
    if commit != CLUSTPY_COMMIT:
        raise RuntimeError(f"ClustPy commit mismatch: expected {CLUSTPY_COMMIT}, found {commit}")
    if status != "M clustpy/deep/dipdeck.py":
        raise RuntimeError(
            "ClustPy checkout must contain only the recorded DipDECK CUDA device patch"
        )
    if not source.is_file():
        raise FileNotFoundError(f"DipDECK source is missing: {source}")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if digest != DIPDECK_SOURCE_SHA256:
        raise RuntimeError(
            f"DipDECK source mismatch: expected {DIPDECK_SOURCE_SHA256}, found {digest}"
        )


def _package_version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "unavailable"


def _thread_counts() -> dict[str, int | str]:
    from threadpoolctl import threadpool_info

    counts: dict[str, int | str] = {
        "numpy_blas_threads": "", "scipy_blas_threads": "",
        "sklearn_openmp_threads": "", "torch_openmp_threads": "",
    }
    for pool in threadpool_info():
        path = str(pool.get("filepath", "")).lower()
        threads = int(pool.get("num_threads", 0))
        if pool.get("user_api") == "blas" and "numpy" in path:
            counts["numpy_blas_threads"] = threads
        elif pool.get("user_api") == "blas" and "scipy" in path:
            counts["scipy_blas_threads"] = threads
        elif pool.get("user_api") == "openmp" and "scikit" in path:
            counts["sklearn_openmp_threads"] = threads
        elif pool.get("user_api") == "openmp" and "torch" in path:
            counts["torch_openmp_threads"] = threads
    return counts


def _create_dipdeck(
    config: SmokeConfig, device: str, source_root: Path | None
) -> tuple[Any, dict[str, Any]]:
    checkout = _discover_clustpy_source(source_root)
    _verify_clustpy_source(checkout)
    import torch
    from sklearn.cluster import KMeans
    from clustpy.deep import DipDECK

    installed_version = _package_version("clustpy")
    installed_source = Path(inspect.getsourcefile(DipDECK) or "")
    installed_digest = (
        hashlib.sha256(installed_source.read_bytes()).hexdigest()
        if installed_source.is_file() else "missing"
    )
    if installed_version != CLUSTPY_VERSION or installed_digest != DIPDECK_SOURCE_SHA256:
        raise RuntimeError(
            "Imported ClustPy does not match the selected release: "
            f"version={installed_version}, DipDECK SHA256={installed_digest}"
        )
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Smoke run requested CUDA, but torch.cuda.is_available() is false")
    torch_device = torch.device(device)
    estimator = DipDECK(
        n_clusters_init=config.n_clusters_init,
        dip_merge_threshold=config.dip_merge_threshold,
        clustering_loss_weight=1.0,
        ssl_loss_weight=1.0,
        max_n_clusters=np.inf,
        min_n_clusters=1,
        batch_size=config.batch_size,
        pretrain_optimizer_params={"lr": 1e-3},
        clustering_optimizer_params={"lr": 1e-4},
        pretrain_epochs=config.pretrain_epochs,
        clustering_epochs=config.clustering_epochs,
        optimizer_class=torch.optim.Adam,
        ssl_loss_fn=torch.nn.MSELoss(reduction="mean"),
        neural_network=None,
        neural_network_weights=None,
        embedding_size=config.embedding_size,
        max_cluster_size_diff_factor=2,
        pval_strategy="table",
        n_boots=1_000,
        custom_dataloaders=None,
        augmentation_invariance=False,
        initial_clustering_class=KMeans,
        initial_clustering_params={},
        device=torch_device,
        random_state=config.seed,
        debug=True,
    )
    runtime = {
        "torch": torch,
        "device_name": torch.cuda.get_device_name(torch_device) if device == "cuda" else "CPU",
        "torch_intraop_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        **_thread_counts(),
    }
    return estimator, runtime


def _validate_fitted_outputs(
    estimator: Any, X: np.ndarray, protein_ids: np.ndarray, embedding_size: int
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    labels = np.asarray(estimator.labels_)
    if labels.ndim != 1 or labels.shape != (len(X),) or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("DipDECK labels must be a one-dimensional integer array aligned to input")
    labels = labels.astype(np.int32, copy=False)
    clusters = np.unique(labels)
    n_clusters = int(estimator.n_clusters_)
    if not np.array_equal(clusters, np.arange(n_clusters, dtype=clusters.dtype)):
        raise ValueError("DipDECK labels must be contiguous from zero with no empty cluster IDs")
    predicted = _validate_predicted_labels(
        estimator.predict(X), n_records=len(X), n_clusters=n_clusters
    )
    if np.mean(labels == predicted) <= 0.99:
        raise ValueError(
            "DipDECK training labels must agree with predictions on more than 99% of input rows"
        )
    centers_input = np.asarray(estimator.cluster_centers_, dtype=np.float32)
    learned = np.asarray(estimator.transform(X), dtype=np.float32)
    centers_learned = np.asarray(estimator.transform(centers_input), dtype=np.float32)
    arrays = {
        "cluster_center_input": centers_input,
        "learned_embedding": learned,
        "cluster_center_learned": centers_learned,
    }
    expected = {
        "cluster_center_input": (n_clusters, X.shape[1]),
        "learned_embedding": (len(X), embedding_size),
        "cluster_center_learned": (n_clusters, embedding_size),
    }
    for name, shape in expected.items():
        if arrays[name].shape != shape or not np.isfinite(arrays[name]).all():
            raise ValueError(f"DipDECK {name} has invalid shape or non-finite values")
    match_counts, matching_label_counts, cluster_rows = [], [], []
    for cluster, center in enumerate(centers_input):
        matches = np.all(X == center, axis=1)
        match_count = int(matches.sum())
        matching_label_count = int(np.count_nonzero(matches & (labels == cluster)))
        if match_count == 0:
            raise ValueError(f"DipDECK centre {cluster} is not an exact supplied input row")
        match_counts.append(match_count)
        matching_label_counts.append(matching_label_count)
        members = int(np.count_nonzero(labels == cluster))
        cluster_rows.append({
            "cluster": cluster,
            "size": members,
            "proportion": members / len(X),
            "representative_protein_id": str(protein_ids[matches][0]) if match_count == 1 else "",
            "representative_match_count": match_count,
            "representative_matching_label_count": matching_label_count,
        })
    payload = {
        "protein_id": protein_ids.astype(str),
        "label": labels,
        "learned_embedding": learned,
        "cluster_center_input": centers_input,
        "cluster_center_learned": centers_learned,
        "center_input_match_count": np.asarray(match_counts, dtype=np.int32),
        "center_matching_label_count": np.asarray(matching_label_counts, dtype=np.int32),
    }
    return payload, cluster_rows


def _scientific_row_id(
    config: SmokeConfig, held_out_fold: str, *, dedup: bool = False, input_space: str = "raw1024"
) -> str:
    threshold = str(config.dip_merge_threshold).replace(".", "p")
    records = "_dedup" if dedup else ""
    return (
        f"dipdeck_scientific{records}_{input_space}_fold{held_out_fold}_"
        f"k{config.n_clusters_init:03d}_t{threshold}_s{config.seed:03d}"
    )


def _fit_scientific_fold(
    split: FoldSplit,
    config: SmokeConfig,
    *,
    device: str,
    source_root: Path | None,
    estimator_factory: Callable[[SmokeConfig, str, Path | None], tuple[Any, dict[str, Any]]] = _create_dipdeck,
    dedup: bool = False,
    input_space: str = "raw1024",
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, np.ndarray], str]:
    train_X, train_ids = _canonical_input(split.train.X, split.train.ids)
    test_X, test_ids = _canonical_input(split.test.X, split.test.ids)
    _validate_configuration(
        train_X,
        n_clusters_init=config.n_clusters_init,
        dip_merge_threshold=config.dip_merge_threshold,
        embedding_size=config.embedding_size,
        batch_size=config.batch_size,
        pretrain_epochs=config.pretrain_epochs,
        clustering_epochs=config.clustering_epochs,
        seed=config.seed,
    )
    row = _scientific_row_id(
        config, split.held_out_fold, dedup=dedup, input_space=input_space
    )
    stream = io.StringIO()
    caught: list[warnings.WarningMessage] = []
    runtime: dict[str, Any] = {}
    elapsed = 0.0
    peak_cuda_bytes: int | str = ""
    peak_resident_bytes: int | str = ""
    clustering_epoch_updates: int | str = ""
    clustering_epoch_status = "unavailable"
    try:
        estimator, runtime = estimator_factory(config, device, source_root)
        torch = runtime.get("torch")
        if device == "cuda" and torch is not None:
            torch.cuda.reset_peak_memory_stats()
        with warnings.catch_warnings(record=True) as caught, redirect_stdout(stream), redirect_stderr(stream):
            warnings.simplefilter("always")
            with _count_dipdeck_clustering_epochs(estimator) as epoch_updates:
                with _PeakResidentMemory() as resident_memory:
                    started = perf_counter()
                    estimator.fit(train_X)
                    elapsed = perf_counter() - started
            peak_resident_bytes = resident_memory.peak_bytes
            if epoch_updates is not None:
                clustering_epoch_updates = epoch_updates[0]
                clustering_epoch_status = "measured_selected_source_tqdm"
            train_payload, _train_cluster_rows = _validate_fitted_outputs(
                estimator, train_X, train_ids, config.embedding_size
            )
            test_labels = _validate_predicted_labels(
                estimator.predict(test_X), n_records=len(test_X), n_clusters=int(estimator.n_clusters_)
            )
        if device == "cuda" and torch is not None:
            peak_cuda_bytes = int(torch.cuda.max_memory_allocated())
    except Exception:
        _append_warning_records(stream, caught)
        raise
    _append_warning_records(stream, caught)
    scores = _held_out_internal_scores(test_X, test_labels)
    test_sizes = np.bincount(test_labels, minlength=int(estimator.n_clusters_))
    cluster_rows = [
        {
            "row": row,
            "held_out_fold": split.held_out_fold,
            "cluster": cluster,
            "size": int(size),
            "proportion": size / len(test_X),
        }
        for cluster, size in enumerate(test_sizes)
        if size
    ]
    network_layers, parameter_count, parameter_bytes = _network_stats(estimator, runtime)
    versions = _runtime_versions(runtime)
    run_row = {
        "row": row,
        "run_role": "scientific_graphpart_fold_dedup" if dedup else "scientific_graphpart_fold",
        "held_out_fold": split.held_out_fold,
        "train_n": len(train_X),
        "train_subset_hash": split.train.subset_hash,
        "test_n": len(test_X),
        "test_subset_hash": split.test.subset_hash,
        "removed_n": len(split.removed_ids),
        "fold_assignment_sha256": split.assignment_sha256,
        "input_space": input_space,
        "n_dims": train_X.shape[1],
        "seed": config.seed,
        "n_clusters_init": config.n_clusters_init,
        "dip_merge_threshold": config.dip_merge_threshold,
        "embedding_size": config.embedding_size,
        "batch_size": config.batch_size,
        "pretrain_epochs": config.pretrain_epochs,
        "clustering_epochs": config.clustering_epochs,
        "clustpy_version": CLUSTPY_VERSION,
        "clustpy_commit": CLUSTPY_COMMIT,
        "dipdeck_source_sha256": DIPDECK_SOURCE_SHA256,
        "clustering_epoch_updates": clustering_epoch_updates,
        "clustering_epoch_status": clustering_epoch_status,
        "n_training_clusters": int(estimator.n_clusters_),
        "n_test_clusters": int(np.count_nonzero(test_sizes)),
        "warning_count": len(caught),
        "merge_count": sum(bool(MERGE_RECORD.match(line)) for line in stream.getvalue().splitlines()),
        "elapsed_seconds": elapsed,
        "peak_cuda_bytes": peak_cuda_bytes,
        "peak_cuda_status": "measured_torch" if device == "cuda" else "not_cuda",
        "peak_resident_bytes": peak_resident_bytes,
        "peak_resident_status": "measured_psutil_sampled_50ms",
        "network_layers": network_layers,
        "model_parameter_count": parameter_count,
        "model_parameter_bytes": parameter_bytes,
        "status": "success",
        "git_sha": _git_sha(),
        "clustpy_version": CLUSTPY_VERSION,
        "clustpy_commit": CLUSTPY_COMMIT,
        "dipdeck_source_sha256": DIPDECK_SOURCE_SHA256,
        "python_version": platform.python_version(),
        **scores,
        **versions,
    }
    payload = {
        "protein_id": test_ids,
        "label": test_labels,
        "training_protein_id": train_ids,
        "training_label": train_payload["label"],
        "cluster_center_input": train_payload["cluster_center_input"],
        "cluster_center_learned": train_payload["cluster_center_learned"],
    }
    if "pca" in input_space:
        payload["test_input"] = test_X
    return run_row, cluster_rows, payload, stream.getvalue()


def _validate_realised_clustering_epochs(
    updates: float | int | str, status: str, nominal_epochs: int, n_clusters: int
) -> None:
    count = float(updates)
    if (
        status != "measured_selected_source_tqdm"
        or not np.isfinite(count)
        or not count.is_integer()
        or count < 1
        or (n_clusters > 1 and count < nominal_epochs)
    ):
        raise ValueError("scientific DipDECK run does not record valid realised clustering epochs")


def _validate_scientific_outputs(
    paths: ScientificPaths,
    expected_rows: int,
    dataset: Dataset | None = None,
    *,
    dedup: bool = False,
    pca_dimensions: int | None = None,
    embedding: str = "prott5",
    seeds: tuple[int, ...] = SCIENTIFIC_SEEDS,
) -> pd.DataFrame:
    runs = pd.read_csv(paths.runs)
    clusters = pd.read_csv(paths.clusters)
    if tuple(runs.columns) != SCIENTIFIC_RUN_COLUMNS or len(runs) != expected_rows:
        raise ValueError("scientific DipDECK run table does not match its expected schema or size")
    if tuple(clusters.columns) != SCIENTIFIC_CLUSTER_COLUMNS:
        raise ValueError("scientific DipDECK cluster table does not match its schema")
    if runs["row"].duplicated().any() or not (runs["status"] == "success").all():
        raise ValueError("scientific DipDECK runs must have unique successful row IDs")
    if not (runs["clustpy_commit"] == CLUSTPY_COMMIT).all() or not (
        runs["dipdeck_source_sha256"] == DIPDECK_SOURCE_SHA256
    ).all():
        raise ValueError("scientific DipDECK runs do not record the selected ClustPy source")
    expected_archives = {f"{row}.npz" for row in runs["row"]}
    expected_logs = {f"{row}.log" for row in runs["row"]}
    if {path.name for path in paths.assignments.glob("*.npz")} != expected_archives:
        raise ValueError("scientific DipDECK assignment archives do not match run rows")
    if {path.name for path in paths.logs.glob("*.log")} != expected_logs:
        raise ValueError("scientific DipDECK logs do not match run rows")
    scientific_splits: dict[str, FoldSplit] = {}
    expected_run_rows: dict[str, SmokeConfig] = {}
    declared_configs = _scientific_configs(seeds=seeds)
    if expected_rows == len(SCIENTIFIC_FOLDS) * len(declared_configs):
        live_dataset = _load_embedding_dataset(embedding) if dataset is None else dataset
        input_space = _input_space(embedding, pca_dimensions)
        if pca_dimensions is None:
            splits = _scientific_fold_splits(live_dataset, dedup=dedup)
        else:
            manifest = json.loads(paths.manifest.read_text(encoding="utf-8"))
            pca_folds = manifest.get("pca_folds")
            raw_splits = _scientific_fold_splits(live_dataset, dedup=dedup)
            if not isinstance(pca_folds, list) or len(pca_folds) != len(raw_splits):
                raise ValueError("scientific DipDECK PCA provenance does not cover every fold")
            expected_metadata = {
                split.held_out_fold: (len(split.train), split.train.subset_hash)
                for split in raw_splits
            }
            for metadata in pca_folds:
                fold = str(metadata.get("held_out_fold"))
                if (
                    fold not in expected_metadata
                    or metadata.get("input_space") != input_space
                    or metadata.get("pca_dimensions") != pca_dimensions
                    or tuple(
                        (metadata.get("pca_fit_records"), metadata.get("pca_fit_subset_hash"))
                    ) != expected_metadata[fold]
                    or not re.fullmatch(r"[0-9a-f]{64}", str(metadata.get("pca_components_sha256")))
                    or not 0 < float(metadata.get("variance_retained", 0)) <= 1
                ):
                    raise ValueError("scientific DipDECK PCA provenance is invalid")
            splits = raw_splits
            recomputed_test_inputs = {
                split.held_out_fold: split.test.X
                for split in _pca_scientific_splits(
                    live_dataset, pca_dimensions, dedup=dedup, embedding=embedding
                )[0]
            }
            checked_test_inputs: dict[str, np.ndarray] = {}
        scientific_splits = {
            split.held_out_fold: split
            for split in splits
        }
        expected_run_rows = {
            _scientific_row_id(
                config, split.held_out_fold, dedup=dedup, input_space=input_space
            ): config
            for split in scientific_splits.values()
            for config in declared_configs
        }
        if set(runs["row"]) != set(expected_run_rows):
            raise ValueError("scientific DipDECK runs do not contain the declared fold grid")
        expected_role = "scientific_graphpart_fold_dedup" if dedup else "scientific_graphpart_fold"
        if not (runs["run_role"] == expected_role).all():
            raise ValueError("scientific DipDECK run role does not match the declared record set")
    for run in runs.to_dict("records"):
        split = scientific_splits.get(str(run["held_out_fold"]))
        config = expected_run_rows.get(str(run["row"]))
        if split is not None and config is not None:
            expected_fields = {
                "train_n": len(split.train),
                "train_subset_hash": split.train.subset_hash,
                "test_n": len(split.test),
                "test_subset_hash": split.test.subset_hash,
                "removed_n": len(split.removed_ids),
                "fold_assignment_sha256": split.assignment_sha256,
                "input_space": input_space,
                "n_dims": EMBEDDINGS[embedding].dims if pca_dimensions is None else pca_dimensions,
                "seed": config.seed,
                "n_clusters_init": config.n_clusters_init,
                "dip_merge_threshold": config.dip_merge_threshold,
                "embedding_size": config.embedding_size,
                "batch_size": config.batch_size,
                "pretrain_epochs": config.pretrain_epochs,
                "clustering_epochs": config.clustering_epochs,
            }
            for field, expected in expected_fields.items():
                if run[field] != expected:
                    raise ValueError(
                        f"scientific DipDECK run {run['row']} has unexpected {field}"
                    )
            _validate_realised_clustering_epochs(
                run["clustering_epoch_updates"], run["clustering_epoch_status"],
                int(run["clustering_epochs"]), int(run["n_training_clusters"]),
            )
        row_clusters = clusters.loc[clusters["row"] == run["row"]].sort_values("cluster")
        if (
            len(row_clusters) != int(run["n_test_clusters"])
            or int(row_clusters["size"].sum()) != int(run["test_n"])
        ):
            raise ValueError("scientific DipDECK cluster rows do not reproduce held-out assignments")
        with np.load(paths.assignments / f"{run['row']}.npz", allow_pickle=False) as archive:
            required = {
                "protein_id", "label", "training_protein_id", "training_label",
                "cluster_center_input", "cluster_center_learned",
            }
            if pca_dimensions is not None:
                required.add("test_input")
            if set(archive.files) != required:
                raise ValueError("scientific DipDECK assignment archive does not match its schema")
            protein_ids = archive["protein_id"]
            labels = _validate_predicted_labels(
                archive["label"], n_records=int(run["test_n"]), n_clusters=int(run["n_training_clusters"])
            )
            training_ids = archive["training_protein_id"]
            if (
                protein_ids.dtype.kind != "U"
                or training_ids.dtype.kind != "U"
                or protein_ids.tolist() != sorted(protein_ids.tolist())
                or training_ids.tolist() != sorted(training_ids.tolist())
                or len(set(protein_ids.tolist())) != len(protein_ids)
                or len(set(training_ids.tolist())) != len(training_ids)
                or set(protein_ids) & set(training_ids)
                or len(training_ids) != int(run["train_n"])
            ):
                raise ValueError("scientific DipDECK archive IDs do not reproduce the fold split")
            if split is not None:
                test_X, test_ids = _canonical_input(split.test.X, split.test.ids)
                if protein_ids.tolist() != test_ids.tolist():
                    raise ValueError("scientific DipDECK archive IDs do not match held-out input")
                if training_ids.tolist() != sorted(split.train.ids):
                    raise ValueError("scientific DipDECK archive training IDs do not match the fold")
                scoring_X = archive["test_input"] if pca_dimensions is not None else test_X
                if pca_dimensions is not None:
                    fold = str(run["held_out_fold"])
                    key = (fold, hashlib.sha256(np.ascontiguousarray(scoring_X).tobytes()).hexdigest())
                    if key not in checked_test_inputs:
                        _check_same_pca_coordinates(
                            scoring_X, recomputed_test_inputs[fold], f"fold {fold}"
                        )
                        checked_test_inputs[key] = scoring_X
                if (
                    scoring_X.dtype != np.float32
                    or scoring_X.shape != (len(test_ids), int(run["n_dims"]))
                    or not np.isfinite(scoring_X).all()
                ):
                    raise ValueError("scientific DipDECK PCA archive input is invalid")
                expected_scores = _held_out_internal_scores(scoring_X, labels)
                if run["score_status"] != expected_scores["score_status"]:
                    raise ValueError("scientific DipDECK score status does not match assignments")
                for field in ("silhouette", "calinski_harabasz", "davies_bouldin"):
                    expected_score = expected_scores[field]
                    observed_score = run[field]
                    if expected_score == "":
                        if not pd.isna(observed_score):
                            raise ValueError(
                                f"scientific DipDECK undefined {field} must be blank"
                            )
                    elif not _stored_score_matches(field, observed_score, expected_score):
                        raise ValueError(
                            f"scientific DipDECK {field} does not match held-out assignments"
                        )
            expected_sizes = np.bincount(labels, minlength=int(run["n_training_clusters"]))
            if not np.array_equal(
                row_clusters["size"].to_numpy(dtype=int),
                expected_sizes[expected_sizes > 0],
            ):
                raise ValueError("scientific DipDECK cluster sizes do not match assignment labels")
    return runs


def _persist_scientific_outputs(
    paths: ScientificPaths,
    runs: list[dict[str, Any]],
    clusters: list[dict[str, Any]],
    archives: dict[str, dict[str, np.ndarray]],
    logs: dict[str, str],
    *,
    dedup: bool = False,
    pca_metadata: list[dict[str, Any]] | None = None,
    embedding: str = "prott5",
    seeds: tuple[int, ...] = SCIENTIFIC_SEEDS,
) -> pd.DataFrame:
    run_table = pd.DataFrame(runs, columns=SCIENTIFIC_RUN_COLUMNS)
    cluster_table = pd.DataFrame(clusters, columns=SCIENTIFIC_CLUSTER_COLUMNS)
    run_table.to_csv(paths.runs, index=False)
    cluster_table.to_csv(paths.clusters, index=False)
    for row, archive in archives.items():
        np.savez_compressed(paths.assignments / f"{row}.npz", **archive)
    for row, log in logs.items():
        (paths.logs / f"{row}.log").write_text(log, encoding="utf-8")
    manifest = {
        "rows": len(runs), "embedding": embedding, "seeds": list(seeds),
        "run_ids": [run["row"] for run in runs],
    }
    if pca_metadata is not None:
        manifest["pca_folds"] = pca_metadata
    paths.manifest.write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    return _validate_scientific_outputs(
        paths,
        expected_rows=len(runs),
        dedup=dedup,
        pca_dimensions=None if pca_metadata is None else int(pca_metadata[0]["pca_dimensions"]),
        embedding=embedding,
        seeds=seeds,
    )


def _network_stats(estimator: Any, runtime: dict[str, Any]) -> tuple[str, int, int]:
    if "model_stats" in runtime:
        return runtime["model_stats"]
    torch = runtime["torch"]
    network = estimator.neural_network_trained_
    linear = [module for module in network.modules() if isinstance(module, torch.nn.Linear)]
    if not linear:
        raise ValueError("fitted DipDECK network contains no linear layers")
    layers = [linear[0].in_features, *(module.out_features for module in linear)]
    parameters = list(network.parameters())
    return (
        "-".join(str(width) for width in layers),
        sum(parameter.numel() for parameter in parameters),
        sum(parameter.numel() * parameter.element_size() for parameter in parameters),
    )


def _runtime_versions(runtime: dict[str, Any]) -> dict[str, str]:
    distributions = {
        "torch_version": "torch", "torchvision_version": "torchvision",
        "numpy_version": "numpy", "scipy_version": "scipy",
        "sklearn_version": "scikit-learn", "pandas_version": "pandas",
        "h5py_version": "h5py", "pyarrow_version": "pyarrow", "tqdm_version": "tqdm",
        "threadpoolctl_version": "threadpoolctl",
    }
    if "versions" in runtime:
        return runtime["versions"]
    return {field: _package_version(distribution) for field, distribution in distributions.items()}


def _validate_archive_arrays(arrays: Any, run_row: dict[str, Any]) -> None:
    required = {
        "protein_id", "label", "learned_embedding", "cluster_center_input",
        "cluster_center_learned", "center_input_match_count", "center_matching_label_count",
    }
    if "pca" in str(run_row.get("input_space", "")):
        required = required | {"input"}
    if set(arrays.keys()) != required:
        raise ValueError("reopened DipDECK archive arrays do not match the expected schema")
    protein_ids = arrays["protein_id"]
    labels = arrays["label"]
    learned = arrays["learned_embedding"]
    centers_input = arrays["cluster_center_input"]
    centers_learned = arrays["cluster_center_learned"]
    match_counts = arrays["center_input_match_count"]
    matching_label_counts = arrays["center_matching_label_count"]
    n = int(run_row["n"])
    n_clusters = int(run_row["n_clusters"])
    n_dims = int(run_row["n_dims"])
    embedding_size = int(run_row["embedding_size"])

    if protein_ids.dtype.kind != "U" or protein_ids.shape != (n,):
        raise ValueError("reopened DipDECK protein IDs must be a one-dimensional Unicode array")
    protein_values = protein_ids.tolist()
    if any(not protein_id for protein_id in protein_values) or len(set(protein_values)) != n:
        raise ValueError("reopened DipDECK protein IDs must be non-empty and unique")
    if protein_values != sorted(protein_values):
        raise ValueError("reopened DipDECK protein IDs are not in canonical order")
    if labels.dtype != np.int32 or labels.shape != (n,):
        raise ValueError("reopened DipDECK labels have the wrong dtype or shape")
    unique_labels = np.unique(labels)
    if not np.array_equal(unique_labels, np.arange(n_clusters, dtype=np.int32)):
        raise ValueError("reopened DipDECK labels do not reproduce the recorded cluster count")

    expected_float_arrays = {
        "learned_embedding": (learned, (n, embedding_size)),
        "cluster_center_input": (centers_input, (n_clusters, n_dims)),
        "cluster_center_learned": (centers_learned, (n_clusters, embedding_size)),
    }
    if "input" in required:
        expected_float_arrays["input"] = (arrays["input"], (n, n_dims))
    for name, (array, shape) in expected_float_arrays.items():
        if array.dtype != np.float32 or array.shape != shape or not np.isfinite(array).all():
            raise ValueError(f"reopened DipDECK {name} has the wrong dtype, shape, or values")
    for name, array in {
        "center_input_match_count": match_counts,
        "center_matching_label_count": matching_label_counts,
    }.items():
        if array.dtype != np.int32 or array.shape != (n_clusters,):
            raise ValueError(f"reopened DipDECK {name} has the wrong dtype or shape")
    if (match_counts <= 0).any():
        raise ValueError("reopened DipDECK centre match counts must be positive")
    if (matching_label_counts < 0).any() or (matching_label_counts > match_counts).any():
        raise ValueError("reopened DipDECK matching-label counts are outside their bounds")


def _validate_existing_outputs(
    paths: OutputPaths,
) -> tuple[pd.DataFrame, pd.DataFrame, Path]:
    run_table = pd.read_csv(paths.runs)
    cluster_table = pd.read_csv(paths.clusters)
    if tuple(run_table.columns) != RUN_COLUMNS or len(run_table) != 1:
        raise ValueError("reopened DipDECK run table does not match the expected schema")
    if tuple(cluster_table.columns) != CLUSTER_COLUMNS:
        raise ValueError("reopened DipDECK cluster table does not match the expected schema")
    run_row = run_table.iloc[0].to_dict()
    row = str(run_row["row"])
    n = int(run_row["n"])
    n_clusters = int(run_row["n_clusters"])
    if run_row["status"] != "success":
        raise ValueError("reopened DipDECK run is not marked successful")
    accepted_sources = {(CLUSTPY_COMMIT, DIPDECK_SOURCE_SHA256)}
    if str(run_row["run_role"]).startswith("engineering_"):
        accepted_sources.add((LEGACY_CLUSTPY_COMMIT, LEGACY_DIPDECK_SOURCE_SHA256))
    if (
        run_row["clustpy_version"] != CLUSTPY_VERSION
        or (run_row["clustpy_commit"], run_row["dipdeck_source_sha256"]) not in accepted_sources
    ):
        raise ValueError("reopened DipDECK run does not identify the selected release")
    if len(cluster_table) != n_clusters or set(cluster_table["row"].astype(str)) != {row}:
        raise ValueError("reopened DipDECK cluster table does not match the run row")
    ordered_clusters = cluster_table.sort_values("cluster", kind="stable")
    if not np.array_equal(
        ordered_clusters["cluster"].to_numpy(dtype=int), np.arange(n_clusters)
    ):
        raise ValueError("reopened DipDECK cluster rows do not cover every cluster")
    sizes = ordered_clusters["size"].to_numpy(dtype=int)
    if (sizes <= 0).any() or int(sizes.sum()) != n:
        raise ValueError("reopened DipDECK cluster sizes do not reproduce the input count")

    archives = list(paths.assignments.glob("*.npz"))
    if [path.name for path in archives] != [f"{row}.npz"]:
        raise ValueError("Smoke validation requires exactly one matching assignment archive")
    archive = archives[0]
    with np.load(archive, allow_pickle=False) as loaded:
        _validate_archive_arrays(loaded, run_row)
        expected_sizes = np.bincount(loaded["label"], minlength=n_clusters)
        if not np.array_equal(sizes, expected_sizes):
            raise ValueError("reopened DipDECK cluster sizes do not match the assignment archive")
        for table_column, archive_name in (
            ("representative_match_count", "center_input_match_count"),
            ("representative_matching_label_count", "center_matching_label_count"),
        ):
            if not np.array_equal(
                ordered_clusters[table_column].to_numpy(dtype=np.int32), loaded[archive_name]
            ):
                raise ValueError(f"reopened DipDECK {table_column} does not match the archive")
    logs = list(paths.logs.glob("*.log"))
    if [path.name for path in logs] != [f"{row}.log"]:
        raise ValueError("Smoke validation requires exactly one matching per-run log")
    return run_table, cluster_table, archive


def _persist_and_reopen(
    paths: OutputPaths, row: str, run_row: dict[str, Any], cluster_rows: list[dict[str, Any]],
    payload: dict[str, np.ndarray],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    run_table = pd.DataFrame([run_row], columns=RUN_COLUMNS)
    cluster_table = pd.DataFrame(
        [{"row": row, **cluster_row} for cluster_row in cluster_rows], columns=CLUSTER_COLUMNS
    )
    run_table.to_csv(paths.runs, index=False)
    cluster_table.to_csv(paths.clusters, index=False)
    archive = paths.assignments / f"{row}.npz"
    np.savez_compressed(archive, **payload)
    reopened_run, reopened_clusters, reopened_archive = _validate_existing_outputs(paths)
    with np.load(reopened_archive, allow_pickle=False) as loaded:
        _validate_archive_arrays(loaded, run_row)
        for name, expected_array in payload.items():
            if not np.array_equal(loaded[name], expected_array):
                raise ValueError(f"reopened DipDECK archive changed {name}")
    return reopened_run, reopened_clusters


def _append_warning_records(stream: io.StringIO, caught: list[warnings.WarningMessage]) -> None:
    for item in caught:
        stream.write(
            f"WARNING {item.category.__name__}: {item.message} ({item.filename}:{item.lineno})\n"
        )


def _run_configuration(
    dataset: Dataset, paths: OutputPaths, *, config: SmokeConfig = SMOKE_CONFIG,
    identity: RunIdentity = SMOKE_IDENTITY,
    device: str = "cuda", source_root: Path | None = None,
    estimator_factory: Callable[[SmokeConfig, str, Path | None], tuple[Any, dict[str, Any]]] = _create_dipdeck,
    require_live_dataset: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    X, protein_ids = _canonical_input(dataset.X, dataset.ids)
    _validate_configuration(
        X, n_clusters_init=config.n_clusters_init,
        dip_merge_threshold=config.dip_merge_threshold, embedding_size=config.embedding_size,
        batch_size=config.batch_size, pretrain_epochs=config.pretrain_epochs,
        clustering_epochs=config.clustering_epochs, seed=config.seed,
    )
    if require_live_dataset:
        _validate_live_dataset(dataset, X, protein_ids)
    git_sha = _git_sha()
    row = _row_id(dataset, config, identity)
    _reset_outputs(paths)
    log_path = paths.logs / f"{row}.log"
    stream = io.StringIO()
    caught: list[warnings.WarningMessage] = []
    runtime: dict[str, Any] = {}
    elapsed = 0.0
    peak_cuda_bytes: int | str = ""
    peak_resident_bytes: int | str = ""
    effective_clustering_epochs: int | str = ""
    effective_epochs_status = "unavailable_selected_release"
    try:
        estimator, runtime = estimator_factory(config, device, source_root)
        torch = runtime.get("torch")
        if device == "cuda" and torch is not None:
            torch.cuda.reset_peak_memory_stats()
        with warnings.catch_warnings(record=True) as caught, redirect_stdout(stream), redirect_stderr(stream):
            warnings.simplefilter("always")
            with _count_dipdeck_clustering_epochs(estimator) as epoch_updates:
                with _PeakResidentMemory() as resident_memory:
                    started = perf_counter()
                    estimator.fit(X)
                    elapsed = perf_counter() - started
            peak_resident_bytes = resident_memory.peak_bytes
            if epoch_updates is not None:
                effective_clustering_epochs = epoch_updates[0]
                effective_epochs_status = "measured_selected_source_tqdm"
            payload, cluster_rows = _validate_fitted_outputs(
                estimator, X, protein_ids, config.embedding_size
            )
        if device == "cuda" and torch is not None:
            peak_cuda_bytes = int(torch.cuda.max_memory_allocated())
    except Exception:
        _append_warning_records(stream, caught)
        log_path.write_text(stream.getvalue(), encoding="utf-8")
        raise
    _append_warning_records(stream, caught)
    log_path.write_text(stream.getvalue(), encoding="utf-8")
    labels = payload["label"]
    sizes = np.bincount(labels)
    network_layers, parameter_count, parameter_bytes = _network_stats(estimator, runtime)
    versions = _runtime_versions(runtime)
    run_row = {
        "row": row, "run_role": identity.run_role, "dataset": dataset.name,
        "records": identity.records, "n": len(X), "subset_hash": dataset.subset_hash,
        "input_space": identity.input_space, "n_dims": X.shape[1],
        "input_pipeline": identity.input_pipeline,
        "transform_fit_records": identity.transform_fit_records,
        "transform_sha256": identity.transform_sha256,
        "variance_retained": identity.variance_retained,
        "seed": config.seed, "pca_random_state": "", "dipdeck_random_state": config.seed,
        "device": device, "device_name": runtime.get("device_name", "unavailable"),
        "torch_intraop_threads": runtime.get("torch_intraop_threads", ""),
        "torch_interop_threads": runtime.get("torch_interop_threads", ""),
        "numpy_blas_threads": runtime.get("numpy_blas_threads", ""),
        "scipy_blas_threads": runtime.get("scipy_blas_threads", ""),
        "sklearn_openmp_threads": runtime.get("sklearn_openmp_threads", ""),
        "torch_openmp_threads": runtime.get("torch_openmp_threads", ""),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS", ""),
        "mkl_num_threads": os.environ.get("MKL_NUM_THREADS", ""),
        "openblas_num_threads": os.environ.get("OPENBLAS_NUM_THREADS", ""),
        "deterministic_algorithms": runtime.get("deterministic_algorithms", False),
        "n_clusters_init": config.n_clusters_init,
        "dip_merge_threshold": config.dip_merge_threshold, "embedding_size": config.embedding_size,
        "max_cluster_size_diff_factor": 2, "pval_strategy": "table", "min_n_clusters": 1,
        "max_n_clusters": np.inf, "augmentation_invariance": False,
        "network_layers": network_layers, "activation": "LeakyReLU", "batch_size": config.batch_size,
        "pretrain_epochs": config.pretrain_epochs, "clustering_epochs": config.clustering_epochs,
        "pretrain_learning_rate": 1e-3, "clustering_learning_rate": 1e-4,
        "optimizer": "Adam", "ssl_loss": "MSELoss", "ssl_loss_reduction": "mean",
        "ssl_loss_weight": 1.0, "clustering_loss_weight": 1.0, "status": "success",
        "n_clusters": len(sizes), "cluster_size_minimum": int(sizes.min()),
        "cluster_size_median": float(np.median(sizes)), "cluster_size_maximum": int(sizes.max()),
        "warning_count": len(caught),
        "merge_count": sum(bool(MERGE_RECORD.match(line)) for line in stream.getvalue().splitlines()),
        "effective_clustering_epochs": effective_clustering_epochs,
        "effective_epochs_status": effective_epochs_status,
        "clustering_space": "learned", "clustering_distance": "squared_euclidean",
        "scoring_space": "", "scoring_distance": "", "elapsed_seconds": elapsed,
        "peak_cuda_bytes": peak_cuda_bytes,
        "peak_cuda_status": "measured_torch" if device == "cuda" else "not_cuda",
        "peak_resident_bytes": peak_resident_bytes,
        "peak_resident_status": "measured_psutil_sampled_50ms",
        "model_parameter_count": parameter_count, "model_parameter_bytes": parameter_bytes,
        "git_sha": git_sha, "clustpy_version": CLUSTPY_VERSION, "clustpy_commit": CLUSTPY_COMMIT,
        "dipdeck_source_sha256": DIPDECK_SOURCE_SHA256, "python_version": platform.python_version(),
        **versions,
    }
    if "pca" in identity.input_space:
        payload["input"] = X
    return _persist_and_reopen(paths, row, run_row, cluster_rows, payload)


def _prepare_scaling_inputs(dataset: Dataset) -> tuple[Dataset, Dataset, np.ndarray, np.ndarray, str]:
    X, protein_ids = _canonical_input(dataset.X, dataset.ids)
    _validate_live_dataset(dataset, X, protein_ids)
    standardised, feature_mean, feature_scale, digest = _standardise_features(X)
    empty_annotations = pd.DataFrame(index=np.arange(len(X)))
    raw_dataset = Dataset(dataset.name, protein_ids.tolist(), X, empty_annotations)
    standardised_dataset = Dataset(
        dataset.name, protein_ids.tolist(), standardised, empty_annotations.copy()
    )
    return raw_dataset, standardised_dataset, feature_mean, feature_scale, digest


def _pretraining_loss_observations(log_path: Path) -> tuple[float | str, float | str, str]:
    values = [float(value) for value in PRETRAINING_LOSS_RECORD.findall(
        log_path.read_text(encoding="utf-8")
    )]
    if not values:
        return "", "", "unavailable_selected_release_log"
    return values[0], values[-1], "observed_selected_release_tqdm_postfix"


def _reopened_labels(paths: OutputPaths) -> tuple[pd.DataFrame, np.ndarray]:
    run_table, _cluster_table, archive = _validate_existing_outputs(paths)
    with np.load(archive, allow_pickle=False) as loaded:
        return run_table, loaded["label"].copy()


def _scaling_summary_row(paths: OutputPaths, run_table: pd.DataFrame) -> dict[str, Any]:
    run = run_table.iloc[0]
    first_loss, final_loss, loss_status = _pretraining_loss_observations(
        paths.logs / f"{run['row']}.log"
    )
    return {
        "input_space": run["input_space"], "run_row": run["row"],
        "input_pipeline": run["input_pipeline"],
        "transform_fit_records": run["transform_fit_records"],
        "transform_sha256": run["transform_sha256"],
        "pretraining_loss_first_observed": first_loss,
        "pretraining_loss_final_observed": final_loss,
        "pretraining_loss_status": loss_status, "finite_outputs": True,
        "n_clusters": int(run["n_clusters"]), "elapsed_seconds": float(run["elapsed_seconds"]),
    }


def _validate_scaling_outputs(paths: ScalingPaths) -> pd.DataFrame:
    transform_digest, _protein_ids = _validate_standardiser(paths.standardiser)
    raw_run, raw_labels = _reopened_labels(paths.raw)
    standardised_run, standardised_labels = _reopened_labels(paths.standardised)
    expected_runs = {
        "raw1024": raw_run.iloc[0], "zraw1024": standardised_run.iloc[0],
    }
    for input_space, run in expected_runs.items():
        if (
            run["run_role"] != "engineering_scaling_preflight"
            or run["input_space"] != input_space
            or int(run["pretrain_epochs"]) != SMOKE_CONFIG.pretrain_epochs
            or int(run["clustering_epochs"]) != SMOKE_CONFIG.clustering_epochs
        ):
            raise ValueError(f"Scaling {input_space} run identity or epoch settings changed")
    standardised = standardised_run.iloc[0]
    if (
        standardised["transform_fit_records"] != "full_unsplit_engineering_population"
        or standardised["transform_sha256"] != transform_digest
    ):
        raise ValueError("Scaling standardised run does not identify its fitted transform")
    if not pd.isna(raw_run.loc[0, "transform_sha256"]):
        raise ValueError("Scaling raw run must not identify an input transform")

    summary = pd.read_csv(paths.scaling_summary)
    if tuple(summary.columns) != SCALING_COLUMNS or set(summary["input_space"]) != set(expected_runs):
        raise ValueError("Scaling summary does not match its schema or expected arms")
    expected_ari = float(adjusted_rand_score(raw_labels, standardised_labels))
    for input_space, run in expected_runs.items():
        row = summary.loc[summary["input_space"] == input_space]
        if len(row) != 1:
            raise ValueError(f"Scaling requires one {input_space} comparison row")
        row = row.iloc[0]
        expected_arm_ari = 1.0 if input_space == "raw1024" else expected_ari
        if (
            row["run_row"] != run["row"]
            or not isinstance(row["finite_outputs"], (bool, np.bool_))
            or not bool(row["finite_outputs"])
            or int(row["n_clusters"]) != int(run["n_clusters"])
            or not np.isclose(float(row["elapsed_seconds"]), float(run["elapsed_seconds"]))
            or not np.isclose(float(row["partition_ari_to_raw"]), expected_arm_ari)
        ):
            raise ValueError(f"Scaling {input_space} summary does not reproduce reopened outputs")
        loss_values = row[[
            "pretraining_loss_first_observed", "pretraining_loss_final_observed"
        ]]
        if row["pretraining_loss_status"] == "observed_selected_release_tqdm_postfix":
            if loss_values.isna().any() or not np.isfinite(loss_values.astype(float)).all():
                raise ValueError(f"Scaling {input_space} observed losses are invalid")
        elif row["pretraining_loss_status"] != "unavailable_selected_release_log":
            raise ValueError(f"Scaling {input_space} has an unknown loss status")
    return summary


def _run_scaling(
    output_root: Path | None, device: str, source_root: Path | None,
    estimator_factory: Callable[[SmokeConfig, str, Path | None], tuple[Any, dict[str, Any]]] = _create_dipdeck,
) -> pd.DataFrame:
    paths = _scaling_paths(output_root)
    paths.scaling_summary.parent.mkdir(parents=True, exist_ok=True)
    paths.scaling_summary.unlink(missing_ok=True)
    paths.standardiser.unlink(missing_ok=True)
    raw_dataset, standardised_dataset, feature_mean, feature_scale, transform_digest = (
        _prepare_scaling_inputs(load_dataset(DATASET))
    )
    saved_digest = _save_standardiser(
        paths.standardiser, np.asarray(raw_dataset.ids), feature_mean, feature_scale
    )
    if saved_digest != transform_digest:
        raise ValueError("In-memory and persisted standardiser digests differ")
    raw_identity = RunIdentity(run_role="engineering_scaling_preflight")
    standardised_identity = RunIdentity(
        run_role="engineering_scaling_preflight",
        input_space="zraw1024",
        input_pipeline="raw1024_then_featurewise_sklearn_standardscaler_float32",
        transform_fit_records="full_unsplit_engineering_population",
        transform_sha256=transform_digest,
    )
    raw_run, _raw_clusters = _run_configuration(
        raw_dataset, paths.raw, identity=raw_identity, device=device, source_root=source_root,
        estimator_factory=estimator_factory, require_live_dataset=False,
    )
    standardised_run, _standardised_clusters = _run_configuration(
        standardised_dataset, paths.standardised, identity=standardised_identity, device=device,
        source_root=source_root, estimator_factory=estimator_factory, require_live_dataset=False,
    )
    _validated_raw, raw_labels = _reopened_labels(paths.raw)
    _validated_standardised, standardised_labels = _reopened_labels(paths.standardised)
    raw_summary = _scaling_summary_row(paths.raw, raw_run)
    standardised_summary = _scaling_summary_row(paths.standardised, standardised_run)
    raw_summary["partition_ari_to_raw"] = 1.0
    standardised_summary["partition_ari_to_raw"] = float(
        adjusted_rand_score(raw_labels, standardised_labels)
    )
    pd.DataFrame(
        [raw_summary, standardised_summary], columns=SCALING_COLUMNS
    ).to_csv(paths.scaling_summary, index=False)
    return _validate_scaling_outputs(paths)


def _validate_full_setting_output(paths: ScalingPaths) -> pd.DataFrame:
    transform_digest, _protein_ids = _validate_standardiser(paths.standardiser)
    run_table, _labels = _reopened_labels(paths.full_setting)
    run = run_table.iloc[0]
    if (
        run["run_role"] != "engineering_full_setting_timing"
        or run["input_space"] != "zraw1024"
        or run["transform_fit_records"] != "full_unsplit_engineering_population"
        or run["transform_sha256"] != transform_digest
        or int(run["pretrain_epochs"]) != FULL_SETTING_CONFIG.pretrain_epochs
        or int(run["clustering_epochs"]) != FULL_SETTING_CONFIG.clustering_epochs
    ):
        raise ValueError("Full-setting run identity, transform, or published epoch settings changed")
    timing = pd.read_csv(paths.full_timing)
    if tuple(timing.columns) != TIMING_COLUMNS or len(timing) != 1:
        raise ValueError("Full-setting timing table does not match its schema")
    row = timing.iloc[0]
    cuda_status = str(run["peak_cuda_status"])
    cuda_bytes_match = (
        cuda_status == "measured_torch"
        and int(row["peak_cuda_bytes"]) > 0
        and int(row["peak_cuda_bytes"]) == int(run["peak_cuda_bytes"])
    ) or (
        cuda_status == "not_cuda"
        and pd.isna(row["peak_cuda_bytes"])
        and pd.isna(run["peak_cuda_bytes"])
    )
    epochs_status = str(run["effective_epochs_status"])
    if epochs_status == "measured_selected_source_tqdm":
        _validate_realised_clustering_epochs(
            run["effective_clustering_epochs"], epochs_status,
            int(run["clustering_epochs"]), int(run["n_clusters"]),
        )
    elif (
        epochs_status != "unavailable_selected_release"
        or not pd.isna(run["effective_clustering_epochs"])
    ):
        raise ValueError("Full-setting run has an invalid clustering-epoch observation")
    epoch_counts_match = (
        pd.isna(row["effective_clustering_epochs"])
        and pd.isna(run["effective_clustering_epochs"])
    ) or (
        epochs_status == "measured_selected_source_tqdm"
        and row["effective_clustering_epochs"] == run["effective_clustering_epochs"]
    )
    if (
        row["run_row"] != run["row"]
        or row["input_selection_basis"] != "quoted_paper_numerical_featurewise_z_transform"
        or row["split_policy"] != "full_unsplit_non_evaluative_engineering_timing"
        or row["phase_timing_status"] != "unavailable_selected_release_no_hooks"
        or not pd.isna(row["pretraining_seconds"])
        or not pd.isna(row["clustering_seconds"])
        or row["peak_cuda_status"] != cuda_status
        or not cuda_bytes_match
        or int(row["peak_resident_bytes"]) <= 0
        or int(row["peak_resident_bytes"]) != int(run["peak_resident_bytes"])
        or row["peak_resident_status"] != run["peak_resident_status"]
        or not np.isclose(float(row["elapsed_seconds"]), float(run["elapsed_seconds"]))
        or float(row["rss_sample_interval_seconds"]) != RSS_SAMPLE_INTERVAL_SECONDS
        or row["effective_epochs_status"] != epochs_status
        or not epoch_counts_match
    ):
        raise ValueError("Full-setting timing row does not reproduce the run or declared limitations")
    return timing


def _run_full_setting(
    output_root: Path | None, device: str, source_root: Path | None,
    estimator_factory: Callable[[SmokeConfig, str, Path | None], tuple[Any, dict[str, Any]]] = _create_dipdeck,
) -> pd.DataFrame:
    paths = _scaling_paths(output_root)
    paths.full_timing.parent.mkdir(parents=True, exist_ok=True)
    paths.full_timing.unlink(missing_ok=True)
    _raw_dataset, standardised_dataset, feature_mean, feature_scale, transform_digest = (
        _prepare_scaling_inputs(load_dataset(DATASET))
    )
    saved_digest = _save_standardiser(
        paths.standardiser, np.asarray(standardised_dataset.ids), feature_mean, feature_scale
    )
    if saved_digest != transform_digest:
        raise ValueError("In-memory and persisted standardiser digests differ")
    identity = RunIdentity(
        run_role="engineering_full_setting_timing",
        input_space="zraw1024",
        input_pipeline="raw1024_then_featurewise_sklearn_standardscaler_float32",
        transform_fit_records="full_unsplit_engineering_population",
        transform_sha256=transform_digest,
    )
    run_table, _cluster_table = _run_configuration(
        standardised_dataset, paths.full_setting, config=FULL_SETTING_CONFIG,
        identity=identity, device=device, source_root=source_root,
        estimator_factory=estimator_factory, require_live_dataset=False,
    )
    run = run_table.iloc[0]
    timing_row = {
        "run_row": run["row"], "input_space": run["input_space"],
        "input_selection_basis": "quoted_paper_numerical_featurewise_z_transform",
        "split_policy": "full_unsplit_non_evaluative_engineering_timing",
        "pretrain_epochs": int(run["pretrain_epochs"]),
        "nominal_clustering_epochs": int(run["clustering_epochs"]),
        "elapsed_seconds": float(run["elapsed_seconds"]),
        "pretraining_seconds": "", "clustering_seconds": "",
        "phase_timing_status": "unavailable_selected_release_no_hooks",
        "peak_cuda_bytes": run["peak_cuda_bytes"],
        "peak_cuda_status": run["peak_cuda_status"],
        "peak_resident_bytes": int(run["peak_resident_bytes"]),
        "peak_resident_status": run["peak_resident_status"],
        "rss_sample_interval_seconds": RSS_SAMPLE_INTERVAL_SECONDS,
        "effective_clustering_epochs": run["effective_clustering_epochs"],
        "effective_epochs_status": run["effective_epochs_status"],
        "psutil_version": _package_version("psutil"),
    }
    pd.DataFrame([timing_row], columns=TIMING_COLUMNS).to_csv(
        paths.full_timing, index=False
    )
    return _validate_full_setting_output(paths)


def _run_smoke(output_root: Path | None, device: str, source_root: Path | None) -> None:
    run_table, _cluster_table = _run_configuration(
        load_dataset(DATASET), _output_paths(output_root), device=device, source_root=source_root
    )
    print(
        f"Smoke run completed: row={run_table.loc[0, 'row']}, "
        f"clusters={int(run_table.loc[0, 'n_clusters'])}"
    )


def _preflight() -> None:
    X = np.asarray([[1.0, 0.0], [0.0, 1.0], [2.0, 2.0]], dtype=np.float64)
    canonical_X, canonical_ids = _canonical_input(X, ["p3", "p1", "p2"])
    _validate_configuration(
        canonical_X,
        n_clusters_init=2,
        dip_merge_threshold=DIP_MERGE_THRESHOLD,
        embedding_size=EMBEDDING_SIZE if EMBEDDING_SIZE <= X.shape[1] else X.shape[1],
        batch_size=BATCH_SIZE,
        pretrain_epochs=PRETRAIN_EPOCHS,
        clustering_epochs=CLUSTERING_EPOCHS,
        seed=SEED,
    )
    if canonical_ids.tolist() != ["p1", "p2", "p3"]:
        raise ValueError("canonical protein-ID ordering failed")
    print("Smoke preflight passed; no DipDECK fit was launched")


def _run_scientific_grid(
    output_root: Path | None,
    device: str,
    source_root: Path | None,
    *,
    dataset: Dataset | None = None,
    splits: tuple[FoldSplit, ...] | None = None,
    configs: tuple[SmokeConfig, ...] | None = None,
    estimator_factory: Callable[[SmokeConfig, str, Path | None], tuple[Any, dict[str, Any]]] = _create_dipdeck,
    dedup: bool = False,
    pca_dimensions: int | None = None,
    embedding: str = "prott5",
    seeds: tuple[int, ...] = SCIENTIFIC_SEEDS,
) -> pd.DataFrame:
    live_dataset = _load_embedding_dataset(embedding) if dataset is None else dataset
    pca_metadata: list[dict[str, Any]] | None = None
    if splits is None:
        if pca_dimensions is None:
            fold_splits = _scientific_fold_splits(live_dataset, dedup=dedup)
        else:
            fold_splits, pca_metadata = _pca_scientific_splits(
                live_dataset, pca_dimensions, dedup=dedup, embedding=embedding
            )
    else:
        fold_splits = splits
    input_space = _input_space(embedding, pca_dimensions)
    grid = _scientific_configs(seeds=seeds) if configs is None else configs
    if not fold_splits or not grid:
        raise ValueError("scientific DipDECK requires at least one fold and one configuration")
    paths = _scientific_paths(output_root, dedup=dedup, input_space=input_space)
    _reset_scientific_outputs(paths)
    runs: list[dict[str, Any]] = []
    clusters: list[dict[str, Any]] = []
    archives: dict[str, dict[str, np.ndarray]] = {}
    logs: dict[str, str] = {}
    try:
        for split in fold_splits:
            for config in grid:
                run, run_clusters, archive, log = _fit_scientific_fold(
                    split,
                    config,
                    device=device,
                    source_root=source_root,
                    estimator_factory=estimator_factory,
                    dedup=dedup,
                    input_space=input_space,
                )
                if run["row"] in archives:
                    raise ValueError("scientific DipDECK grid generated duplicate run IDs")
                runs.append(run)
                clusters.extend(run_clusters)
                archives[run["row"]] = archive
                logs[run["row"]] = log
        return _persist_scientific_outputs(
            paths, runs, clusters, archives, logs, dedup=dedup, pca_metadata=pca_metadata,
            embedding=embedding, seeds=seeds,
        )
    except Exception:
        shutil.rmtree(paths.root, ignore_errors=True)
        raise


def _run_scientific_reference(
    output_root: Path | None,
    device: str,
    source_root: Path | None,
    *,
    dataset: Dataset | None = None,
    configs: tuple[SmokeConfig, ...] | None = None,
    estimator_factory: Callable[[SmokeConfig, str, Path | None], tuple[Any, dict[str, Any]]] = _create_dipdeck,
    dedup: bool = False,
    pca_dimensions: int | None = None,
    embedding: str = "prott5",
    seeds: tuple[int, ...] = SCIENTIFIC_SEEDS,
) -> pd.DataFrame:
    live_dataset = _load_embedding_dataset(embedding) if dataset is None else dataset
    if pca_dimensions is None:
        reference_dataset = _scientific_reference_dataset(live_dataset, dedup=dedup)
        identity = _raw_reference_identity(embedding, dedup=dedup)
    else:
        reference_dataset, pca_metadata = _pca_scientific_reference_dataset(
            live_dataset, pca_dimensions, dedup=dedup, embedding=embedding
        )
        identity = _pca_scientific_reference_identity(
            pca_metadata, dedup=dedup, embedding=embedding
        )
    reference_configs = _scientific_reference_configs(seeds) if configs is None else configs
    if not reference_configs:
        raise ValueError("scientific DipDECK reference requires at least one configuration")
    runs: list[pd.DataFrame] = []
    for config in reference_configs:
        run, _clusters = _run_configuration(
            reference_dataset,
            _scientific_reference_paths(
                output_root, config.seed, dedup=dedup, input_space=identity.input_space
            ),
            config=config,
            identity=identity,
            device=device,
            source_root=source_root,
            estimator_factory=estimator_factory,
            require_live_dataset=False,
        )
        runs.append(run)
    return pd.concat(runs, ignore_index=True)


def _validate_scientific_reference_outputs(
    output_root: Path | None,
    dataset: Dataset | None = None,
    *,
    dedup: bool = False,
    pca_dimensions: int | None = None,
    embedding: str = "prott5",
    seeds: tuple[int, ...] = SCIENTIFIC_SEEDS,
) -> pd.DataFrame:
    live_dataset = _load_embedding_dataset(embedding) if dataset is None else dataset
    reference_dataset = _scientific_reference_dataset(live_dataset, dedup=dedup)
    if pca_dimensions is None:
        identity = _raw_reference_identity(embedding, dedup=dedup)
    else:
        identity = RunIdentity(
            run_role="scientific_reference",
            records="graphpart_retained_dedup" if dedup else "graphpart_retained",
            input_space=_input_space(embedding, pca_dimensions),
            input_pipeline=(
                f"graphpart_retained_{EMBEDDINGS[embedding].raw_space}_then_pca_unwhitened_float32"
            ),
            transform_fit_records=f"graphpart_retained_{len(reference_dataset)}_records",
        )
    runs: list[pd.DataFrame] = []
    recomputed_reference: Dataset | None = None
    for config in _scientific_reference_configs(seeds):
        run, _clusters, _archive = _validate_existing_outputs(
            _scientific_reference_paths(
                output_root, config.seed, dedup=dedup, input_space=identity.input_space
            )
        )
        row = run.iloc[0]
        if (
            row["run_role"] != identity.run_role
            or row["records"] != identity.records
            or int(row["n"]) != len(reference_dataset)
            or row["subset_hash"] != reference_dataset.subset_hash
            or int(row["seed"]) != config.seed
            or int(row["n_clusters_init"]) != config.n_clusters_init
            or float(row["dip_merge_threshold"]) != config.dip_merge_threshold
            or int(row["pretrain_epochs"]) != config.pretrain_epochs
            or int(row["clustering_epochs"]) != config.clustering_epochs
            or row["input_space"] != identity.input_space
            or row["input_pipeline"] != identity.input_pipeline
            or ("" if pd.isna(row["transform_fit_records"]) else row["transform_fit_records"])
            != identity.transform_fit_records
            or (
                pca_dimensions is not None
                and (
                    not re.fullmatch(r"[0-9a-f]{64}", str(row["transform_sha256"]))
                    or not 0 < float(row["variance_retained"]) <= 1
                )
            )
        ):
            raise ValueError("scientific DipDECK reference does not match its declared configuration")
        if row["clustpy_commit"] != CLUSTPY_COMMIT or row["dipdeck_source_sha256"] != DIPDECK_SOURCE_SHA256:
            raise ValueError("scientific DipDECK reference does not record the selected ClustPy source")
        if pca_dimensions is None:
            _canonical_X, reference_ids = _canonical_input(reference_dataset.X, reference_dataset.ids)
            with np.load(_archive, allow_pickle=False) as arrays:
                if arrays["protein_id"].tolist() != reference_ids.tolist():
                    raise ValueError("scientific DipDECK reference proteins differ from the input")
        if pca_dimensions is not None:
            if recomputed_reference is None:
                recomputed_reference = _pca_scientific_reference_dataset(
                    live_dataset, pca_dimensions, dedup=dedup, embedding=embedding
                )[0]
            with np.load(_archive, allow_pickle=False) as arrays:
                if arrays["protein_id"].tolist() != recomputed_reference.ids:
                    raise ValueError("scientific DipDECK PCA reference proteins differ from the input")
                _check_same_pca_coordinates(
                    arrays["input"], recomputed_reference.X, f"reference seed {config.seed}"
                )
        _validate_realised_clustering_epochs(
            row["effective_clustering_epochs"], row["effective_epochs_status"],
            config.clustering_epochs, int(row["n_clusters"]),
        )
        runs.append(run)
    return pd.concat(runs, ignore_index=True)


PCA_VARIANCE_ATOL = 1e-6


def _check_pca_provenance_agrees(fold_lists: list[list[dict[str, Any]] | None]) -> None:
    if all(folds is None for folds in fold_lists):
        return
    if any(folds is None for folds in fold_lists):
        raise ValueError("only some seed halves recorded PCA provenance")
    exact = ("held_out_fold", "input_space", "pca_dimensions", "pca_fit_records", "pca_fit_subset_hash")
    first = {str(fold["held_out_fold"]): fold for fold in fold_lists[0]}
    for folds in fold_lists[1:]:
        other = {str(fold["held_out_fold"]): fold for fold in folds}
        if set(other) != set(first):
            raise ValueError("seed halves recorded PCA for different folds")
        for name, fold in other.items():
            if any(fold[key] != first[name][key] for key in exact) or abs(
                float(fold["variance_retained"]) - float(first[name]["variance_retained"])
            ) > PCA_VARIANCE_ATOL:
                raise ValueError(f"seed halves disagree on the PCA for held-out fold {name}")


def _merge_seed_halves(
    sources: list[Path], output_root: Path | None, *, embedding: str,
    pca_dimensions: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    input_space = _input_space(embedding, pca_dimensions)
    destination = _scientific_paths(output_root, input_space=input_space)
    source_seeds: list[tuple[int, ...]] = []
    manifests = []
    for source in sources:
        paths = _scientific_paths(source, input_space=input_space)
        manifest = json.loads(paths.manifest.read_text(encoding="utf-8"))
        seeds = tuple(int(seed) for seed in manifest["seeds"])
        if manifest.get("embedding") != embedding:
            raise ValueError(f"{source} was fitted on {manifest.get('embedding')}, not {embedding}")
        _validate_scientific_outputs(
            paths, expected_rows=len(SCIENTIFIC_FOLDS) * len(_scientific_configs(seeds=seeds)),
            pca_dimensions=pca_dimensions, embedding=embedding, seeds=seeds,
        )
        _validate_scientific_reference_outputs(
            source, pca_dimensions=pca_dimensions, embedding=embedding, seeds=seeds
        )
        source_seeds.append(seeds)
        manifests.append(manifest)
    all_seeds = [seed for seeds in source_seeds for seed in seeds]
    if len(set(all_seeds)) != len(all_seeds) or tuple(sorted(all_seeds)) != SCIENTIFIC_SEEDS:
        raise ValueError(f"seed halves must be disjoint and cover {SCIENTIFIC_SEEDS}: {source_seeds}")
    _check_pca_provenance_agrees([manifest.get("pca_folds") for manifest in manifests])
    _reset_scientific_outputs(destination)
    try:
        runs, clusters = [], []
        for source in sources:
            paths = _scientific_paths(source, input_space=input_space)
            runs.append(pd.read_csv(paths.runs))
            clusters.append(pd.read_csv(paths.clusters))
            for archive in paths.assignments.glob("*.npz"):
                shutil.copy2(archive, destination.assignments / archive.name)
            for log in paths.logs.glob("*.log"):
                shutil.copy2(log, destination.logs / log.name)
        run_table = pd.concat(runs, ignore_index=True)
        pd.concat(clusters, ignore_index=True).to_csv(destination.clusters, index=False)
        run_table.to_csv(destination.runs, index=False)
        merged = {
            "rows": len(run_table), "embedding": embedding, "seeds": list(SCIENTIFIC_SEEDS),
            "merged_from": [str(source) for source in sources],
            "run_ids": run_table["row"].tolist(),
        }
        if manifests[0].get("pca_folds") is not None:
            merged["pca_folds"] = manifests[0]["pca_folds"]
            merged["pca_folds_by_source"] = [manifest["pca_folds"] for manifest in manifests]
        destination.manifest.write_text(json.dumps(merged, indent=2) + "\n", encoding="utf-8")
        for source, seeds in zip(sources, source_seeds):
            for seed in seeds:
                source_reference = _scientific_reference_paths(
                    source, seed, input_space=input_space
                ).root
                target_reference = _scientific_reference_paths(
                    output_root, seed, input_space=input_space
                ).root
                if target_reference.exists():
                    shutil.rmtree(target_reference)
                shutil.copytree(source_reference, target_reference)
        grid = _validate_scientific_outputs(
            destination, expected_rows=1_000, pca_dimensions=pca_dimensions, embedding=embedding
        )
        references = _validate_scientific_reference_outputs(
            output_root, pca_dimensions=pca_dimensions, embedding=embedding
        )
    except Exception:
        shutil.rmtree(destination.root, ignore_errors=True)
        raise
    return grid, references


def _summarise_scientific_grid(
    runs: pd.DataFrame, assignments: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    settings = ["n_clusters_init", "dip_merge_threshold"]
    scores = ["silhouette", "calinski_harabasz", "davies_bouldin"]
    fold_rows: list[dict[str, Any]] = []
    for (n_init, threshold, fold), group in runs.groupby([*settings, "held_out_fold"]):
        labels = {}
        for row in group.sort_values("seed").to_dict("records"):
            with np.load(assignments / f"{row['row']}.npz", allow_pickle=False) as archive:
                labels[int(row["seed"])] = archive["label"]
        agreement = [
            adjusted_rand_score(labels[first], labels[second])
            for first, second in combinations(sorted(labels), 2)
        ]
        scored = group.loc[group["score_status"] == "defined"]
        fold_rows.append({
            "n_clusters_init": int(n_init),
            "dip_merge_threshold": float(threshold),
            "held_out_fold": str(fold),
            "test_n": int(group["test_n"].iloc[0]),
            "runs": len(group),
            "runs_scored": len(scored),
            "training_clusters_median": float(group["n_training_clusters"].median()),
            "test_clusters_median": float(group["n_test_clusters"].median()),
            "test_clusters_min": int(group["n_test_clusters"].min()),
            "test_clusters_max": int(group["n_test_clusters"].max()),
            **{f"{score}_median": float(scored[score].median()) for score in scores},
            "seed_ari_median": float(np.median(agreement)) if agreement else np.nan,
            "seed_ari_min": float(np.min(agreement)) if agreement else np.nan,
            "seed_pairs": len(agreement),
            "elapsed_seconds_median": float(group["elapsed_seconds"].median()),
        })
    folds = pd.DataFrame(fold_rows)
    setting_rows: list[dict[str, Any]] = []
    for (n_init, threshold), group in folds.groupby(settings):
        row: dict[str, Any] = {
            "n_clusters_init": int(n_init),
            "dip_merge_threshold": float(threshold),
            "folds": len(group),
            "runs": int(group["runs"].sum()),
            "runs_scored": int(group["runs_scored"].sum()),
            "training_clusters_mean": float(group["training_clusters_median"].mean()),
            "test_clusters_mean": float(group["test_clusters_median"].mean()),
        }
        for measure in [*scores, "seed_ari"]:
            row[f"{measure}_mean"] = float(group[f"{measure}_median"].mean())
            row[f"{measure}_sd"] = float(group[f"{measure}_median"].std(ddof=1))
        setting_rows.append(row)
    return folds, pd.DataFrame(setting_rows)


def _reference_seed_agreement(
    output_root: Path | None, *, dedup: bool = False, input_space: str = "raw1024"
) -> pd.DataFrame:
    labels: dict[int, np.ndarray] = {}
    for config in _scientific_reference_configs():
        paths = _scientific_reference_paths(
            output_root, config.seed, dedup=dedup, input_space=input_space
        )
        _run, _clusters, archive = _validate_existing_outputs(paths)
        with np.load(archive, allow_pickle=False) as arrays:
            labels[config.seed] = (arrays["protein_id"], arrays["label"])
    rows = []
    for first, second in combinations(sorted(labels), 2):
        if labels[first][0].tolist() != labels[second][0].tolist():
            raise ValueError("DipDECK reference seeds were not fitted on the same proteins")
        rows.append({
            "seed_a": first, "seed_b": second, "n": len(labels[first][0]),
            "ari": float(adjusted_rand_score(labels[first][1], labels[second][1])),
        })
    return pd.DataFrame(rows)


def _write_scientific_summaries(
    output_root: Path | None, *, dedup: bool = False, pca_dimensions: int | None = None,
    embedding: str = "prott5",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    input_space = _input_space(embedding, pca_dimensions)
    paths = _scientific_paths(output_root, dedup=dedup, input_space=input_space)
    runs = _validate_scientific_outputs(
        paths, expected_rows=1_000, dedup=dedup, pca_dimensions=pca_dimensions,
        embedding=embedding,
    )
    _validate_scientific_reference_outputs(
        output_root, dedup=dedup, pca_dimensions=pca_dimensions, embedding=embedding
    )
    folds, settings = _summarise_scientific_grid(runs, paths.assignments)
    folds.to_csv(paths.runs.parent / "dipdeck_scientific_fold_summary.csv", index=False)
    settings.to_csv(paths.runs.parent / "dipdeck_scientific_setting_summary.csv", index=False)
    reference_root = _scientific_reference_paths(
        output_root, 0, dedup=dedup, input_space=input_space
    ).root.parent
    _reference_seed_agreement(output_root, dedup=dedup, input_space=input_space).to_csv(
        reference_root / "dipdeck_scientific_reference_seed_ari.csv",
        index=False,
    )
    return folds, settings


def _deduplicated_rescore(
    runs: pd.DataFrame, assignments: Path, dataset: Dataset, splits: tuple[FoldSplit, ...]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    survivors = set(deduplicate(dataset).ids)
    tests = {
        split.held_out_fold: _canonical_input(split.test.X, split.test.ids)
        for split in splits
    }
    scores = ["silhouette", "calinski_harabasz", "davies_bouldin"]
    rows: list[dict[str, Any]] = []
    for run in runs.to_dict("records"):
        test_X, test_ids = tests[str(run["held_out_fold"])]
        with np.load(assignments / f"{run['row']}.npz", allow_pickle=False) as archive:
            if archive["protein_id"].tolist() != test_ids.tolist():
                raise ValueError("scientific DipDECK archive IDs do not match held-out input")
            labels = archive["label"]
        keep = np.fromiter((protein in survivors for protein in test_ids), dtype=bool)
        full = _held_out_internal_scores(test_X, labels)
        collapsed = _held_out_internal_scores(test_X[keep], labels[keep])
        row: dict[str, Any] = {
            "row": run["row"], "n_clusters_init": int(run["n_clusters_init"]),
            "dip_merge_threshold": float(run["dip_merge_threshold"]),
            "held_out_fold": str(run["held_out_fold"]), "seed": int(run["seed"]),
            "test_n": len(test_ids), "dedup_n": int(keep.sum()),
            "test_clusters": len(np.unique(labels)),
            "dedup_clusters": len(np.unique(labels[keep])),
            "score_status": full["score_status"], "dedup_score_status": collapsed["score_status"],
        }
        for score in scores:
            row[score] = np.nan if full[score] == "" else float(full[score])
            row[f"{score}_dedup"] = np.nan if collapsed[score] == "" else float(collapsed[score])
        rows.append(row)
    per_run = pd.DataFrame(rows)
    fold_rows = []
    for fold, group in per_run.groupby("held_out_fold"):
        scored = group.dropna(subset=[f"{score}_dedup" for score in scores] + scores)
        fold_row: dict[str, Any] = {
            "held_out_fold": fold, "test_n": int(group["test_n"].iloc[0]),
            "dedup_n": int(group["dedup_n"].iloc[0]), "runs_scored": len(scored),
            "runs_losing_a_cluster": int((group["dedup_clusters"] < group["test_clusters"]).sum()),
        }
        for score in scores:
            change = scored[f"{score}_dedup"] - scored[score]
            fold_row[f"{score}_change_median"] = float(change.median())
            fold_row[f"{score}_change_min"] = float(change.min())
            fold_row[f"{score}_change_max"] = float(change.max())
        fold_rows.append(fold_row)
    return per_run, pd.DataFrame(fold_rows)


def _write_deduplicated_rescore(output_root: Path | None) -> pd.DataFrame:
    paths = _scientific_paths(output_root)
    dataset = load_dataset(DATASET)
    runs = _validate_scientific_outputs(paths, expected_rows=1_000, dataset=dataset)
    per_run, folds = _deduplicated_rescore(
        runs, paths.assignments, dataset, _scientific_fold_splits(dataset)
    )
    per_run.to_csv(paths.runs.parent / "dipdeck_scientific_dedup_rescore.csv", index=False)
    folds.to_csv(paths.runs.parent / "dipdeck_scientific_dedup_rescore_folds.csv", index=False)
    return folds


def _compare_record_sets(
    rescore: pd.DataFrame,
    full_assignments: Path,
    dedup_runs: pd.DataFrame,
    dedup_assignments: Path,
) -> pd.DataFrame:
    key = ["n_clusters_init", "dip_merge_threshold", "held_out_fold", "seed"]
    refits = dedup_runs.assign(held_out_fold=dedup_runs["held_out_fold"].astype(str))
    paired = rescore.assign(held_out_fold=rescore["held_out_fold"].astype(str)).merge(
        refits[[*key, "row", "test_n", "n_test_clusters", "score_status",
                "silhouette", "calinski_harabasz", "davies_bouldin"]],
        on=key, suffixes=("_full", "_refit"), validate="one_to_one",
    )
    if len(paired) != len(rescore) or len(paired) != len(dedup_runs):
        raise ValueError("full and deduplicated DipDECK grids do not pair one to one")
    if not (paired["dedup_n"] == paired["test_n_refit"]).all():
        raise ValueError("deduplicated refits were not scored on the deduplicated held-out records")
    agreement = []
    for row in paired.to_dict("records"):
        with (
            np.load(full_assignments / f"{row['row_full']}.npz", allow_pickle=False) as full,
            np.load(dedup_assignments / f"{row['row_refit']}.npz", allow_pickle=False) as refit,
        ):
            position = {protein: index for index, protein in enumerate(full["protein_id"])}
            shared = [position[protein] for protein in refit["protein_id"]]
            agreement.append(float(adjusted_rand_score(full["label"][shared], refit["label"])))
    scores = ["silhouette", "calinski_harabasz", "davies_bouldin"]
    return pd.DataFrame({
        **{column: paired[column] for column in key},
        "row_full": paired["row_full"], "row_refit": paired["row_refit"],
        "n": paired["dedup_n"],
        "full_test_clusters": paired["dedup_clusters"],
        "refit_test_clusters": paired["n_test_clusters"],
        "full_score_status": paired["dedup_score_status"],
        "refit_score_status": paired["score_status_refit"],
        **{f"{score}_full": paired[f"{score}_dedup"] for score in scores},
        **{f"{score}_refit": paired[f"{score}_refit"] for score in scores},
        "ari_full_vs_refit": agreement,
    })


def _compare_reference_record_sets(output_root: Path | None) -> pd.DataFrame:
    rows = []
    for config in _scientific_reference_configs():
        labels = []
        for dedup in (False, True):
            paths = _scientific_reference_paths(output_root, config.seed, dedup=dedup)
            _run, _clusters, archive = _validate_existing_outputs(paths)
            with np.load(archive, allow_pickle=False) as arrays:
                labels.append((arrays["protein_id"], arrays["label"]))
        (full_ids, full_labels), (dedup_ids, dedup_labels) = labels
        position = {protein: index for index, protein in enumerate(full_ids)}
        shared = [position[protein] for protein in dedup_ids]
        rows.append({
            "seed": config.seed, "n": len(dedup_ids),
            "ari_full_vs_refit": float(adjusted_rand_score(full_labels[shared], dedup_labels)),
        })
    return pd.DataFrame(rows)


def _write_record_set_comparison(output_root: Path | None) -> pd.DataFrame:
    dataset = load_dataset(DATASET)
    full_paths = _scientific_paths(output_root)
    dedup_paths = _scientific_paths(output_root, dedup=True)
    full_runs = _validate_scientific_outputs(full_paths, expected_rows=1_000, dataset=dataset)
    dedup_runs = _validate_scientific_outputs(
        dedup_paths, expected_rows=1_000, dataset=dataset, dedup=True
    )
    _validate_scientific_reference_outputs(output_root, dataset)
    _validate_scientific_reference_outputs(output_root, dataset, dedup=True)
    rescore, _folds = _deduplicated_rescore(
        full_runs, full_paths.assignments, dataset, _scientific_fold_splits(dataset)
    )
    paired = _compare_record_sets(rescore, full_paths.assignments, dedup_runs, dedup_paths.assignments)
    paired.to_csv(dedup_paths.runs.parent / "dipdeck_scientific_full_vs_dedup.csv", index=False)
    result_root = output_root if output_root is not None else REPO / "results" / "dipdeck"
    _compare_reference_record_sets(output_root).to_csv(
        result_root / "scientific_reference_dedup" / "dipdeck_scientific_reference_full_vs_dedup_ari.csv",
        index=False,
    )
    return paired


COMBINED_RUN_COLUMNS = (
    "row", "record_set", "run_role", "held_out_fold", "seed", "n_clusters_init",
    "dip_merge_threshold", "fit_n", "partition_n", "partition_subset_hash", "n_training_clusters",
    "n_partition_clusters", "score_status", "silhouette", "calinski_harabasz", "davies_bouldin",
    "warning_count", "merge_count", "elapsed_seconds", "git_sha", "dipdeck_source_sha256",
    "assignment",
)
COMBINED_CLUSTER_COLUMNS = (
    "row", "record_set", "run_role", "held_out_fold", "cluster", "size", "proportion",
    "representative_protein_id", "representative_match_count",
    "representative_matching_label_count",
)
ELIGIBLE_PARTITION_COLUMNS = (
    "method", "row", "record_set", "seed", "partition_n", "partition_subset_hash",
    "n_partition_clusters", "assignment", "basis",
)


REFERENCE_SCORE_COLUMNS = (
    "row", "record_set", "seed", "n", "subset_hash", "n_clusters", "scoring_space",
    "scoring_distance", "score_status", "silhouette", "calinski_harabasz", "davies_bouldin",
)


def _reference_scores_path(output_root: Path | None) -> Path:
    result_root = output_root if output_root is not None else REPO / "results" / "dipdeck"
    return result_root / "metrics" / "dipdeck_reference_scores.csv"


def _score_scientific_reference(
    output_root: Path | None, dataset: Dataset | None = None
) -> pd.DataFrame:
    live_dataset = load_dataset(DATASET) if dataset is None else dataset
    rows: list[dict[str, Any]] = []
    for dedup in (False, True):
        reference_dataset = _scientific_reference_dataset(live_dataset, dedup=dedup)
        X, protein_ids = _canonical_input(reference_dataset.X, reference_dataset.ids)
        for config in _scientific_reference_configs():
            paths = _scientific_reference_paths(output_root, config.seed, dedup=dedup)
            run = pd.read_csv(paths.runs)
            if len(run) != 1:
                raise ValueError(f"{paths.runs} holds {len(run)} rows, expected 1")
            row = run.iloc[0]
            if row["subset_hash"] != reference_dataset.subset_hash:
                raise ValueError(f"{row['row']}: subset hash differs from the rebuilt reference set")
            with np.load(paths.assignments / f"{row['row']}.npz", allow_pickle=False) as arrays:
                archived_ids, labels = arrays["protein_id"], arrays["label"]
            if archived_ids.astype(str).tolist() != protein_ids.tolist():
                raise ValueError(f"{row['row']}: partition proteins or order differ from the input")
            if len(np.unique(labels)) != int(row["n_clusters"]):
                raise ValueError(f"{row['row']}: partition cluster count differs from the run table")
            rows.append({
                "row": row["row"],
                "record_set": "dedup" if dedup else "full",
                "seed": config.seed,
                "n": len(labels),
                "subset_hash": row["subset_hash"],
                "n_clusters": int(row["n_clusters"]),
                "scoring_space": INPUT_SPACE,
                "scoring_distance": "euclidean",
                **_held_out_internal_scores(X, labels),
            })
    return pd.DataFrame(rows)[list(REFERENCE_SCORE_COLUMNS)]


def _score_pca_references(output_root: Path | None, dataset: Dataset | None = None) -> pd.DataFrame:
    live_dataset = load_dataset(DATASET) if dataset is None else dataset
    rows: list[dict[str, Any]] = []
    for dimensions, dedup in product(SCIENTIFIC_PCA_DIMS, (False, True)):
        input_space = f"pca{dimensions}"
        _validate_scientific_reference_outputs(
            output_root, live_dataset, dedup=dedup, pca_dimensions=dimensions
        )
        for config in _scientific_reference_configs():
            paths = _scientific_reference_paths(
                output_root, config.seed, dedup=dedup, input_space=input_space
            )
            row = pd.read_csv(paths.runs).iloc[0]
            with np.load(paths.assignments / f"{row['row']}.npz", allow_pickle=False) as arrays:
                X, labels = arrays["input"], arrays["label"]
            rows.append({
                "row": row["row"],
                "record_set": "dedup" if dedup else "full",
                "seed": config.seed,
                "n": len(labels),
                "subset_hash": row["subset_hash"],
                "n_clusters": int(row["n_clusters"]),
                "scoring_space": input_space,
                "scoring_distance": "euclidean",
                **_held_out_internal_scores(X, labels),
            })
    return pd.DataFrame(rows)[list(REFERENCE_SCORE_COLUMNS)]


def _repo_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _combined_scientific_tables(output_root: Path | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    expected = set(product(
        SCIENTIFIC_FOLDS, SCIENTIFIC_INITIAL_CLUSTERS, SCIENTIFIC_MERGE_THRESHOLDS,
        SCIENTIFIC_SEEDS,
    ))
    reference_configs = _scientific_reference_configs()
    runs: list[pd.DataFrame] = []
    clusters: list[pd.DataFrame] = []
    for dedup in (False, True):
        record_set = "dedup" if dedup else "full"
        paths = _scientific_paths(output_root, dedup=dedup)
        grid = pd.read_csv(paths.runs, dtype={"held_out_fold": str})
        if len(grid) != len(expected):
            raise ValueError(f"{record_set} grid has {len(grid)} rows, expected {len(expected)}")
        observed = set(zip(
            grid["held_out_fold"], grid["n_clusters_init"].astype(int),
            grid["dip_merge_threshold"].round(2), grid["seed"].astype(int),
        ))
        if observed != expected:
            raise ValueError(f"{record_set} grid does not cover the recorded fold grid exactly")
        runs.append(pd.DataFrame({
            "row": grid["row"],
            "record_set": record_set,
            "run_role": grid["run_role"],
            "held_out_fold": grid["held_out_fold"],
            "seed": grid["seed"].astype(int),
            "n_clusters_init": grid["n_clusters_init"].astype(int),
            "dip_merge_threshold": grid["dip_merge_threshold"],
            "fit_n": grid["train_n"],
            "partition_n": grid["test_n"],
            "partition_subset_hash": grid["test_subset_hash"],
            "n_training_clusters": grid["n_training_clusters"],
            "n_partition_clusters": grid["n_test_clusters"],
            "score_status": grid["score_status"],
            "silhouette": grid["silhouette"],
            "calinski_harabasz": grid["calinski_harabasz"],
            "davies_bouldin": grid["davies_bouldin"],
            "warning_count": grid["warning_count"],
            "merge_count": grid["merge_count"],
            "elapsed_seconds": grid["elapsed_seconds"],
            "git_sha": grid["git_sha"],
            "dipdeck_source_sha256": grid["dipdeck_source_sha256"],
            "assignment": [_repo_relative(paths.assignments / f"{row}.npz") for row in grid["row"]],
        }))
        grid_clusters = pd.read_csv(paths.clusters, dtype={"held_out_fold": str})
        grid_clusters.insert(1, "record_set", record_set)
        grid_clusters.insert(2, "run_role", grid["run_role"].iloc[0])
        clusters.append(grid_clusters)
        for config in reference_configs:
            seed_paths = _scientific_reference_paths(output_root, config.seed, dedup=dedup)
            reference = pd.read_csv(seed_paths.runs)
            if len(reference) != 1:
                raise ValueError(f"{seed_paths.runs} holds {len(reference)} rows, expected 1")
            row = reference.iloc[0]
            runs.append(pd.DataFrame([{
                "row": row["row"],
                "record_set": record_set,
                "run_role": row["run_role"],
                "held_out_fold": "",
                "seed": int(row["seed"]),
                "n_clusters_init": int(row["n_clusters_init"]),
                "dip_merge_threshold": row["dip_merge_threshold"],
                "fit_n": int(row["n"]),
                "partition_n": int(row["n"]),
                "partition_subset_hash": row["subset_hash"],
                "n_training_clusters": int(row["n_clusters"]),
                "n_partition_clusters": int(row["n_clusters"]),
                "score_status": "not_computed",
                "silhouette": np.nan,
                "calinski_harabasz": np.nan,
                "davies_bouldin": np.nan,
                "warning_count": row["warning_count"],
                "merge_count": row["merge_count"],
                "elapsed_seconds": row["elapsed_seconds"],
                "git_sha": row["git_sha"],
                "dipdeck_source_sha256": row["dipdeck_source_sha256"],
                "assignment": _repo_relative(seed_paths.assignments / f"{row['row']}.npz"),
            }]))
            reference_clusters = pd.read_csv(seed_paths.clusters)
            reference_clusters.insert(1, "record_set", record_set)
            reference_clusters.insert(2, "run_role", row["run_role"])
            reference_clusters.insert(3, "held_out_fold", "")
            clusters.append(reference_clusters)
    combined = pd.concat(runs, ignore_index=True)[list(COMBINED_RUN_COLUMNS)]
    scores_path = _reference_scores_path(output_root)
    if scores_path.is_file():
        scores = pd.read_csv(scores_path).set_index("row")
        reference_ids = combined.loc[combined["run_role"] == "scientific_reference", "row"]
        if sorted(scores.index) != sorted(reference_ids):
            raise ValueError("reference scores do not cover exactly the 20 reference fits")
        mask = combined["run_role"] == "scientific_reference"
        for field in ("score_status", "silhouette", "calinski_harabasz", "davies_bouldin"):
            combined.loc[mask, field] = combined.loc[mask, "row"].map(scores[field]).to_numpy()
    combined_clusters = pd.concat(clusters, ignore_index=True).reindex(
        columns=list(COMBINED_CLUSTER_COLUMNS)
    )
    reference_rows = combined["run_role"] == "scientific_reference"
    if int((~reference_rows).sum()) != 2 * len(expected) or int(reference_rows.sum()) != 2 * len(
        reference_configs
    ):
        raise ValueError("combined table does not hold both record sets' grid and reference rows")
    if combined["row"].duplicated().any():
        raise ValueError("a row ID appears in more than one scientific tree")
    for record in combined[
        ["row", "partition_n", "partition_subset_hash", "assignment"]
    ].to_dict("records"):
        archive_path = REPO / record["assignment"]
        if not archive_path.is_file():
            archive_path = Path(record["assignment"])
        with np.load(archive_path, allow_pickle=False) as arrays:
            protein_ids, labels = arrays["protein_id"], arrays["label"]
        if len(protein_ids) != record["partition_n"] or len(labels) != record["partition_n"]:
            raise ValueError(f"{record['row']}: NPZ length does not match partition_n")
        if len(set(protein_ids.tolist())) != len(protein_ids):
            raise ValueError(f"{record['row']}: NPZ repeats a protein ID")
        members = Dataset(
            name="", ids=protein_ids.tolist(), X=np.empty((len(protein_ids), 0)), ann=pd.DataFrame()
        )
        if members.subset_hash != record["partition_subset_hash"]:
            raise ValueError(f"{record['row']}: NPZ proteins do not match partition_subset_hash")
    cluster_sizes = combined_clusters.groupby("row")["size"].sum()
    if not cluster_sizes.reindex(combined["row"]).eq(combined["partition_n"].to_numpy()).all():
        raise ValueError("cluster sizes do not sum to partition_n for every run")
    return combined, combined_clusters


def _eligible_partitions(combined: pd.DataFrame) -> pd.DataFrame:
    reference = combined.loc[combined["run_role"] == "scientific_reference"]
    return pd.DataFrame({
        "method": "dipdeck",
        "row": reference["row"],
        "record_set": reference["record_set"],
        "seed": reference["seed"],
        "partition_n": reference["partition_n"],
        "partition_subset_hash": reference["partition_subset_hash"],
        "n_partition_clusters": reference["n_partition_clusters"],
        "assignment": reference["assignment"],
        "basis": "reference fit on all retained records",
    })[list(ELIGIBLE_PARTITION_COLUMNS)].reset_index(drop=True)


def _pca_reference_eligible_partitions(output_root: Path | None) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for dimensions, dedup in product(SCIENTIFIC_PCA_DIMS, (False, True)):
        runs = _validate_scientific_reference_outputs(
            output_root, dedup=dedup, pca_dimensions=dimensions
        )
        for config in _scientific_reference_configs():
            paths = _scientific_reference_paths(
                output_root,
                config.seed,
                dedup=dedup,
                input_space=f"pca{dimensions}",
            )
            run = runs.loc[runs["seed"] == config.seed].iloc[0]
            rows.append(
                {
                    "method": "dipdeck",
                    "row": run["row"],
                    "record_set": "dedup" if dedup else "full",
                    "seed": int(run["seed"]),
                    "partition_n": int(run["n"]),
                    "partition_subset_hash": run["subset_hash"],
                    "n_partition_clusters": int(run["n_clusters"]),
                    "assignment": _repo_relative(paths.assignments / f"{run['row']}.npz"),
                    "basis": (
                        f"PCA {dimensions} reference fit on all retained "
                        f"{'deduplicated' if dedup else 'full'} records"
                    ),
                }
            )
    return pd.DataFrame(rows, columns=ELIGIBLE_PARTITION_COLUMNS)


PCA_BASIS_CHECK_COLUMNS = (
    "record_set", "pca_dimensions", "held_out_fold", "train_n", "all_records_n",
    "mean_squared_cosine", "min_squared_cosine", "variance_captured_ratio",
    "train_variance_retained", "all_records_variance_retained",
)


def _pca_basis_check(dataset: Dataset | None = None) -> pd.DataFrame:
    live_dataset = load_dataset(DATASET) if dataset is None else dataset
    all_X, _all_ids = _canonical_input(live_dataset.X, live_dataset.ids)
    centred = np.asarray(all_X, dtype=np.float64) - np.asarray(all_X, dtype=np.float64).mean(axis=0)
    rows: list[dict[str, Any]] = []
    for dimensions in SCIENTIFIC_PCA_DIMS:
        everyone = PCA(n_components=dimensions, svd_solver="full", whiten=False).fit(all_X)
        everyone_variance = float(((centred @ everyone.components_.T) ** 2).sum())
        for dedup in (False, True):
            for split in _scientific_fold_splits(live_dataset, dedup=dedup):
                train_X, _train_ids = _canonical_input(split.train.X, split.train.ids)
                train = PCA(n_components=dimensions, svd_solver="full", whiten=False).fit(train_X)
                cosines = np.cos(subspace_angles(everyone.components_.T, train.components_.T)) ** 2
                rows.append({
                    "record_set": "dedup" if dedup else "full",
                    "pca_dimensions": dimensions,
                    "held_out_fold": split.held_out_fold,
                    "train_n": len(train_X),
                    "all_records_n": len(all_X),
                    "mean_squared_cosine": float(cosines.mean()),
                    "min_squared_cosine": float(cosines.min()),
                    "variance_captured_ratio": float(
                        ((centred @ train.components_.T) ** 2).sum() / everyone_variance
                    ),
                    "train_variance_retained": float(train.explained_variance_ratio_.sum()),
                    "all_records_variance_retained": float(everyone.explained_variance_ratio_.sum()),
                })
    return pd.DataFrame(rows)[list(PCA_BASIS_CHECK_COLUMNS)]


def _write_pca_reference_eligible_partitions(output_root: Path | None) -> pd.DataFrame:
    eligible = _pca_reference_eligible_partitions(output_root)
    result_root = output_root if output_root is not None else REPO / "results" / "dipdeck"
    metrics = result_root / "metrics"
    metrics.mkdir(parents=True, exist_ok=True)
    eligible.to_csv(metrics / "dipdeck_pca_reference_eligible_partitions.csv", index=False)
    _score_pca_references(output_root).to_csv(
        metrics / "dipdeck_pca_reference_scores.csv", index=False
    )
    return eligible


def _write_combined_scientific_tables(
    output_root: Path | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    combined, combined_clusters = _combined_scientific_tables(output_root)
    eligible = _eligible_partitions(combined)
    result_root = output_root if output_root is not None else REPO / "results" / "dipdeck"
    metrics = result_root / "metrics"
    metrics.mkdir(parents=True, exist_ok=True)
    combined.to_csv(metrics / "dipdeck_all_runs.csv", index=False)
    combined_clusters.to_csv(metrics / "dipdeck_all_clusters.csv", index=False)
    eligible.to_csv(metrics / "dipdeck_eligible_partitions.csv", index=False)
    return combined, combined_clusters, eligible


EC_SHUFFLE_SEEDS = tuple(range(5))
EC_BASELINE_MEASURES = ("ari", "nmi", "ami")
EC_ARMS = {
    "prott5": ((False, None), (True, None),
               *((dedup, d) for d in SCIENTIFIC_PCA_DIMS for dedup in (False, True))),
    "esm2_3b": ((False, None), (False, 45)),
}


def _ec_rows(protein_ids, labels, base: dict[str, Any], table: pd.DataFrame) -> list[dict[str, Any]]:
    from sklearn.metrics import (
        adjusted_mutual_info_score, adjusted_rand_score, normalized_mutual_info_score,
    )

    shuffle_score = {
        "ari": adjusted_rand_score, "nmi": normalized_mutual_info_score,
        "ami": adjusted_mutual_info_score,
    }
    ids = np.asarray(protein_ids, dtype=str)
    predicted = np.asarray(labels)
    rows = []
    for level in EC_LEVELS:
        row = {**base, **external_scores(ids, predicted, level, table=table)}
        labelled = table.loc[ids, f"ec{level}_n"].to_numpy() == 1
        classes = table.loc[ids[labelled], f"ec{level}"].to_numpy(dtype=str)
        clusters = predicted[labelled]
        for measure in EC_BASELINE_MEASURES:
            if row["score_status"] != "scored":
                row[f"{measure}_shuffled_mean"] = np.nan
                continue
            values = [
                shuffle_score[measure](
                    np.random.default_rng(seed).permutation(classes), clusters
                )
                for seed in EC_SHUFFLE_SEEDS
            ]
            row[f"{measure}_shuffled_mean"] = float(np.mean(values))
        rows.append(row)
    return rows


def _single_ec3(table: pd.DataFrame, protein_ids) -> pd.Series:
    members = table.loc[list(protein_ids), ["ec3", "ec3_n"]]
    return members.loc[members["ec3_n"] == 1, "ec3"]


def _majority_share(classes: pd.Series) -> float:
    return float(classes.value_counts(normalize=True).iloc[0]) if len(classes) else np.nan


def _hard_test_ids(split: FoldSplit, table: pd.DataFrame) -> set[str]:
    single = table["ec3_n"] == 1
    train_classes = set(table.loc[[i for i in split.train.ids if single.get(i, False)], "ec3"])
    return {
        protein for protein in split.test.ids
        if single.get(protein, False) and table.at[protein, "ec3"] not in train_classes
    }


def _score_ec_arm(
    output_root: Path | None, *, dedup: bool, pca_dimensions: int | None,
    table: pd.DataFrame, dataset: Dataset, validate: bool = True, embedding: str = "prott5",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    input_space = _input_space(embedding, pca_dimensions)
    record_set = "dedup" if dedup else "full"
    paths = _scientific_paths(output_root, dedup=dedup, input_space=input_space)
    if validate:
        runs = _validate_scientific_outputs(
            paths, expected_rows=1_000, dataset=dataset, dedup=dedup,
            pca_dimensions=pca_dimensions, embedding=embedding,
        )
        _validate_scientific_reference_outputs(
            output_root, dataset, dedup=dedup, pca_dimensions=pca_dimensions, embedding=embedding
        )
    else:
        runs = pd.read_csv(paths.runs)
    hard = (
        {split.held_out_fold: _hard_test_ids(split, table) for split in _scientific_fold_splits(dataset)}
        if not dedup and pca_dimensions is None else {}
    )
    scores: list[dict[str, Any]] = []
    landing: list[dict[str, Any]] = []
    for run in runs.to_dict("records"):
        with np.load(paths.assignments / f"{run['row']}.npz", allow_pickle=False) as archive:
            ids, labels = archive["protein_id"].astype(str), archive["label"]
            train_ids = archive["training_protein_id"].astype(str)
            train_labels = archive["training_label"]
        if len(ids) != int(run["test_n"]):
            raise ValueError(f"{run['row']}: {len(ids)} assigned proteins, {run['test_n']} recorded")
        fold = str(run["held_out_fold"])
        base = {
            "embedding": embedding, "input_space": input_space, "record_set": record_set,
            "run_role": "fold",
            "row": run["row"], "held_out_fold": fold, "seed": int(run["seed"]),
            "n_clusters_init": int(run["n_clusters_init"]),
            "dip_merge_threshold": float(run["dip_merge_threshold"]),
        }
        scores.extend(_ec_rows(ids, labels, base, table))
        for protein in sorted(hard.get(fold, ())):
            position = int(np.flatnonzero(ids == protein)[0])
            cluster = labels[position]
            own = table.at[protein, "ec3"]
            test_classes = _single_ec3(table, ids[labels == cluster])
            others = test_classes.drop(index=protein, errors="ignore")
            train_classes = _single_ec3(table, train_ids[train_labels == cluster])
            landing.append({
                "embedding": embedding, "row": run["row"], "held_out_fold": fold,
                "seed": int(run["seed"]), "n_clusters_init": int(run["n_clusters_init"]),
                "dip_merge_threshold": float(run["dip_merge_threshold"]),
                "protein_id": protein, "ec3": own, "cluster": int(cluster),
                "test_cluster_size": int((labels == cluster).sum()),
                "test_cluster_labelled": int(len(test_classes)),
                "test_cluster_majority_ec3": test_classes.mode().iloc[0] if len(test_classes) else "",
                "test_cluster_majority_share": _majority_share(test_classes),
                "test_share_of_others_with_same_ec3": float((others == own).mean())
                if len(others) else np.nan,
                "train_cluster_size": int((train_labels == cluster).sum()),
                "train_cluster_labelled": int(len(train_classes)),
                "train_cluster_majority_ec3": train_classes.mode().iloc[0] if len(train_classes) else "",
                "train_cluster_majority_share": _majority_share(train_classes),
            })
    for config in _scientific_reference_configs():
        reference_paths = _scientific_reference_paths(
            output_root, config.seed, dedup=dedup, input_space=input_space
        )
        run = pd.read_csv(reference_paths.runs).iloc[0]
        with np.load(reference_paths.assignments / f"{run['row']}.npz", allow_pickle=False) as archive:
            ids, labels = archive["protein_id"].astype(str), archive["label"]
        if len(ids) != int(run["n"]):
            raise ValueError(f"{run['row']}: {len(ids)} assigned proteins, {run['n']} recorded")
        base = {
            "embedding": embedding, "input_space": input_space, "record_set": record_set,
            "run_role": "reference",
            "row": run["row"], "held_out_fold": "", "seed": config.seed,
            "n_clusters_init": config.n_clusters_init,
            "dip_merge_threshold": config.dip_merge_threshold,
        }
        scores.extend(_ec_rows(ids, labels, base, table))
    return scores, landing


def _summarise_ec(scores: pd.DataFrame) -> pd.DataFrame:
    grid = scores[scores["run_role"] == "fold"]
    keys = ["embedding", "input_space", "record_set", "n_clusters_init", "dip_merge_threshold", "level"]
    rows = []
    for key, group in grid.groupby(keys, sort=True):
        for measure in (*EC_BASELINE_MEASURES, "homogeneity", "completeness",
                        "n_clusters", "largest_cluster_share"):
            row = dict(zip(keys, key), measure=measure)
            per_fold = group.groupby("held_out_fold")[measure].mean()
            for fold in SCIENTIFIC_FOLDS:
                row[f"fold{fold}"] = float(per_fold.get(fold, np.nan))
            folds = np.array([row[f"fold{fold}"] for fold in SCIENTIFIC_FOLDS], dtype=float)
            scored = folds[~np.isnan(folds)]
            row["n_folds_scored"] = int(len(scored))
            row["mean_of_folds"] = float(scored.mean()) if len(scored) else np.nan
            row["sd_of_folds"] = float(scored.std(ddof=1)) if len(scored) > 1 else np.nan
            row["shuffled_mean"] = (
                float(group[f"{measure}_shuffled_mean"].mean())
                if f"{measure}_shuffled_mean" in group else np.nan
            )
            row["n_unscored"] = int((group["score_status"] != "scored").sum())
            rows.append(row)
    return pd.DataFrame(rows)


def _write_ec_scores(
    output_root: Path | None, *, validate: bool = True, embedding: str = "prott5",
) -> dict[str, pd.DataFrame]:
    table = ec_table()
    dataset = _load_embedding_dataset(embedding)
    scores: list[dict[str, Any]] = []
    landing: list[dict[str, Any]] = []
    for dedup, pca_dimensions in EC_ARMS[embedding]:
        arm_scores, arm_landing = _score_ec_arm(
            output_root, dedup=dedup, pca_dimensions=pca_dimensions, table=table,
            dataset=dataset, validate=validate, embedding=embedding,
        )
        scores.extend(arm_scores)
        landing.extend(arm_landing)
    score_table = pd.DataFrame(scores)
    suffix = "" if embedding == "prott5" else f"_{embedding}"
    tables = {
        f"dipdeck_ec_scores{suffix}.csv": score_table,
        f"dipdeck_ec_setting_summary{suffix}.csv": _summarise_ec(score_table),
        f"dipdeck_ec_hard_test{suffix}.csv": pd.DataFrame(landing),
    }
    metrics = (output_root or REPO / "results" / "dipdeck") / "metrics"
    metrics.mkdir(parents=True, exist_ok=True)
    for name, frame in tables.items():
        frame.to_csv(metrics / name, index=False)
    return tables


def _medoid_reference_seed(output_root: Path | None, *, input_space: str = "raw1024") -> tuple[int, pd.DataFrame]:
    reference_root = _scientific_reference_paths(output_root, 0, input_space=input_space).root.parent
    pairs = pd.read_csv(reference_root / "dipdeck_scientific_reference_seed_ari.csv")
    rows = []
    for seed in SCIENTIFIC_SEEDS:
        values = pairs.loc[(pairs["seed_a"] == seed) | (pairs["seed_b"] == seed), "ari"]
        if len(values) != len(SCIENTIFIC_SEEDS) - 1:
            raise ValueError(f"seed {seed} has {len(values)} ARI pairs, not {len(SCIENTIFIC_SEEDS) - 1}")
        rows.append({"seed": seed, "mean_ari_to_others": float(values.mean()),
                     "median_ari_to_others": float(values.median())})
    agreement = pd.DataFrame(rows)
    best = agreement.sort_values(["mean_ari_to_others", "seed"], ascending=[False, True]).iloc[0]
    return int(best["seed"]), agreement


def _protein_family_labels() -> pd.DataFrame:
    ann = read_bundle_part(DATASETS[DATASET][1], 0).set_index("protein_id")
    family = ann["protein_families"].fillna("").str.split("|").str[0].str.strip()
    truncated = family.str.startswith("In the ")
    return pd.DataFrame({
        "protein_family": family.where(~truncated, ""),
        "family_truncated": truncated,
        "superfamily": ann["superfamily"].fillna("").str.strip(),
    })


def _ec3_contingency(
    ids: np.ndarray, labels: np.ndarray, table: pd.DataFrame,
    families: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    members = table.loc[ids, ["ec3", "ec3_n", "ec4", "ec4_n"]].copy()
    members["cluster"] = labels
    members["ec3_column"] = np.where(
        members["ec3_n"] == 1, members["ec3"],
        np.where(members["ec3_n"] == 0, "no_class", "several_classes"),
    )
    counts = pd.crosstab(members["cluster"], members["ec3_column"])
    single = members[members["ec3_n"] == 1]
    clusters = []
    for cluster, group in members.groupby("cluster"):
        labelled = group.loc[group["ec3_n"] == 1, "ec3"]
        ec4 = group.loc[group["ec4_n"] == 1, "ec4"]
        clusters.append({
            "cluster": int(cluster), "size": len(group), "ec3_labelled": len(labelled),
            "ec3_classes_present": int(labelled.nunique()),
            "majority_ec3": labelled.mode().iloc[0] if len(labelled) else "",
            "ec3_purity": _majority_share(labelled),
            "ec4_labelled": len(ec4),
            "majority_ec4": ec4.mode().iloc[0] if len(ec4) else "",
            "ec4_purity": _majority_share(ec4),
        })
        if families is not None:
            text = families.loc[group.index]
            for column in ("protein_family", "superfamily"):
                named = text.loc[text[column] != "", column]
                clusters[-1][f"majority_{column}"] = named.mode().iloc[0] if len(named) else ""
                clusters[-1][f"{column}_share"] = _majority_share(named)
                clusters[-1][f"{column}_named"] = len(named)
            clusters[-1]["protein_family_truncated"] = int(text["family_truncated"].sum())
    classes = []
    for ec3, group in single.groupby("ec3"):
        spread = group["cluster"].value_counts()
        classes.append({
            "ec3": ec3, "proteins": len(group), "clusters_spanned": int(len(spread)),
            "largest_cluster": int(spread.index[0]),
            "share_in_largest_cluster": float(spread.iloc[0] / len(group)),
        })
    return {
        "contingency": counts.reset_index(),
        "clusters": pd.DataFrame(clusters),
        "classes": pd.DataFrame(classes).sort_values("proteins", ascending=False),
    }


def _family_seed_summary(
    ids: np.ndarray, labels: np.ndarray, table: pd.DataFrame, families: pd.DataFrame,
) -> dict[str, Any]:
    from sklearn.metrics import (
        adjusted_mutual_info_score, completeness_score, homogeneity_score,
    )

    frame = pd.DataFrame({"cluster": labels}, index=ids)
    named = families.loc[ids, "protein_family"].to_numpy() != ""
    family = families.loc[ids, "protein_family"].to_numpy()[named]
    clusters = labels[named]
    no_ec = (table.loc[ids, "ec1_n"] == 0).to_numpy()
    ec_clusters = set(frame.loc[~no_ec, "cluster"])
    no_ec_clusters = sorted(set(frame["cluster"]) - ec_clusters)
    return {
        "n_records": len(ids), "n_clusters": int(len(set(labels.tolist()))),
        "family_named": int(named.sum()),
        "family_ami": float(adjusted_mutual_info_score(family, clusters)),
        "family_ami_shuffled_mean": float(np.mean([
            adjusted_mutual_info_score(np.random.default_rng(seed).permutation(family), clusters)
            for seed in EC_SHUFFLE_SEEDS
        ])),
        "family_homogeneity": float(homogeneity_score(family, clusters)),
        "family_completeness": float(completeness_score(family, clusters)),
        "no_ec_number": int(no_ec.sum()),
        "clusters_without_ec_number": len(no_ec_clusters),
        "no_ec_share_in_those_clusters": float(frame.loc[no_ec, "cluster"].isin(no_ec_clusters).mean())
        if no_ec.any() else np.nan,
    }


def _write_ec3_table(output_root: Path | None) -> dict[str, pd.DataFrame]:
    seed, agreement = _medoid_reference_seed(output_root)
    _validate_scientific_reference_outputs(output_root)
    table, families = ec_table(), _protein_family_labels()
    profiles, summaries = [], []
    for each in SCIENTIFIC_SEEDS:
        each_paths = _scientific_reference_paths(output_root, each, input_space="raw1024")
        each_run = pd.read_csv(each_paths.runs).iloc[0]
        with np.load(each_paths.assignments / f"{each_run['row']}.npz", allow_pickle=False) as archive:
            each_ids, each_labels = archive["protein_id"].astype(str), archive["label"]
        if len(each_ids) != int(each_run["n"]):
            raise ValueError(f"{each_run['row']}: {len(each_ids)} assigned proteins, {each_run['n']} recorded")
        if each == seed:
            run, ids, labels = each_run, each_ids, each_labels
        profile = _ec3_contingency(each_ids, each_labels, table, families)["clusters"]
        profiles.append(profile.assign(seed=each, row=each_run["row"]))
        summaries.append({"seed": each, "row": each_run["row"], "medoid": each == seed,
                          **_family_seed_summary(each_ids, each_labels, table, families)})
    tables = _ec3_contingency(ids, labels, table, families)
    agreement["chosen"] = agreement["seed"] == seed
    agreement["row"] = np.where(agreement["chosen"], run["row"], "")
    tables["seed_choice"] = agreement
    tables["profiles_all_seeds"] = pd.concat(profiles, ignore_index=True)
    tables["seed_summary"] = pd.DataFrame(summaries)
    metrics = (output_root or REPO / "results" / "dipdeck") / "metrics"
    metrics.mkdir(parents=True, exist_ok=True)
    names = {
        "contingency": "dipdeck_ec3_contingency.csv", "clusters": "dipdeck_ec3_cluster_profile.csv",
        "classes": "dipdeck_ec3_class_profile.csv", "seed_choice": "dipdeck_ec3_table_seed_choice.csv",
        "profiles_all_seeds": "dipdeck_ec3_cluster_profile_all_seeds.csv",
        "seed_summary": "dipdeck_family_seed_summary.csv",
    }
    for key, name in names.items():
        tables[key].to_csv(metrics / name, index=False)
    return tables


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--run-smoke", action="store_true", help="run the bounded smoke fit")
    modes.add_argument(
        "--run-scaling", action="store_true",
        help="run the two-arm reduced-epoch scaling preflight",
    )
    modes.add_argument(
        "--run-full-setting", action="store_true",
        help="run the published 100 plus nominal 50 epoch full-setting timing fit",
    )
    modes.add_argument(
        "--run-scaling-and-timing", action="store_true",
        help="run the scaling preflight, then the full-setting timing fit",
    )
    modes.add_argument(
        "--validate-output",
        action="store_true",
        help="validate an existing smoke output set without fitting",
    )
    modes.add_argument(
        "--validate-scaling-output", action="store_true",
        help="validate copied scaling outputs without importing or fitting ClustPy",
    )
    modes.add_argument(
        "--validate-scaling-timing-output", action="store_true",
        help="validate complete copied scaling and full-setting outputs without fitting",
    )
    modes.add_argument(
        "--run-scientific",
        action="store_true",
        help="run the 20-setting, 10-seed, five-fold scientific grid",
    )
    modes.add_argument(
        "--run-scientific-pca",
        action="store_true",
        help="run one declared PCA version of the 20-setting, 10-seed, five-fold grid",
    )
    modes.add_argument(
        "--run-scientific-reference",
        action="store_true",
        help="run the predeclared paper-default full-data reference across ten seeds",
    )
    modes.add_argument(
        "--run-scientific-pca-reference",
        action="store_true",
        help="run one PCA paper-default full-data reference across ten seeds",
    )
    modes.add_argument(
        "--validate-scientific-output",
        action="store_true",
        help="validate a complete returned scientific-grid output tree without fitting",
    )
    modes.add_argument(
        "--validate-scientific-pca-output",
        action="store_true",
        help="validate one complete returned PCA scientific-grid output tree without fitting",
    )
    modes.add_argument(
        "--summarise-scientific-output",
        action="store_true",
        help="validate the grid and reference, then write their summary tables; "
        "add --pca-dim for one PCA width",
    )
    modes.add_argument(
        "--rescore-scientific-dedup",
        action="store_true",
        help="validate the grid, then rescore held-out partitions with duplicates collapsed",
    )
    modes.add_argument(
        "--compare-scientific-dedup",
        action="store_true",
        help="validate both record sets, then pair each full fit with its deduplicated refit",
    )
    modes.add_argument(
        "--score-scientific-reference",
        action="store_true",
        help="score the 20 reference partitions in raw input space, then rebuild the combined tables",
    )
    modes.add_argument(
        "--combine-scientific-output",
        action="store_true",
        help="stack both record sets' grid and reference tables into one run and cluster table",
    )
    modes.add_argument(
        "--check-pca-basis",
        action="store_true",
        help="compare each fold's training-only PCA basis with one fitted on all records",
    )
    modes.add_argument(
        "--index-scientific-pca-references",
        action="store_true",
        help="validate and index all PCA whole-record reference partitions for comparison",
    )
    modes.add_argument(
        "--validate-scientific-reference-output",
        action="store_true",
        help="validate the complete predeclared full-data reference without fitting",
    )
    modes.add_argument(
        "--validate-scientific-pca-reference-output",
        action="store_true",
        help="validate one complete PCA full-data reference without fitting",
    )
    modes.add_argument(
        "--score-ec",
        action="store_true",
        help="validate every grid and reference arm of --embedding, then score its partitions "
        "against EC",
    )
    modes.add_argument(
        "--ec3-table",
        action="store_true",
        help="write the cluster by EC3 table for the raw reference fit at the medoid seed",
    )
    modes.add_argument(
        "--merge-seed-halves",
        nargs="+",
        type=Path,
        metavar="SOURCE_ROOT",
        help="join grid and reference trees fitted on disjoint seeds into --output-root; "
        "add --pca-dim for the PCA arm",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(os.environ["DIPDECK_OUTPUT_ROOT"]) if os.environ.get("DIPDECK_OUTPUT_ROOT") else None,
        help="write below this DipDECK result root instead of results/dipdeck",
    )
    parser.add_argument(
        "--device", choices=("cpu", "cuda"), default="cuda",
        help="explicit engineering execution device; default: cuda",
    )
    parser.add_argument(
        "--clustpy-source-root", type=Path, default=None,
        help="checked-out ClustPy root used to verify the selected commit and source digest",
    )
    parser.add_argument(
        "--dedup", action="store_true",
        help="use one record per exact-duplicate group, in the *_dedup output trees",
    )
    parser.add_argument(
        "--pca-dim", type=int, choices=ALL_PCA_DIMS,
        help="PCA width for a PCA run or validation mode, or for --summarise-scientific-output",
    )
    parser.add_argument(
        "--embedding", choices=tuple(EMBEDDINGS), default="prott5",
        help="embedding to cluster; default prott5 keeps every earlier name and path",
    )
    parser.add_argument(
        "--seeds", type=_parse_seeds, default=SCIENTIFIC_SEEDS,
        help="run or validate only these declared seeds, e.g. 0-4; default all ten",
    )
    arguments = parser.parse_args(argv)
    dedup = arguments.dedup
    embedding = arguments.embedding
    seeds = arguments.seeds
    if arguments.pca_dim is not None and arguments.pca_dim not in _pca_dims(embedding):
        parser.error(f"--pca-dim for {embedding} must be one of {_pca_dims(embedding)}")
    seed_modes = (
        arguments.run_scientific, arguments.run_scientific_pca, arguments.run_scientific_reference,
        arguments.run_scientific_pca_reference,
        arguments.validate_scientific_output, arguments.validate_scientific_pca_output,
        arguments.validate_scientific_reference_output,
        arguments.validate_scientific_pca_reference_output,
    )
    if seeds != SCIENTIFIC_SEEDS and not any(seed_modes):
        parser.error("--seeds applies only to the scientific run and validation modes")
    expected_grid_rows = len(SCIENTIFIC_FOLDS) * len(_scientific_configs(seeds=seeds))
    dedup_modes = (
        arguments.run_scientific, arguments.run_scientific_pca, arguments.run_scientific_reference,
        arguments.run_scientific_pca_reference,
        arguments.validate_scientific_output, arguments.validate_scientific_pca_output,
        arguments.validate_scientific_reference_output,
        arguments.validate_scientific_pca_reference_output,
        arguments.summarise_scientific_output,
    )
    if dedup and not any(dedup_modes):
        parser.error("--dedup applies only to the scientific run, validation and summary modes")
    pca_modes = (
        arguments.run_scientific_pca,
        arguments.validate_scientific_pca_output,
        arguments.run_scientific_pca_reference,
        arguments.validate_scientific_pca_reference_output,
    )
    if any(pca_modes) and arguments.pca_dim is None:
        parser.error("--pca-dim is required for PCA scientific modes")
    if arguments.pca_dim is not None and not (
        any(pca_modes) or arguments.summarise_scientific_output or arguments.merge_seed_halves
    ):
        parser.error("--pca-dim applies only to PCA scientific modes and the summary mode")
    if arguments.score_ec:
        tables = _write_ec_scores(arguments.output_root, embedding=embedding)
        print(", ".join(f"{name}: rows={len(frame)}" for name, frame in tables.items()))
        return
    if arguments.ec3_table:
        tables = _write_ec3_table(arguments.output_root)
        chosen = tables["seed_choice"].loc[tables["seed_choice"]["chosen"], "seed"].iloc[0]
        print(f"EC3 table written for reference seed {chosen}: clusters={len(tables['clusters'])}, "
              f"classes={len(tables['classes'])}")
        return
    if arguments.merge_seed_halves:
        grid, references = _merge_seed_halves(
            arguments.merge_seed_halves, arguments.output_root,
            embedding=embedding, pca_dimensions=arguments.pca_dim,
        )
        print(f"seed halves merged: grid rows={len(grid)}, reference rows={len(references)}")
        return
    if arguments.validate_scientific_output:
        runs = _validate_scientific_outputs(
            _scientific_paths(
                arguments.output_root, dedup=dedup, input_space=_input_space(embedding, None)
            ),
            expected_rows=expected_grid_rows, dedup=dedup, embedding=embedding, seeds=seeds,
        )
        print(f"scientific output validation passed: rows={len(runs)}")
        return
    if arguments.validate_scientific_pca_output:
        runs = _validate_scientific_outputs(
            _scientific_paths(
                arguments.output_root, dedup=dedup,
                input_space=_input_space(embedding, arguments.pca_dim),
            ),
            expected_rows=expected_grid_rows,
            dedup=dedup,
            pca_dimensions=arguments.pca_dim,
            embedding=embedding,
            seeds=seeds,
        )
        print(f"scientific PCA output validation passed: rows={len(runs)}")
        return
    if arguments.summarise_scientific_output:
        folds, settings = _write_scientific_summaries(
            arguments.output_root, dedup=dedup, pca_dimensions=arguments.pca_dim,
            embedding=embedding,
        )
        print(f"scientific summaries written: folds={len(folds)}, settings={len(settings)}")
        return
    if arguments.rescore_scientific_dedup:
        folds = _write_deduplicated_rescore(arguments.output_root)
        print(f"deduplicated rescore written: folds={len(folds)}")
        return
    if arguments.compare_scientific_dedup:
        paired = _write_record_set_comparison(arguments.output_root)
        print(f"record-set comparison written: rows={len(paired)}")
        return
    if arguments.score_scientific_reference:
        scores = _score_scientific_reference(arguments.output_root)
        path = _reference_scores_path(arguments.output_root)
        path.parent.mkdir(parents=True, exist_ok=True)
        scores.to_csv(path, index=False)
        combined, _clusters, _eligible = _write_combined_scientific_tables(arguments.output_root)
        print(
            f"reference scores written: rows={len(scores)}, "
            f"defined={int((scores['score_status'] == 'defined').sum())}; "
            f"combined tables rebuilt: runs={len(combined)}"
        )
        return
    if arguments.combine_scientific_output:
        combined, combined_clusters, eligible = _write_combined_scientific_tables(
            arguments.output_root
        )
        print(
            f"combined tables written: runs={len(combined)}, clusters={len(combined_clusters)}, "
            f"eligible={len(eligible)}"
        )
        return
    if arguments.check_pca_basis:
        check = _pca_basis_check()
        result_root = arguments.output_root or REPO / "results" / "dipdeck"
        path = result_root / "metrics" / "dipdeck_pca_basis_check.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        check.to_csv(path, index=False)
        print(f"PCA basis check written: rows={len(check)}, path={path}")
        return
    if arguments.index_scientific_pca_references:
        eligible = _write_pca_reference_eligible_partitions(arguments.output_root)
        print(f"PCA reference eligible partitions written: rows={len(eligible)}")
        return
    if arguments.validate_scientific_reference_output:
        runs = _validate_scientific_reference_outputs(
            arguments.output_root, dedup=dedup, embedding=embedding, seeds=seeds
        )
        print(f"scientific reference validation passed: rows={len(runs)}")
        return
    if arguments.validate_scientific_pca_reference_output:
        runs = _validate_scientific_reference_outputs(
            arguments.output_root, dedup=dedup, pca_dimensions=arguments.pca_dim,
            embedding=embedding, seeds=seeds,
        )
        print(f"scientific PCA reference validation passed: rows={len(runs)}")
        return
    if arguments.validate_output:
        run_table, _cluster_table, archive = _validate_existing_outputs(
            _output_paths(arguments.output_root)
        )
        print(
            f"Smoke output validation passed: row={run_table.loc[0, 'row']}, archive={archive}"
        )
        return
    if arguments.validate_scaling_output:
        summary = _validate_scaling_outputs(_scaling_paths(arguments.output_root))
        print(f"Scaling output validation passed: arms={len(summary)}")
        return
    if arguments.validate_scaling_timing_output:
        summary = _validate_scaling_outputs(_scaling_paths(arguments.output_root))
        timing = _validate_full_setting_output(_scaling_paths(arguments.output_root))
        print(f"Scaling and timing output validation passed: arms={len(summary)}, timing_rows={len(timing)}")
        return
    if arguments.run_smoke:
        _run_smoke(arguments.output_root, arguments.device, arguments.clustpy_source_root)
        return
    if arguments.run_scaling:
        summary = _run_scaling(
            arguments.output_root, arguments.device, arguments.clustpy_source_root
        )
        print(f"Scaling preflight completed: arms={len(summary)}")
        return
    if arguments.run_full_setting:
        timing = _run_full_setting(
            arguments.output_root, arguments.device, arguments.clustpy_source_root
        )
        print(f"Full-setting timing completed: rows={len(timing)}")
        return
    if arguments.run_scaling_and_timing:
        summary = _run_scaling(
            arguments.output_root, arguments.device, arguments.clustpy_source_root
        )
        timing = _run_full_setting(
            arguments.output_root, arguments.device, arguments.clustpy_source_root
        )
        print(f"Scaling and timing completed: arms={len(summary)}, timing_rows={len(timing)}")
        return
    if arguments.run_scientific:
        runs = _run_scientific_grid(
            arguments.output_root, arguments.device, arguments.clustpy_source_root, dedup=dedup,
            embedding=embedding, seeds=seeds,
        )
        print(f"scientific grid completed: rows={len(runs)}")
        return
    if arguments.run_scientific_pca:
        runs = _run_scientific_grid(
            arguments.output_root,
            arguments.device,
            arguments.clustpy_source_root,
            dedup=dedup,
            pca_dimensions=arguments.pca_dim,
            embedding=embedding,
            seeds=seeds,
        )
        print(f"scientific PCA grid completed: rows={len(runs)}")
        return
    if arguments.run_scientific_reference:
        runs = _run_scientific_reference(
            arguments.output_root, arguments.device, arguments.clustpy_source_root, dedup=dedup,
            embedding=embedding, seeds=seeds,
        )
        print(f"scientific reference completed: rows={len(runs)}")
        return
    if arguments.run_scientific_pca_reference:
        runs = _run_scientific_reference(
            arguments.output_root,
            arguments.device,
            arguments.clustpy_source_root,
            dedup=dedup,
            pca_dimensions=arguments.pca_dim,
            embedding=embedding,
            seeds=seeds,
        )
        print(f"scientific PCA reference completed: rows={len(runs)}")
        return
    _preflight()


if __name__ == "__main__":
    main()