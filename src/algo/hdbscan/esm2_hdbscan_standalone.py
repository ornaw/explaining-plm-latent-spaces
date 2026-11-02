from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import combinations, product
from pathlib import Path
from typing import Any

import hdbscan
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src import esm2_analysis
from src import esm2_embeddings
from src.algo.hdbscan import hdbscan_standalone as prott5_hdbscan


REPO = Path(__file__).resolve().parents[3]
SOURCE_RUNS_PATH = REPO / "results/hdbscan/metrics/hdbscan_standalone_runs.csv"
FASTA_PATH = REPO / "data/phosphatase/phosphatase/tmp/sequences.fasta"
BUNDLE_SCHEMA = "esm2-hdbscan-bundle-v1"

DATASET = "phosphatase"
FULL_COUNT = 1_587
DEDUPLICATED_COUNT = 1_480
LENGTH_MAXIMUM = 1_022
LENGTH_SUBSET_COUNT = 1_443
CONTROL_SEEDS = (0, 1, 2, 3, 4)
MIN_CLUSTER_SIZES = (5, 10, 20, 40)
MIN_SAMPLES_VALUES = (1, 5, 10, 20, 40)
METRICS = ("euclidean", "cosine")
METRIC_ALGORITHMS = {"euclidean": "prims_kdtree", "cosine": "generic"}
SOURCE_RECORD_SETS = ("full", "deduplicated")
MODEL_WIDTHS = {
    "prott5": 1_024,
    "esm2_t6_8M_UR50D": 320,
    "esm2_t36_3B_UR50D": 2_560,
}
ESM2_MODELS = tuple(model for model in MODEL_WIDTHS if model != "prott5")
MAIN_OUTPUT_ROOTS = {
    model: Path(f"results/hdbscan/{model}") for model in ESM2_MODELS
}
LENGTH_OUTPUT_ROOT = Path("results/hdbscan/esm2_comparison/length_le_1022")
CONTROL_OUTPUT_ROOT = Path("results/hdbscan/esm2_comparison/matched_n_control")
COMPARISON_OUTPUT_ROOT = Path("results/hdbscan/esm2_comparison")
COMPARISON_FIGURES_DIR = Path("figures")
NOISE_FIGURE_NAME = "esm2_full_noise_grid.png"
CLUSTER_COUNT_FIGURE_NAME = "esm2_full_cluster_count_grid.png"
AGREEMENT_FIGURE_NAME = "esm2_full_partition_agreement.png"
NOISE_FREQUENCY_FIGURE_NAME = "esm2_full_noise_frequency_agreement.png"
PROTT5_ASSIGNMENTS_DIR = Path(
    "results/hdbscan/assignments/hdbscan_standalone_grid"
)
COMPARISON_SCHEMA = "esm2-hdbscan-comparison-v1"
GRID_KEY_COLUMNS = (
    "records",
    "metric",
    "min_cluster_size",
    "min_samples",
)
SOURCE_REQUIRED_COLUMNS = {
    "row",
    "dataset",
    "records",
    "space",
    "metric",
    "min_cluster_size",
    "min_samples",
    "cluster_selection_method",
    "algorithm",
}
COSINE_NORM_FLOOR = 1e-8
RUN_COLUMNS = (
    "row",
    "source_prott5_row",
    "dataset",
    "embedding_model",
    "run_group",
    "records",
    "n",
    "subset_hash",
    "subset_seed",
    "space",
    "n_dims",
    "metric",
    "min_cluster_size",
    "min_samples",
    "cluster_selection_method",
    "algorithm",
    "n_clusters",
    "dbcv_scaled_by_clustered_fraction",
    "dbcv_status",
    "silhouette",
    "silhouette_n_scored",
    "n_clustered",
    "n_noise",
    "noise_pct",
    "mean_clustered_probability",
    "mean_clustered_probability_status",
    "git_sha",
    "hdbscan_version",
    "numpy_version",
    "sklearn_version",
    "pandas_version",
)
CLUSTER_COLUMNS = (
    "row",
    "dataset",
    "embedding_model",
    "run_group",
    "records",
    "subset_hash",
    "space",
    "metric",
    "min_cluster_size",
    "min_samples",
    "cluster",
    "n",
    "dbcv",
    "persistence",
    "persistence_status",
    "mean_membership_probability",
)
AGREEMENT_COLUMNS = (
    "comparison_scope",
    "comparison",
    "comparison_label",
    "left_model",
    "right_model",
    "left_row",
    "right_row",
    "left_records",
    "right_records",
    "left_space",
    "right_space",
    "left_metric",
    "right_metric",
    "min_cluster_size",
    "min_samples",
    "left_subset_seed",
    "right_subset_seed",
    "ari_all",
    "ari_clustered_both",
    "n_common",
    "n_clustered_both",
)
FIGURE_RUN_COLUMNS = (
    "dataset",
    "embedding_model",
    "records",
    "n",
    "space",
    "metric",
    "min_cluster_size",
    "min_samples",
    "n_clusters",
    "n_clustered",
    "n_noise",
    "noise_pct",
)
MODEL_DISPLAY_NAMES = {
    "prott5": "ProtT5",
    "esm2_t6_8M_UR50D": "ESM-2 8M",
    "esm2_t36_3B_UR50D": "ESM-2 3B",
}


def _integer_field(value: object, name: str) -> int:
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, float) and value.is_integer():
        return int(value)
    raise ValueError(f"source grid field {name} is not an integer: {value!r}")


@dataclass(frozen=True)
class RunSpecification:
    run_id: str
    run_group: str
    embedding_model: str
    records: str
    space: str
    metric: str
    min_cluster_size: int
    min_samples: int
    cluster_selection_method: str
    algorithm: str
    source_row: str
    subset_seed: int | None = None


@dataclass(frozen=True)
class FitDataset:
    embedding_model: str
    records: str
    ids: tuple[str, ...]
    X: np.ndarray

    @property
    def subset_hash(self) -> str:
        return hashlib.sha1("\n".join(sorted(self.ids)).encode("ascii")).hexdigest()[:12]


@dataclass(frozen=True)
class FitResult:
    run_row: dict[str, Any]
    cluster_rows: tuple[dict[str, Any], ...]
    protein_ids: tuple[str, ...]
    labels: np.ndarray
    probabilities: np.ndarray


@dataclass(frozen=True)
class OutputPaths:
    root: Path
    runs: Path
    clusters: Path
    assignments: Path
    manifest: Path


@dataclass(frozen=True)
class Partition:
    run_id: str
    embedding_model: str
    records: str
    space: str
    metric: str
    min_cluster_size: int
    min_samples: int
    subset_seed: int | None
    protein_ids: tuple[str, ...]
    labels: np.ndarray


@dataclass(frozen=True)
class ComparisonPaths:
    root: Path
    agreement: Path
    length_control: Path
    manifest: Path


def _expected_source_keys() -> set[tuple[str, str, int, int]]:
    return set(
        product(
            SOURCE_RECORD_SETS,
            METRICS,
            MIN_CLUSTER_SIZES,
            MIN_SAMPLES_VALUES,
        )
    )


def read_source_grid(path: Path = SOURCE_RUNS_PATH) -> pd.DataFrame:
    table = pd.read_csv(path)
    missing_columns = SOURCE_REQUIRED_COLUMNS - set(table.columns)
    if missing_columns:
        raise ValueError(f"source run table is missing columns: {sorted(missing_columns)}")

    selected = table.loc[
        (table["dataset"] == DATASET)
        & (table["space"] == "raw1024")
        & table["records"].isin(SOURCE_RECORD_SETS)
        & table["metric"].isin(METRICS)
    ].copy()
    if selected["row"].duplicated().any():
        raise ValueError("source raw grid contains duplicate row IDs")
    if selected.duplicated(list(GRID_KEY_COLUMNS)).any():
        raise ValueError("source raw grid contains duplicate configurations")

    actual_keys = {
        (
            str(row.records),
            str(row.metric),
            _integer_field(row.min_cluster_size, "min_cluster_size"),
            _integer_field(row.min_samples, "min_samples"),
        )
        for row in selected.itertuples(index=False)
    }
    expected_keys = _expected_source_keys()
    if actual_keys != expected_keys:
        missing = sorted(expected_keys - actual_keys)
        extra = sorted(actual_keys - expected_keys)
        raise ValueError(
            f"source raw grid does not match the frozen 80 configurations: "
            f"{len(missing)} missing, {len(extra)} extra"
        )
    if len(selected) != len(expected_keys):
        raise ValueError(f"source raw grid has {len(selected)} rows, expected {len(expected_keys)}")
    expected_algorithms = selected["metric"].map(METRIC_ALGORITHMS)
    if not selected["algorithm"].equals(expected_algorithms):
        raise ValueError("source raw grid does not use the frozen metric algorithms")
    if set(selected["cluster_selection_method"]) != {"eom"}:
        raise ValueError("source raw grid does not use eom cluster selection")

    return selected.sort_values(list(GRID_KEY_COLUMNS), kind="stable").reset_index(drop=True)


def _validate_full_figure_runs(table: pd.DataFrame) -> pd.DataFrame:
    missing_columns = set(FIGURE_RUN_COLUMNS) - set(table.columns)
    if missing_columns:
        raise ValueError(
            f"figure run table is missing columns: {sorted(missing_columns)}"
        )
    selected = table.loc[:, list(FIGURE_RUN_COLUMNS)].copy()
    numeric_columns = (
        "n",
        "min_cluster_size",
        "min_samples",
        "n_clusters",
        "n_clustered",
        "n_noise",
        "noise_pct",
    )
    try:
        selected.loc[:, numeric_columns] = selected.loc[:, numeric_columns].apply(
            pd.to_numeric, errors="raise"
        )
    except (TypeError, ValueError) as error:
        raise ValueError("figure run table contains a non-numeric measure") from error
    if not np.isfinite(selected.loc[:, numeric_columns].to_numpy(dtype=float)).all():
        raise ValueError("figure run table contains a non-finite measure")

    expected_keys = set(product(METRICS, MIN_CLUSTER_SIZES, MIN_SAMPLES_VALUES))
    for model, dimensions in MODEL_WIDTHS.items():
        rows = selected.loc[selected["embedding_model"] == model]
        actual_keys = {
            (
                str(row.metric),
                _integer_field(row.min_cluster_size, "min_cluster_size"),
                _integer_field(row.min_samples, "min_samples"),
            )
            for row in rows.itertuples(index=False)
        }
        if len(rows) != len(expected_keys) or actual_keys != expected_keys:
            raise ValueError(f"figure run rows are incomplete for {model}")
        if rows.duplicated(
            subset=["metric", "min_cluster_size", "min_samples"]
        ).any():
            raise ValueError(f"figure run rows contain duplicates for {model}")
        if set(rows["dataset"]) != {DATASET} or set(rows["records"]) != {"full"}:
            raise ValueError(f"figure run rows do not use full {DATASET} records for {model}")
        if set(rows["space"]) != {f"raw{dimensions}"}:
            raise ValueError(f"figure run rows do not use native coordinates for {model}")
        if set(rows["n"].astype(int)) != {FULL_COUNT}:
            raise ValueError(f"figure run rows have the wrong record count for {model}")

    if set(selected["embedding_model"]) != set(MODEL_WIDTHS):
        raise ValueError("figure run table does not contain exactly the three models")
    integer_columns = (
        "n",
        "min_cluster_size",
        "min_samples",
        "n_clusters",
        "n_clustered",
        "n_noise",
    )
    for column in integer_columns:
        if any(_integer_field(value, column) < 0 for value in selected[column]):
            raise ValueError(f"figure run table contains a negative {column}")
    if not np.array_equal(
        selected["n_clustered"].to_numpy(dtype=int)
        + selected["n_noise"].to_numpy(dtype=int),
        selected["n"].to_numpy(dtype=int),
    ):
        raise ValueError("figure run clustered and noise counts do not sum to n")
    expected_noise = (
        100
        * selected["n_noise"].to_numpy(dtype=float)
        / selected["n"].to_numpy(dtype=float)
    )
    if not np.allclose(
        selected["noise_pct"].to_numpy(dtype=float),
        expected_noise,
        rtol=1e-10,
        atol=1e-10,
    ):
        raise ValueError("figure run noise percentages do not match their counts")
    return selected.sort_values(
        by=["embedding_model", "metric", "min_cluster_size", "min_samples"],
        kind="stable",
    ).reset_index(drop=True)


def load_full_figure_runs(
    *,
    source_grid_path: Path,
    esm2_roots: dict[str, Path],
) -> pd.DataFrame:
    if set(esm2_roots) != set(ESM2_MODELS):
        raise ValueError("figure inputs must identify both ESM-2 model roots")
    prott5 = read_source_grid(source_grid_path)
    prott5 = prott5.loc[prott5["records"] == "full"].copy()
    prott5["embedding_model"] = "prott5"
    tables = [prott5]
    for model in ESM2_MODELS:
        path = Path(esm2_roots[model]) / "metrics/hdbscan_runs.csv"
        try:
            model_table = pd.read_csv(path)
        except (OSError, pd.errors.ParserError) as error:
            raise ValueError(f"cannot read figure run table: {path}") from error
        tables.append(
            model_table.loc[
                (model_table.get("run_group") == "main")
                & (model_table.get("records") == "full")
                & (model_table.get("embedding_model") == model)
            ].copy()
        )
    return _validate_full_figure_runs(pd.concat(tables, ignore_index=True))


def _draw_full_run_grid(
    run_table: pd.DataFrame,
    *,
    column: str,
    value_format: str,
    vmax: float,
    title: str,
    colour_bar_label: str,
    footer: str,
    output_path: Path,
) -> Path:
    table = _validate_full_figure_runs(run_table)
    fig, axes = plt.subplots(
        len(METRICS),
        len(MODEL_WIDTHS),
        figsize=(13, 7.5),
        sharex=True,
        sharey=True,
    )
    fig.suptitle(
        title,
        fontsize=14,
        fontweight="bold",
    )
    image = None
    for metric_index, metric in enumerate(METRICS):
        for model_index, model in enumerate(MODEL_WIDTHS):
            axis = axes[metric_index, model_index]
            rows = table.loc[
                (table["embedding_model"] == model) & (table["metric"] == metric)
            ]
            values = rows.pivot(
                index="min_cluster_size",
                columns="min_samples",
                values=column,
            ).reindex(
                index=list(MIN_CLUSTER_SIZES), columns=list(MIN_SAMPLES_VALUES)
            ).to_numpy(dtype=float)
            if not np.isfinite(values).all():
                raise ValueError(
                    f"{column} figure has a missing cell for {model} {metric}"
                )
            image = axis.imshow(
                values,
                aspect="auto",
                cmap="viridis",
                vmin=0,
                vmax=vmax,
            )
            prott5_hdbscan._annotate_heatmap(axis, image, values, value_format)
            if metric_index == 0:
                dimensions = int(rows["space"].iloc[0].removeprefix("raw"))
                axis.set_title(f"{MODEL_DISPLAY_NAMES[model]} ({dimensions:,} dimensions)")
            axis.set_xticks(np.arange(len(MIN_SAMPLES_VALUES)), MIN_SAMPLES_VALUES)
            axis.set_yticks(np.arange(len(MIN_CLUSTER_SIZES)), MIN_CLUSTER_SIZES)
            axis.set_xlabel("min_samples")
            if model_index == 0:
                axis.set_ylabel(f"{metric.title()}\nmin_cluster_size")
    if image is None:
        raise ValueError(f"{column} figure has no panels")
    colour_axis = fig.add_axes((0.92, 0.16, 0.018, 0.68))
    fig.colorbar(image, cax=colour_axis, label=colour_bar_label)
    record_count = int(table["n"].iloc[0])
    fig.text(
        0.5,
        0.025,
        f"Full phosphatase set, n={record_count:,}. {footer}",
        ha="center",
        fontsize=9,
        color="grey",
    )
    fig.subplots_adjust(
        left=0.09,
        right=0.89,
        top=0.90,
        bottom=0.12,
        hspace=0.28,
        wspace=0.10,
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def draw_noise_figure(run_table: pd.DataFrame, output_path: Path) -> Path:
    return _draw_full_run_grid(
        run_table,
        column="noise_pct",
        value_format=".1f",
        vmax=100,
        title="HDBSCAN records labelled as noise across native embedding spaces",
        colour_bar_label="records labelled as noise (%)",
        footer=(
            "Each cell is one matched run; all panels use the same 0% to 100% "
            "scale."
        ),
        output_path=output_path,
    )


def draw_cluster_count_figure(run_table: pd.DataFrame, output_path: Path) -> Path:
    table = _validate_full_figure_runs(run_table)
    maximum = float(table["n_clusters"].max())
    return _draw_full_run_grid(
        table,
        column="n_clusters",
        value_format=".0f",
        vmax=maximum,
        title="HDBSCAN cluster counts across native embedding spaces",
        colour_bar_label="clusters found (noise excluded)",
        footer=(
            f"Each cell is one matched run; all panels use the same 0 to "
            f"{int(maximum)} scale."
        ),
        output_path=output_path,
    )


def load_full_cross_model_agreement(path: Path) -> pd.DataFrame:
    try:
        table = pd.read_csv(path)
    except (OSError, pd.errors.ParserError) as error:
        raise ValueError(f"cannot read figure agreement table: {path}") from error
    missing_columns = set(AGREEMENT_COLUMNS) - set(table.columns)
    if missing_columns:
        raise ValueError(
            f"figure agreement table is missing columns: {sorted(missing_columns)}"
        )
    selected = table.loc[
        (table["comparison_scope"] == "cross_model")
        & (table["left_records"] == "full")
        & (table["right_records"] == "full"),
        list(AGREEMENT_COLUMNS),
    ].copy()
    expected_pairs = tuple(combinations(MODEL_WIDTHS, 2))
    expected_keys = set(
        product(expected_pairs, METRICS, MIN_CLUSTER_SIZES, MIN_SAMPLES_VALUES)
    )
    actual_keys = {
        (
            (str(row.left_model), str(row.right_model)),
            str(row.left_metric),
            _integer_field(row.min_cluster_size, "min_cluster_size"),
            _integer_field(row.min_samples, "min_samples"),
        )
        for row in selected.itertuples(index=False)
    }
    if len(selected) != len(expected_keys) or actual_keys != expected_keys:
        raise ValueError("full cross-model figure agreement rows are incomplete")
    if selected.duplicated(
        subset=[
            "left_model",
            "right_model",
            "left_metric",
            "min_cluster_size",
            "min_samples",
        ]
    ).any():
        raise ValueError("full cross-model figure agreement rows contain duplicates")
    if not (selected["comparison"] == "cross_model").all():
        raise ValueError("full cross-model figure contains another comparison family")
    if not (selected["left_metric"] == selected["right_metric"]).all():
        raise ValueError("full cross-model figure contains unmatched metrics")
    if set(selected["n_common"].astype(int)) != {FULL_COUNT}:
        raise ValueError("full cross-model figure has the wrong common-record count")
    for row in selected.itertuples(index=False):
        if row.left_space != f"raw{MODEL_WIDTHS[str(row.left_model)]}":
            raise ValueError("full cross-model figure has the wrong left native space")
        if row.right_space != f"raw{MODEL_WIDTHS[str(row.right_model)]}":
            raise ValueError("full cross-model figure has the wrong right native space")
    ari = selected.loc[:, ["ari_all", "ari_clustered_both"]].to_numpy(dtype=float)
    clustered_both = selected["n_clustered_both"].to_numpy(dtype=float)
    common = selected["n_common"].to_numpy(dtype=float)
    if not np.isfinite(ari).all() or np.any((ari < -1) | (ari > 1)):
        raise ValueError("full cross-model figure contains an invalid ARI")
    if not np.isfinite(clustered_both).all() or np.any(
        (clustered_both < 0) | (clustered_both > common)
    ):
        raise ValueError("full cross-model figure contains an invalid shared count")
    selected["coverage_pct"] = 100 * clustered_both / common
    return selected.sort_values(
        by=[
            "left_model",
            "right_model",
            "left_metric",
            "min_cluster_size",
            "min_samples",
        ],
        kind="stable",
    ).reset_index(drop=True)


def draw_partition_agreement_figure(
    agreement_table: pd.DataFrame, output_path: Path
) -> Path:
    table = agreement_table.copy()
    required = set(AGREEMENT_COLUMNS) | {"coverage_pct"}
    if not required.issubset(table.columns):
        raise ValueError("partition agreement figure table is missing required columns")
    ari = table["ari_clustered_both"].to_numpy(dtype=float)
    clustered_both = table["n_clustered_both"].to_numpy(dtype=float)
    common = table["n_common"].to_numpy(dtype=float)
    if not np.isfinite(ari).all() or np.any((ari < -1) | (ari > 1)):
        raise ValueError("partition agreement figure contains an invalid ARI")
    if not np.isfinite(clustered_both).all() or np.any(
        (clustered_both < 0) | (clustered_both > common)
    ):
        raise ValueError("partition agreement figure contains an invalid shared count")
    pairs = tuple(combinations(MODEL_WIDTHS, 2))
    expected_rows = (
        len(pairs)
        * len(METRICS)
        * len(MIN_CLUSTER_SIZES)
        * len(MIN_SAMPLES_VALUES)
    )
    if len(table) != expected_rows:
        raise ValueError("partition agreement figure has an incomplete grid")

    fig, axes = plt.subplots(
        len(METRICS),
        len(pairs),
        figsize=(13, 7.5),
        sharex=True,
        sharey=True,
    )
    fig.suptitle(
        "HDBSCAN agreement among records clustered by both models",
        fontsize=14,
        fontweight="bold",
    )
    image = None
    for metric_index, metric in enumerate(METRICS):
        for pair_index, (left_model, right_model) in enumerate(pairs):
            axis = axes[metric_index, pair_index]
            rows = table.loc[
                (table["left_model"] == left_model)
                & (table["right_model"] == right_model)
                & (table["left_metric"] == metric)
            ]
            if len(rows) != len(MIN_CLUSTER_SIZES) * len(MIN_SAMPLES_VALUES):
                raise ValueError(
                    f"partition agreement figure has missing rows for "
                    f"{left_model} and {right_model} under {metric}"
                )
            values = rows.pivot(
                index="min_cluster_size",
                columns="min_samples",
                values="ari_clustered_both",
            ).reindex(
                index=list(MIN_CLUSTER_SIZES), columns=list(MIN_SAMPLES_VALUES)
            ).to_numpy(dtype=float)
            shared_counts = rows.pivot(
                index="min_cluster_size",
                columns="min_samples",
                values="n_clustered_both",
            ).reindex(
                index=list(MIN_CLUSTER_SIZES), columns=list(MIN_SAMPLES_VALUES)
            ).to_numpy(dtype=float)
            if not np.isfinite(values).all() or not np.isfinite(shared_counts).all():
                raise ValueError(
                    f"partition agreement figure has a missing cell for "
                    f"{left_model} and {right_model} under {metric}"
                )
            image = axis.imshow(
                values,
                aspect="auto",
                cmap="viridis",
                vmin=0,
                vmax=1,
            )
            prott5_hdbscan._annotate_heatmap(
                axis,
                image,
                values,
                ".2f",
                secondary_values=shared_counts,
            )
            if metric_index == 0:
                axis.set_title(
                    f"{MODEL_DISPLAY_NAMES[left_model]} vs "
                    f"{MODEL_DISPLAY_NAMES[right_model]}"
                )
            axis.set_xticks(np.arange(len(MIN_SAMPLES_VALUES)), MIN_SAMPLES_VALUES)
            axis.set_yticks(np.arange(len(MIN_CLUSTER_SIZES)), MIN_CLUSTER_SIZES)
            axis.tick_params(axis="x", labelbottom=True)
            axis.set_xlabel("min_samples")
            if pair_index == 0:
                axis.set_ylabel(f"{metric.title()}\nmin_cluster_size")
    if image is None:
        raise ValueError("partition agreement figure has no panels")
    colour_axis = fig.add_axes((0.92, 0.16, 0.018, 0.68))
    fig.colorbar(image, cax=colour_axis, label="ARI, clustered by both models")
    record_count = int(table["n_common"].iloc[0])
    fig.text(
        0.5,
        0.025,
        f"Full phosphatase set, n={record_count:,}. ARI excludes records labelled "
        "noise by either model; each cell also gives the shared clustered n.",
        ha="center",
        fontsize=9,
        color="grey",
    )
    fig.subplots_adjust(
        left=0.08,
        right=0.89,
        top=0.90,
        bottom=0.12,
        hspace=0.28,
        wspace=0.12,
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def noise_frequency_table(
    partitions: dict[tuple[str, str, str, int, int], Partition],
) -> pd.DataFrame:
    reference = _required_partition(
        partitions,
        model="prott5",
        records="full",
        metric=METRICS[0],
        min_cluster_size=MIN_CLUSTER_SIZES[0],
        min_samples=MIN_SAMPLES_VALUES[0],
    )
    canonical_ids = tuple(sorted(reference.protein_ids))
    if len(canonical_ids) != FULL_COUNT or len(set(canonical_ids)) != FULL_COUNT:
        raise ValueError("noise-frequency reference IDs are incomplete")
    rows: list[dict[str, Any]] = []
    for model in MODEL_WIDTHS:
        for metric in METRICS:
            frequencies = np.zeros(FULL_COUNT, dtype=np.int32)
            for min_cluster_size, min_samples in product(
                MIN_CLUSTER_SIZES, MIN_SAMPLES_VALUES
            ):
                partition = _required_partition(
                    partitions,
                    model=model,
                    records="full",
                    metric=metric,
                    min_cluster_size=min_cluster_size,
                    min_samples=min_samples,
                )
                partition_index = pd.Index(partition.protein_ids)
                positions = partition_index.get_indexer(pd.Index(canonical_ids))
                if np.any(positions < 0) or len(partition_index) != FULL_COUNT:
                    raise ValueError(
                        f"noise-frequency IDs do not align for {partition.run_id}"
                    )
                frequencies += (partition.labels[positions] == -1).astype(np.int32)
            rows.extend(
                {
                    "protein_id": protein_id,
                    "embedding_model": model,
                    "metric": metric,
                    "noise_frequency": int(frequency),
                }
                for protein_id, frequency in zip(
                    canonical_ids, frequencies, strict=True
                )
            )
    return _validate_noise_frequency_table(pd.DataFrame(rows))


def _validate_noise_frequency_table(table: pd.DataFrame) -> pd.DataFrame:
    columns = ("protein_id", "embedding_model", "metric", "noise_frequency")
    missing = set(columns) - set(table.columns)
    if missing:
        raise ValueError(f"noise-frequency table is missing columns: {sorted(missing)}")
    selected = table.loc[:, list(columns)].copy()
    try:
        frequencies = pd.to_numeric(
            selected["noise_frequency"], errors="raise"
        ).to_numpy(dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError("noise-frequency table contains a non-numeric frequency") from error
    maximum = len(MIN_CLUSTER_SIZES) * len(MIN_SAMPLES_VALUES)
    if (
        not np.isfinite(frequencies).all()
        or np.any(frequencies != np.floor(frequencies))
        or np.any((frequencies < 0) | (frequencies > maximum))
    ):
        raise ValueError("noise-frequency table contains an invalid frequency")
    selected["noise_frequency"] = frequencies.astype(int)
    canonical_ids: set[str] | None = None
    for model, metric in product(MODEL_WIDTHS, METRICS):
        rows = selected.loc[
            (selected["embedding_model"] == model)
            & (selected["metric"] == metric)
        ]
        ids = set(rows["protein_id"].astype(str))
        if len(rows) != FULL_COUNT or len(ids) != FULL_COUNT:
            raise ValueError(
                f"noise-frequency rows are incomplete for {model} {metric}"
            )
        if canonical_ids is None:
            canonical_ids = ids
        elif ids != canonical_ids:
            raise ValueError("noise-frequency protein IDs do not align across models")
    expected_rows = FULL_COUNT * len(MODEL_WIDTHS) * len(METRICS)
    if len(selected) != expected_rows:
        raise ValueError("noise-frequency table contains unexpected model or metric rows")
    return selected.sort_values(
        by=["embedding_model", "metric", "protein_id"], kind="stable"
    ).reset_index(drop=True)


def draw_noise_frequency_figure(
    frequency_table: pd.DataFrame, output_path: Path
) -> Path:
    table = _validate_noise_frequency_table(frequency_table)
    pairs = tuple(combinations(MODEL_WIDTHS, 2))
    maximum_frequency = len(MIN_CLUSTER_SIZES) * len(MIN_SAMPLES_VALUES)
    panel_counts: dict[tuple[str, str, str], np.ndarray] = {}
    maximum_count = 0
    for left_model, right_model in pairs:
        for metric in METRICS:
            left = table.loc[
                (table["embedding_model"] == left_model)
                & (table["metric"] == metric),
                ["protein_id", "noise_frequency"],
            ].set_index("protein_id")
            right = table.loc[
                (table["embedding_model"] == right_model)
                & (table["metric"] == metric),
                ["protein_id", "noise_frequency"],
            ].set_index("protein_id")
            aligned = left.join(
                right,
                how="inner",
                lsuffix="_left",
                rsuffix="_right",
                validate="one_to_one",
            )
            if len(aligned) != FULL_COUNT:
                raise ValueError("noise-frequency figure protein IDs do not align")
            counts = np.zeros(
                (maximum_frequency + 1, maximum_frequency + 1), dtype=int
            )
            np.add.at(
                counts,
                (
                    aligned["noise_frequency_left"].to_numpy(dtype=int),
                    aligned["noise_frequency_right"].to_numpy(dtype=int),
                ),
                1,
            )
            if int(counts.sum()) != FULL_COUNT:
                raise ValueError("noise-frequency panel does not contain every protein")
            panel_counts[(left_model, right_model, metric)] = counts
            maximum_count = max(maximum_count, int(counts.max()))
    if maximum_count < 1:
        raise ValueError("noise-frequency figure has no proteins")

    colour_map = plt.get_cmap("viridis").with_extremes(bad="white")
    fig, axes = plt.subplots(
        len(METRICS),
        len(pairs),
        figsize=(13, 8),
        sharex=True,
        sharey=True,
    )
    fig.suptitle(
        "Per-protein HDBSCAN noise frequency across matched settings",
        fontsize=14,
        fontweight="bold",
    )
    image = None
    tick_values = tuple(range(0, maximum_frequency + 1, 5))
    for metric_index, metric in enumerate(METRICS):
        for pair_index, (left_model, right_model) in enumerate(pairs):
            axis = axes[metric_index, pair_index]
            counts = panel_counts[(left_model, right_model, metric)]
            image = axis.imshow(
                np.ma.masked_equal(counts, 0),
                origin="lower",
                interpolation="nearest",
                cmap=colour_map,
                norm=LogNorm(vmin=1, vmax=maximum_count),
                extent=(-0.5, maximum_frequency + 0.5, -0.5, maximum_frequency + 0.5),
            )
            axis.plot(
                [-0.5, maximum_frequency + 0.5],
                [-0.5, maximum_frequency + 0.5],
                color="#777777",
                linestyle="--",
                linewidth=1,
            )
            axis.set_xticks(tick_values)
            axis.set_yticks(tick_values)
            if metric_index == 0:
                axis.set_title(
                    f"{MODEL_DISPLAY_NAMES[left_model]} vs "
                    f"{MODEL_DISPLAY_NAMES[right_model]}"
                )
            axis.set_xlabel(f"Noise frequency: {MODEL_DISPLAY_NAMES[right_model]}")
            axis.set_ylabel(
                f"{metric.title()}\nNoise frequency: {MODEL_DISPLAY_NAMES[left_model]}"
            )
    if image is None:
        raise ValueError("noise-frequency figure has no panels")
    colour_axis = fig.add_axes((0.92, 0.16, 0.018, 0.68))
    fig.colorbar(image, cax=colour_axis, label="proteins (log count scale)")
    fig.text(
        0.5,
        0.025,
        f"Each protein contributes once per panel. Frequency is the number of "
        f"noise labels across {maximum_frequency} matched settings; empty cells are white.",
        ha="center",
        fontsize=9,
        color="grey",
    )
    fig.subplots_adjust(
        left=0.10,
        right=0.89,
        top=0.90,
        bottom=0.13,
        hspace=0.30,
        wspace=0.16,
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def _run_id(
    model: str,
    records: str,
    metric: str,
    min_cluster_size: int,
    min_samples: int,
) -> str:
    space = f"raw{MODEL_WIDTHS[model]}"
    return (
        f"{model}_{records}_{space}_{metric}_"
        f"mcs{min_cluster_size:02d}_ms{min_samples:02d}"
    )


def build_run_specifications(source_grid: pd.DataFrame) -> tuple[RunSpecification, ...]:
    actual_keys = {
        (
            str(row.records),
            str(row.metric),
            _integer_field(row.min_cluster_size, "min_cluster_size"),
            _integer_field(row.min_samples, "min_samples"),
        )
        for row in source_grid.itertuples(index=False)
    }
    if actual_keys != _expected_source_keys() or len(source_grid) != 80:
        raise ValueError("run specifications require the validated 80-row source grid")

    specifications: list[RunSpecification] = []
    for model in ESM2_MODELS:
        for row in source_grid.itertuples(index=False):
            min_cluster_size = _integer_field(row.min_cluster_size, "min_cluster_size")
            min_samples = _integer_field(row.min_samples, "min_samples")
            specifications.append(
                RunSpecification(
                    run_id=_run_id(
                        model,
                        str(row.records),
                        str(row.metric),
                        min_cluster_size,
                        min_samples,
                    ),
                    run_group="main",
                    embedding_model=model,
                    records=str(row.records),
                    space=f"raw{MODEL_WIDTHS[model]}",
                    metric=str(row.metric),
                    min_cluster_size=min_cluster_size,
                    min_samples=min_samples,
                    cluster_selection_method=str(row.cluster_selection_method),
                    algorithm=str(row.algorithm),
                    source_row=str(row.row),
                )
            )

    full_grid = source_grid.loc[source_grid["records"] == "full"]
    for model in MODEL_WIDTHS:
        for row in full_grid.itertuples(index=False):
            records = f"length_le_{LENGTH_MAXIMUM}"
            min_cluster_size = _integer_field(row.min_cluster_size, "min_cluster_size")
            min_samples = _integer_field(row.min_samples, "min_samples")
            specifications.append(
                RunSpecification(
                    run_id=_run_id(
                        model,
                        records,
                        str(row.metric),
                        min_cluster_size,
                        min_samples,
                    ),
                    run_group="length_sensitivity",
                    embedding_model=model,
                    records=records,
                    space=f"raw{MODEL_WIDTHS[model]}",
                    metric=str(row.metric),
                    min_cluster_size=min_cluster_size,
                    min_samples=min_samples,
                    cluster_selection_method=str(row.cluster_selection_method),
                    algorithm=str(row.algorithm),
                    source_row=str(row.row),
                )
            )

    for seed in CONTROL_SEEDS:
        records = f"matched_n_seed_{seed}"
        for row in full_grid.itertuples(index=False):
            min_cluster_size = _integer_field(row.min_cluster_size, "min_cluster_size")
            min_samples = _integer_field(row.min_samples, "min_samples")
            specifications.append(
                RunSpecification(
                    run_id=_run_id(
                        "prott5",
                        records,
                        str(row.metric),
                        min_cluster_size,
                        min_samples,
                    ),
                    run_group="matched_n_control",
                    embedding_model="prott5",
                    records=records,
                    space="raw1024",
                    metric=str(row.metric),
                    min_cluster_size=min_cluster_size,
                    min_samples=min_samples,
                    cluster_selection_method=str(row.cluster_selection_method),
                    algorithm=str(row.algorithm),
                    source_row=str(row.row),
                    subset_seed=seed,
                )
            )

    run_ids = [specification.run_id for specification in specifications]
    if len(run_ids) != len(set(run_ids)):
        raise ValueError("run specification IDs are not unique")
    return tuple(specifications)


def bundle_specifications(
    specifications: tuple[RunSpecification, ...],
    *,
    run_group: str,
    model: str | None = None,
) -> tuple[RunSpecification, ...]:
    if run_group == "main":
        if model not in ESM2_MODELS:
            raise ValueError("a main bundle requires one registered ESM-2 model")
        selected = tuple(
            item
            for item in specifications
            if item.run_group == run_group and item.embedding_model == model
        )
        expected_count = 80
    elif run_group == "length_sensitivity":
        if model is not None:
            raise ValueError("the length-sensitivity bundle contains all three models")
        selected = tuple(item for item in specifications if item.run_group == run_group)
        expected_count = 120
    elif run_group == "matched_n_control":
        if model is not None:
            raise ValueError("the matched-n control bundle is ProtT5 only")
        selected = tuple(item for item in specifications if item.run_group == run_group)
        expected_count = 200
    else:
        raise ValueError(f"unknown run group: {run_group}")
    if len(selected) != expected_count:
        raise ValueError(
            f"{run_group} bundle has {len(selected)} specifications, expected {expected_count}"
        )
    return selected


def output_paths(
    output_root: Path,
    *,
    run_group: str,
    model: str | None = None,
    repository_root: Path = REPO,
) -> OutputPaths:
    repository_root = Path(repository_root).resolve()
    if run_group == "main" and model in MAIN_OUTPUT_ROOTS:
        relative_root = MAIN_OUTPUT_ROOTS[model]
    elif run_group == "length_sensitivity" and model is None:
        relative_root = LENGTH_OUTPUT_ROOT
    elif run_group == "matched_n_control" and model is None:
        relative_root = CONTROL_OUTPUT_ROOT
    else:
        raise ValueError("run group and model do not identify an output bundle")
    resolved = Path(output_root).resolve()
    expected = (repository_root / relative_root).resolve()
    if resolved != expected:
        raise ValueError(
            f"output root is not allowlisted for {run_group}: {resolved}"
        )
    return OutputPaths(
        root=resolved,
        runs=resolved / "metrics/hdbscan_runs.csv",
        clusters=resolved / "metrics/hdbscan_clusters.csv",
        assignments=resolved / "assignments",
        manifest=resolved / "run_manifest.json",
    )


def length_subset_ids(
    fasta_path: Path = FASTA_PATH,
    *,
    maximum_length: int = LENGTH_MAXIMUM,
    expected_full_count: int = FULL_COUNT,
    expected_subset_count: int = LENGTH_SUBSET_COUNT,
) -> tuple[str, ...]:
    sequences = esm2_embeddings.read_fasta(Path(fasta_path))
    if len(sequences) != expected_full_count:
        raise ValueError(
            f"FASTA has {len(sequences)} records, expected {expected_full_count}"
        )
    selected = tuple(
        protein_id
        for protein_id in sorted(sequences)
        if len(sequences[protein_id]) <= maximum_length
    )
    if len(selected) != expected_subset_count:
        raise ValueError(
            f"length subset has {len(selected)} records, expected {expected_subset_count}"
        )
    return selected


def matched_n_control_subsets(
    canonical_ids: tuple[str, ...],
    *,
    subset_size: int,
    seeds: tuple[int, ...],
) -> dict[int, tuple[str, ...]]:
    if canonical_ids != tuple(sorted(canonical_ids)):
        raise ValueError("control sampling requires canonical sorted protein IDs")
    if len(canonical_ids) != len(set(canonical_ids)):
        raise ValueError("control sampling requires unique protein IDs")
    if not 0 < subset_size < len(canonical_ids):
        raise ValueError("control subset size must be positive and smaller than the full set")
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("control seeds must be non-empty and unique")

    controls: dict[int, tuple[str, ...]] = {}
    for seed in seeds:
        positions = np.sort(
            np.random.default_rng(seed).choice(
                len(canonical_ids), size=subset_size, replace=False
            )
        )
        controls[seed] = tuple(canonical_ids[int(position)] for position in positions)
    return controls


def canonical_fasta_ids(
    fasta_path: Path, *, expected_count: int = FULL_COUNT
) -> tuple[str, ...]:
    sequences = esm2_embeddings.read_fasta(Path(fasta_path))
    if len(sequences) != expected_count:
        raise ValueError(f"FASTA has {len(sequences)} records, expected {expected_count}")
    return tuple(sorted(sequences))


def _as_fit_dataset(
    dataset: esm2_analysis.AnalysisDataset,
    *,
    embedding_model: str,
    records: str,
) -> FitDataset:
    if dataset.ids != tuple(sorted(dataset.ids)):
        raise ValueError(f"{embedding_model} input is not in canonical protein-ID order")
    return FitDataset(
        embedding_model=embedding_model,
        records=records,
        ids=dataset.ids,
        X=dataset.X,
    )


def _profile_unique_count(model: esm2_analysis.ModelDataset) -> int:
    profile_path = model.output_root / f"embedding_profile_{model.stem}.json"
    try:
        payload = json.loads(profile_path.read_text(encoding="utf-8"))
        count = int(payload["profile"]["sequence_deduplication"]["n_unique_sequences"])
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid ESM-2 profile: {profile_path}") from error
    if payload.get("schema") != esm2_analysis.PROFILE_SCHEMA:
        raise ValueError(f"ESM-2 profile schema mismatch: {profile_path}")
    if payload.get("model", {}).get("stem") != model.stem:
        raise ValueError(f"ESM-2 profile names the wrong model: {profile_path}")
    if payload.get("input", {}).get("h5_sha256") != model.manifest["artifact"]["sha256"]:
        raise ValueError(f"ESM-2 profile names the wrong HDF5 digest: {profile_path}")
    return count


def main_fit_datasets(
    model: esm2_analysis.ModelDataset,
    *,
    expected_full_count: int = FULL_COUNT,
) -> dict[tuple[str, str], FitDataset]:
    if model.stem not in ESM2_MODELS:
        raise ValueError(f"unexpected ESM-2 model stem: {model.stem}")
    if len(model.data) != expected_full_count:
        raise ValueError(
            f"{model.stem} has {len(model.data)} records, expected {expected_full_count}"
        )
    deduplicated = esm2_analysis.deduplicate_by_sequence(model.data)
    measured_count = _profile_unique_count(model)
    if len(deduplicated) != measured_count:
        raise ValueError(
            f"{model.stem} sequence-deduplicated count is {len(deduplicated)}, "
            f"but the ESM-2 profile recorded {measured_count}"
        )
    return {
        (model.stem, "full"): _as_fit_dataset(
            model.data, embedding_model=model.stem, records="full"
        ),
        (model.stem, "deduplicated"): _as_fit_dataset(
            deduplicated, embedding_model=model.stem, records="deduplicated"
        ),
    }


def _subset_fit_dataset(
    dataset: esm2_analysis.AnalysisDataset,
    *,
    embedding_model: str,
    records: str,
    selected_ids: tuple[str, ...],
) -> FitDataset:
    index = pd.Index(dataset.ids)
    if index.has_duplicates:
        raise ValueError(f"{embedding_model} contains duplicate protein IDs")
    positions = index.get_indexer(pd.Index(selected_ids))
    missing = np.flatnonzero(positions == -1)
    if missing.size:
        raise ValueError(
            f"{embedding_model} is missing selected protein {selected_ids[int(missing[0])]}"
        )
    return FitDataset(
        embedding_model=embedding_model,
        records=records,
        ids=selected_ids,
        X=dataset.X[positions],
    )


def length_fit_datasets(
    reference: esm2_analysis.AnalysisDataset,
    models: list[esm2_analysis.ModelDataset],
    selected_ids: tuple[str, ...],
    *,
    expected_full_count: int = FULL_COUNT,
    expected_subset_count: int = LENGTH_SUBSET_COUNT,
) -> dict[tuple[str, str], FitDataset]:
    if len(selected_ids) != expected_subset_count:
        raise ValueError(
            f"length subset has {len(selected_ids)} records, expected {expected_subset_count}"
        )
    if selected_ids != tuple(sorted(selected_ids)) or len(selected_ids) != len(set(selected_ids)):
        raise ValueError("length-subset IDs must be unique and in canonical order")
    datasets = [("prott5", reference), *((model.stem, model.data) for model in models)]
    if {name for name, _dataset in datasets} != set(MODEL_WIDTHS):
        raise ValueError("length sensitivity requires ProtT5 and both registered ESM-2 models")
    expected_ids = reference.ids
    expected_sequences = reference.sequences
    if len(reference) != expected_full_count:
        raise ValueError(
            f"ProtT5 has {len(reference)} records, expected {expected_full_count}"
        )
    for name, dataset in datasets:
        if dataset.ids != expected_ids or dataset.sequences != expected_sequences:
            raise ValueError(f"{name} is not aligned to the ProtT5 IDs and sequences")
    records = f"length_le_{LENGTH_MAXIMUM}"
    return {
        (name, records): _subset_fit_dataset(
            dataset,
            embedding_model=name,
            records=records,
            selected_ids=selected_ids,
        )
        for name, dataset in datasets
    }


def control_fit_datasets(
    reference: esm2_analysis.AnalysisDataset,
    controls: dict[int, tuple[str, ...]],
    *,
    expected_full_count: int = FULL_COUNT,
    expected_subset_count: int = LENGTH_SUBSET_COUNT,
) -> dict[tuple[str, str], FitDataset]:
    if len(reference) != expected_full_count:
        raise ValueError(
            f"ProtT5 has {len(reference)} records, expected {expected_full_count}"
        )
    if tuple(controls) != CONTROL_SEEDS:
        raise ValueError(f"control subsets must use seeds {CONTROL_SEEDS}")
    datasets: dict[tuple[str, str], FitDataset] = {}
    for seed, selected_ids in controls.items():
        if len(selected_ids) != expected_subset_count:
            raise ValueError(
                f"control seed {seed} has {len(selected_ids)} records, "
                f"expected {expected_subset_count}"
            )
        records = f"matched_n_seed_{seed}"
        datasets[("prott5", records)] = _subset_fit_dataset(
            reference,
            embedding_model="prott5",
            records=records,
            selected_ids=selected_ids,
        )
    return datasets


def _relative_or_absolute(path: Path, repository_root: Path) -> str:
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(Path(repository_root).resolve()).as_posix()
    except ValueError:
        return str(resolved)


def input_manifest_context(
    *,
    reference_h5: Path,
    annotation_path: Path,
    fasta_path: Path,
    models: list[esm2_analysis.ModelDataset],
    repository_root: Path = REPO,
) -> dict[str, Any]:
    files = {
        "reference_h5": Path(reference_h5),
        "annotation": Path(annotation_path),
        "fasta": Path(fasta_path),
    }
    context: dict[str, Any] = {
        name: {
            "path": _relative_or_absolute(path, repository_root),
            "sha256": esm2_embeddings.sha256_file(path),
        }
        for name, path in files.items()
    }
    context["esm2_models"] = [
        {
            "model_key": model.model_key,
            "checkpoint": model.checkpoint,
            "revision": model.revision,
            "stem": model.stem,
            "h5_path": _relative_or_absolute(model.h5_path, repository_root),
            "h5_sha256": model.manifest["artifact"]["sha256"],
        }
        for model in models
    ]
    return context


def _partition_key(
    model: str,
    records: str,
    metric: str,
    min_cluster_size: int,
    min_samples: int,
) -> tuple[str, str, str, int, int]:
    return model, records, metric, min_cluster_size, min_samples


def _required_partition(
    partitions: dict[tuple[str, str, str, int, int], Partition],
    *,
    model: str,
    records: str,
    metric: str,
    min_cluster_size: int,
    min_samples: int,
) -> Partition:
    key = _partition_key(model, records, metric, min_cluster_size, min_samples)
    try:
        return partitions[key]
    except KeyError as error:
        raise ValueError(f"partition is missing for {key}") from error


def _agreement_row(
    *,
    comparison_scope: str,
    comparison: str,
    comparison_label: str,
    left: Partition,
    right: Partition,
    expected_common: int,
) -> dict[str, Any]:
    if (
        left.metric,
        left.min_cluster_size,
        left.min_samples,
    ) != (
        right.metric,
        right.min_cluster_size,
        right.min_samples,
    ):
        raise ValueError("partition comparison does not match HDBSCAN parameters")
    agreement = prott5_hdbscan._partition_ari(
        list(left.protein_ids),
        left.labels,
        list(right.protein_ids),
        right.labels,
    )
    if agreement["n_common"] != expected_common:
        raise ValueError(
            f"partition comparison has {agreement['n_common']} common IDs, "
            f"expected {expected_common}"
        )
    return {
        "comparison_scope": comparison_scope,
        "comparison": comparison,
        "comparison_label": comparison_label,
        "left_model": left.embedding_model,
        "right_model": right.embedding_model,
        "left_row": left.run_id,
        "right_row": right.run_id,
        "left_records": left.records,
        "right_records": right.records,
        "left_space": left.space,
        "right_space": right.space,
        "left_metric": left.metric,
        "right_metric": right.metric,
        "min_cluster_size": left.min_cluster_size,
        "min_samples": left.min_samples,
        "left_subset_seed": left.subset_seed,
        "right_subset_seed": right.subset_seed,
        **agreement,
    }


def partition_agreement_table(
    partitions: dict[tuple[str, str, str, int, int], Partition],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    configurations = tuple(product(METRICS, MIN_CLUSTER_SIZES, MIN_SAMPLES_VALUES))
    for records, expected_common in (
        ("full", FULL_COUNT),
        ("deduplicated", DEDUPLICATED_COUNT),
        (f"length_le_{LENGTH_MAXIMUM}", LENGTH_SUBSET_COUNT),
    ):
        for left_model, right_model in combinations(MODEL_WIDTHS, 2):
            for metric, min_cluster_size, min_samples in configurations:
                left = _required_partition(
                    partitions,
                    model=left_model,
                    records=records,
                    metric=metric,
                    min_cluster_size=min_cluster_size,
                    min_samples=min_samples,
                )
                right = _required_partition(
                    partitions,
                    model=right_model,
                    records=records,
                    metric=metric,
                    min_cluster_size=min_cluster_size,
                    min_samples=min_samples,
                )
                rows.append(
                    _agreement_row(
                        comparison_scope="cross_model",
                        comparison="cross_model",
                        comparison_label=f"{left_model}_vs_{right_model}:{records}",
                        left=left,
                        right=right,
                        expected_common=expected_common,
                    )
                )

    for model in MODEL_WIDTHS:
        for metric, min_cluster_size, min_samples in configurations:
            full = _required_partition(
                partitions,
                model=model,
                records="full",
                metric=metric,
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
            )
            for comparison, records, expected_common in (
                ("full_vs_deduplicated", "deduplicated", DEDUPLICATED_COUNT),
                (
                    "full_vs_length",
                    f"length_le_{LENGTH_MAXIMUM}",
                    LENGTH_SUBSET_COUNT,
                ),
            ):
                right = _required_partition(
                    partitions,
                    model=model,
                    records=records,
                    metric=metric,
                    min_cluster_size=min_cluster_size,
                    min_samples=min_samples,
                )
                rows.append(
                    _agreement_row(
                        comparison_scope="record_set",
                        comparison=comparison,
                        comparison_label=f"{model}:{comparison}",
                        left=full,
                        right=right,
                        expected_common=expected_common,
                    )
                )

    for seed in CONTROL_SEEDS:
        records = f"matched_n_seed_{seed}"
        for metric, min_cluster_size, min_samples in configurations:
            left = _required_partition(
                partitions,
                model="prott5",
                records="full",
                metric=metric,
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
            )
            right = _required_partition(
                partitions,
                model="prott5",
                records=records,
                metric=metric,
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
            )
            rows.append(
                _agreement_row(
                    comparison_scope="matched_n_control",
                    comparison="full_vs_matched_n",
                    comparison_label=f"prott5:full_vs_matched_n_seed_{seed}",
                    left=left,
                    right=right,
                    expected_common=LENGTH_SUBSET_COUNT,
                )
            )
    table = pd.DataFrame(rows, columns=AGREEMENT_COLUMNS)
    if len(table) != 800 or table.duplicated(
        ["comparison_label", "left_row", "right_row"]
    ).any():
        raise ValueError("partition agreement table is incomplete or contains duplicates")
    return table


def length_control_summary(agreement_table: pd.DataFrame) -> pd.DataFrame:
    required = set(AGREEMENT_COLUMNS)
    if not required.issubset(agreement_table.columns):
        raise ValueError("partition agreement table is missing required columns")
    length_rows = agreement_table.loc[
        agreement_table["comparison"] == "full_vs_length"
    ]
    control_rows = agreement_table.loc[
        agreement_table["comparison"] == "full_vs_matched_n"
    ]
    rows: list[dict[str, Any]] = []
    for model in MODEL_WIDTHS:
        for metric, min_cluster_size, min_samples in product(
            METRICS, MIN_CLUSTER_SIZES, MIN_SAMPLES_VALUES
        ):
            key = (
                (length_rows["left_model"] == model)
                & (length_rows["left_metric"] == metric)
                & (length_rows["min_cluster_size"] == min_cluster_size)
                & (length_rows["min_samples"] == min_samples)
            )
            matched_length = length_rows.loc[key]
            control_key = (
                (control_rows["left_metric"] == metric)
                & (control_rows["min_cluster_size"] == min_cluster_size)
                & (control_rows["min_samples"] == min_samples)
            )
            matched_controls = control_rows.loc[control_key].sort_values(
                "right_subset_seed"
            )
            if len(matched_length) != 1 or len(matched_controls) != len(CONTROL_SEEDS):
                raise ValueError("length-control comparison rows are incomplete")
            length_row = matched_length.iloc[0]
            for policy, column in (
                ("all_records", "ari_all"),
                ("clustered_both", "ari_clustered_both"),
            ):
                length_ari = float(length_row[column])
                control_values = matched_controls[column].astype(float).to_numpy()
                if not np.isfinite(length_ari) or not np.isfinite(control_values).all():
                    classification = "Unavailable"
                elif length_ari < float(control_values.min()):
                    classification = "Distinguishable"
                else:
                    classification = "Uninterpretable"
                rows.append(
                    {
                        "embedding_model": model,
                        "metric": metric,
                        "min_cluster_size": min_cluster_size,
                        "min_samples": min_samples,
                        "ari_policy": policy,
                        "full_vs_length_ari": length_ari,
                        "control_ari_min": float(np.min(control_values)),
                        "control_ari_median": float(np.median(control_values)),
                        "control_ari_max": float(np.max(control_values)),
                        "n_controls": len(control_values),
                        "classification": classification,
                    }
                )
    summary = pd.DataFrame(rows)
    if len(summary) != 240:
        raise ValueError(f"length-control summary has {len(summary)} rows, expected 240")
    return summary


def _read_partition_archive(path: Path, *, expected_n: int) -> tuple[tuple[str, ...], np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != {"protein_id", "label", "membership_probability"}:
                raise ValueError(f"assignment keys are invalid: {path}")
            protein_ids = tuple(np.asarray(archive["protein_id"]).astype(str).tolist())
            labels = np.asarray(archive["label"])
            probabilities = np.asarray(archive["membership_probability"], dtype=float)
    except OSError as error:
        raise ValueError(f"cannot read partition assignment: {path}") from error
    if len(protein_ids) != expected_n or len(set(protein_ids)) != expected_n:
        raise ValueError(f"assignment ID count is invalid: {path}")
    prott5_hdbscan._cluster_ids(labels)
    prott5_hdbscan._membership_probability(probabilities, labels)
    return protein_ids, np.asarray(labels, dtype=np.int32)


def load_prott5_partitions(
    source_grid: pd.DataFrame,
    assignments_dir: Path,
    *,
    repository_root: Path = REPO,
) -> dict[tuple[str, str, str, int, int], Partition]:
    resolved = Path(assignments_dir).resolve()
    expected = (Path(repository_root).resolve() / PROTT5_ASSIGNMENTS_DIR).resolve()
    if resolved != expected:
        raise ValueError("ProtT5 assignment directory is not the protected source directory")
    if len(source_grid) != 80:
        raise ValueError("ProtT5 partition loading requires the validated 80-row source grid")
    partitions: dict[tuple[str, str, str, int, int], Partition] = {}
    for row in source_grid.itertuples(index=False):
        records = str(row.records)
        metric = str(row.metric)
        min_cluster_size = _integer_field(row.min_cluster_size, "min_cluster_size")
        min_samples = _integer_field(row.min_samples, "min_samples")
        run_id = str(row.row)
        expected_n = _integer_field(row.n, "n")
        archive_path = resolved / f"{run_id}_{records}_raw1024.npz"
        protein_ids, labels = _read_partition_archive(
            archive_path, expected_n=expected_n
        )
        partition = Partition(
            run_id=run_id,
            embedding_model="prott5",
            records=records,
            space="raw1024",
            metric=metric,
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            subset_seed=None,
            protein_ids=protein_ids,
            labels=labels,
        )
        key = _partition_key(
            "prott5", records, metric, min_cluster_size, min_samples
        )
        if key in partitions:
            raise ValueError(f"duplicate ProtT5 partition key: {key}")
        partitions[key] = partition
    return partitions


def load_bundle_partitions(
    output_root: Path,
    specifications: tuple[RunSpecification, ...],
    *,
    run_group: str,
    model: str | None = None,
    repository_root: Path = REPO,
) -> dict[tuple[str, str, str, int, int], Partition]:
    validate_bundle(
        output_root,
        specifications,
        run_group=run_group,
        model=model,
        repository_root=repository_root,
    )
    paths = output_paths(
        output_root,
        run_group=run_group,
        model=model,
        repository_root=repository_root,
    )
    selected = bundle_specifications(
        specifications, run_group=run_group, model=model
    )
    runs = pd.read_csv(paths.runs).set_index("row")
    partitions: dict[tuple[str, str, str, int, int], Partition] = {}
    for specification in selected:
        row = runs.loc[specification.run_id]
        if not isinstance(row, pd.Series):
            raise ValueError("bundle run metadata must have unique row IDs")
        protein_ids, labels = _read_partition_archive(
            paths.assignments / f"{specification.run_id}.npz",
            expected_n=_integer_field(row["n"], "n"),
        )
        subset_seed = (
            None
            if pd.isna(row["subset_seed"])
            else _integer_field(row["subset_seed"], "subset_seed")
        )
        partition = Partition(
            run_id=specification.run_id,
            embedding_model=specification.embedding_model,
            records=specification.records,
            space=specification.space,
            metric=specification.metric,
            min_cluster_size=specification.min_cluster_size,
            min_samples=specification.min_samples,
            subset_seed=subset_seed,
            protein_ids=protein_ids,
            labels=labels,
        )
        key = _partition_key(
            partition.embedding_model,
            partition.records,
            partition.metric,
            partition.min_cluster_size,
            partition.min_samples,
        )
        if key in partitions:
            raise ValueError(f"duplicate bundle partition key: {key}")
        partitions[key] = partition
    return partitions


def comparison_output_paths(
    output_root: Path, *, repository_root: Path = REPO
) -> ComparisonPaths:
    resolved = Path(output_root).resolve()
    expected = (Path(repository_root).resolve() / COMPARISON_OUTPUT_ROOT).resolve()
    if resolved != expected:
        raise ValueError("HDBSCAN comparison output root is not allowlisted")
    return ComparisonPaths(
        root=resolved,
        agreement=resolved / "esm2_partition_agreement.csv",
        length_control=resolved / "length_control_summary.csv",
        manifest=resolved / "comparison_manifest.json",
    )


def _file_record(path: Path, repository_root: Path) -> dict[str, str]:
    return {
        "path": _relative_or_absolute(path, repository_root),
        "sha256": esm2_embeddings.sha256_file(path),
    }


def comparison_source_context(
    *,
    source_grid_path: Path,
    prott5_assignments_dir: Path,
    bundle_roots: list[Path],
    repository_root: Path = REPO,
) -> dict[str, Any]:
    source_grid = read_source_grid(source_grid_path)
    assignments = []
    for row in source_grid.itertuples(index=False):
        path = Path(prott5_assignments_dir) / (
            f"{row.row}_{row.records}_raw1024.npz"
        )
        assignments.append(_file_record(path, repository_root))
    manifests = [
        _file_record(Path(root) / "run_manifest.json", repository_root)
        for root in bundle_roots
    ]
    return {
        "source_grid": _file_record(Path(source_grid_path), repository_root),
        "prott5_assignments": assignments,
        "output_bundles": manifests,
    }


def _validate_comparison_tables(
    agreement: pd.DataFrame, summary: pd.DataFrame
) -> None:
    if tuple(agreement.columns) != AGREEMENT_COLUMNS:
        raise ValueError("comparison agreement columns do not match the schema")
    expected_scopes = {
        "cross_model": 360,
        "record_set": 240,
        "matched_n_control": 200,
    }
    if (
        len(agreement) != 800
        or agreement["comparison_scope"].value_counts().to_dict() != expected_scopes
    ):
        raise ValueError("comparison agreement rows are incomplete")
    if agreement.duplicated(["comparison_label", "left_row", "right_row"]).any():
        raise ValueError("comparison agreement rows contain duplicates")
    recomputed = length_control_summary(agreement)
    try:
        pd.testing.assert_frame_equal(
            summary.reset_index(drop=True),
            recomputed.reset_index(drop=True),
            check_exact=False,
            rtol=1e-12,
            atol=1e-12,
        )
    except AssertionError as error:
        raise ValueError("length-control summary does not match the agreement rows") from error


def _resolve_record_path(record: dict[str, Any], repository_root: Path) -> Path:
    path = Path(str(record.get("path", "")))
    return path if path.is_absolute() else Path(repository_root).resolve() / path


def _validate_comparison_sources(sources: dict[str, Any], repository_root: Path) -> None:
    unvalidated = [sources.get("source_grid")]
    unvalidated.extend(sources.get("prott5_assignments", []))
    unvalidated.extend(sources.get("output_bundles", []))
    records: list[dict[str, Any]] = []
    for record in unvalidated:
        if not isinstance(record, dict):
            raise ValueError("comparison manifest source records are invalid")
        records.append(record)
    for record in records:
        path = _resolve_record_path(record, repository_root)
        if esm2_embeddings.sha256_file(path) != record.get("sha256"):
            raise ValueError(f"comparison source digest mismatch: {path}")


def validate_comparison_outputs(
    output_root: Path, *, repository_root: Path = REPO
) -> dict[str, Any]:
    paths = comparison_output_paths(output_root, repository_root=repository_root)
    try:
        manifest = json.loads(paths.manifest.read_text(encoding="utf-8"))
        agreement = pd.read_csv(paths.agreement)
        summary = pd.read_csv(paths.length_control)
    except (OSError, json.JSONDecodeError, pd.errors.ParserError) as error:
        raise ValueError("invalid HDBSCAN comparison output") from error
    if manifest.get("schema") != COMPARISON_SCHEMA:
        raise ValueError("HDBSCAN comparison manifest schema mismatch")
    _validate_comparison_tables(agreement, summary)
    expected_outputs = {
        paths.agreement.name: esm2_embeddings.sha256_file(paths.agreement),
        paths.length_control.name: esm2_embeddings.sha256_file(paths.length_control),
    }
    if manifest.get("outputs") != expected_outputs:
        raise ValueError("HDBSCAN comparison output digest mismatch")
    sources = manifest.get("sources")
    if not isinstance(sources, dict):
        raise ValueError("HDBSCAN comparison manifest has no sources")
    _validate_comparison_sources(sources, repository_root)
    return manifest


def publish_comparison_outputs(
    agreement: pd.DataFrame,
    summary: pd.DataFrame,
    *,
    output_root: Path,
    sources: dict[str, Any],
    repository_root: Path = REPO,
) -> dict[str, Any]:
    _validate_comparison_tables(agreement, summary)
    paths = comparison_output_paths(output_root, repository_root=repository_root)
    paths.root.mkdir(parents=True, exist_ok=True)
    agreement_text = agreement.to_csv(index=False, lineterminator="\n")
    summary_text = summary.to_csv(index=False, lineterminator="\n")
    outputs = {
        paths.agreement.name: hashlib.sha256(agreement_text.encode("utf-8")).hexdigest(),
        paths.length_control.name: hashlib.sha256(summary_text.encode("utf-8")).hexdigest(),
    }
    manifest = {
        "schema": COMPARISON_SCHEMA,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "agreement_rows": len(agreement),
        "length_control_rows": len(summary),
        "sources": sources,
        "outputs": outputs,
    }
    contents = {
        paths.agreement: agreement_text,
        paths.length_control: summary_text,
        paths.manifest: json.dumps(manifest, indent=2) + "\n",
    }
    token = uuid.uuid4().hex
    staged = {
        target: target.with_name(f".{target.name}.{token}.tmp")
        for target in contents
    }
    backups = {
        target: target.with_name(f".{target.name}.{token}.bak")
        for target in contents
    }
    promoted: list[Path] = []
    backed_up: list[Path] = []
    try:
        for target, text in contents.items():
            staged[target].write_text(text, encoding="utf-8", newline="\n")
        for target in contents:
            if target.exists():
                os.replace(target, backups[target])
                backed_up.append(target)
        for target in contents:
            os.replace(staged[target], target)
            promoted.append(target)
        checked = validate_comparison_outputs(
            paths.root, repository_root=repository_root
        )
        for target in backed_up:
            backups[target].unlink(missing_ok=True)
        return checked
    except BaseException:
        for target in promoted:
            target.unlink(missing_ok=True)
        for target in backed_up:
            if backups[target].exists():
                os.replace(backups[target], target)
        raise
    finally:
        for path in (*staged.values(), *backups.values()):
            path.unlink(missing_ok=True)


def run_comparison(
    *,
    source_grid_path: Path,
    prott5_assignments_dir: Path,
    esm2_8m_root: Path,
    esm2_3b_root: Path,
    length_root: Path,
    control_root: Path,
    output_root: Path,
    repository_root: Path = REPO,
) -> dict[str, Any]:
    source_grid = read_source_grid(source_grid_path)
    specifications = build_run_specifications(source_grid)
    partitions = load_prott5_partitions(
        source_grid,
        prott5_assignments_dir,
        repository_root=repository_root,
    )
    bundle_contracts = (
        (esm2_8m_root, "main", "esm2_t6_8M_UR50D"),
        (esm2_3b_root, "main", "esm2_t36_3B_UR50D"),
        (length_root, "length_sensitivity", None),
        (control_root, "matched_n_control", None),
    )
    for root, run_group, model in bundle_contracts:
        loaded = load_bundle_partitions(
            root,
            specifications,
            run_group=run_group,
            model=model,
            repository_root=repository_root,
        )
        overlap = set(partitions).intersection(loaded)
        if overlap:
            raise ValueError(f"partition sources overlap: {sorted(overlap)}")
        partitions.update(loaded)
    agreement = partition_agreement_table(partitions)
    summary = length_control_summary(agreement)
    sources = comparison_source_context(
        source_grid_path=source_grid_path,
        prott5_assignments_dir=prott5_assignments_dir,
        bundle_roots=[contract[0] for contract in bundle_contracts],
        repository_root=repository_root,
    )
    return publish_comparison_outputs(
        agreement,
        summary,
        output_root=output_root,
        sources=sources,
        repository_root=repository_root,
    )


def _validate_fit_input(specification: RunSpecification, dataset: FitDataset) -> np.ndarray:
    if specification.embedding_model not in MODEL_WIDTHS:
        raise ValueError(f"unknown embedding model: {specification.embedding_model}")
    expected_space = f"raw{MODEL_WIDTHS[specification.embedding_model]}"
    if specification.space != expected_space:
        raise ValueError(
            f"only native {expected_space} coordinates are allowed; got {specification.space}"
        )
    if specification.metric not in METRICS:
        raise ValueError(f"unsupported metric: {specification.metric}")
    if specification.algorithm != METRIC_ALGORITHMS[specification.metric]:
        raise ValueError("run specification does not use the frozen metric algorithm")
    if specification.cluster_selection_method != "eom":
        raise ValueError("run specification does not use eom cluster selection")
    if dataset.embedding_model != specification.embedding_model:
        raise ValueError("fit dataset and run specification name different embedding models")
    if dataset.records != specification.records:
        raise ValueError("fit dataset and run specification name different record sets")
    if len(dataset.ids) != len(set(dataset.ids)):
        raise ValueError("fit dataset contains duplicate protein IDs")
    matrix = np.asarray(dataset.X)
    expected_shape = (len(dataset.ids), MODEL_WIDTHS[specification.embedding_model])
    if matrix.shape != expected_shape:
        raise ValueError(f"fit matrix shape is {matrix.shape}, expected {expected_shape}")
    if not np.isfinite(matrix).all():
        raise ValueError("fit matrix contains non-finite values")
    if specification.metric == "cosine":
        minimum_norm = float(np.linalg.norm(matrix, axis=1).min())
        if minimum_norm <= COSINE_NORM_FLOOR:
            raise ValueError(
                f"cosine input has minimum norm {minimum_norm}; "
                f"expected above {COSINE_NORM_FLOOR}"
            )
    return np.ascontiguousarray(matrix, dtype=np.float64)


def fit_one(
    specification: RunSpecification,
    dataset: FitDataset,
    *,
    estimator_factory: Callable[..., Any] = hdbscan.HDBSCAN,
) -> FitResult:
    matrix = _validate_fit_input(specification, dataset)
    model = estimator_factory(
        min_cluster_size=specification.min_cluster_size,
        min_samples=specification.min_samples,
        metric=specification.metric,
        algorithm=specification.algorithm,
        cluster_selection_method=specification.cluster_selection_method,
    )
    labels = np.asarray(model.fit_predict(matrix))
    cluster_ids = prott5_hdbscan._cluster_ids(labels)
    probabilities = np.asarray(model.probabilities_, dtype=float)
    persistence = prott5_hdbscan._cluster_persistence(
        model.cluster_persistence_, labels
    )
    membership_summary, cluster_membership = prott5_hdbscan._membership_probability(
        probabilities, labels
    )
    noise_summary = prott5_hdbscan._noise_percentage(labels)
    dbcv_summary, cluster_dbcv = prott5_hdbscan._dbcv(
        matrix, labels, specification.metric
    )
    silhouette_summary = prott5_hdbscan._silhouette(
        matrix, labels, specification.metric
    )
    persistence_status = prott5_hdbscan._persistence_status(
        prott5_hdbscan._maximum_exact_multiplicity(matrix),
        specification.min_samples,
    )

    cluster_rows = tuple(
        {
            "row": specification.run_id,
            "dataset": DATASET,
            "embedding_model": specification.embedding_model,
            "run_group": specification.run_group,
            "records": specification.records,
            "subset_hash": dataset.subset_hash,
            "space": specification.space,
            "metric": specification.metric,
            "min_cluster_size": specification.min_cluster_size,
            "min_samples": specification.min_samples,
            "cluster": int(cluster_id),
            "n": int((labels == cluster_id).sum()),
            "dbcv": cluster_dbcv[int(cluster_id)],
            "persistence": persistence[int(cluster_id)],
            "persistence_status": persistence_status,
            "mean_membership_probability": cluster_membership[int(cluster_id)],
        }
        for cluster_id in cluster_ids
    )
    n_clustered = int((labels != -1).sum())
    if sum(int(row["n"]) for row in cluster_rows) != n_clustered:
        raise ValueError("cluster sizes do not sum to the clustered record count")
    if any(int(row["n"]) < specification.min_cluster_size for row in cluster_rows):
        raise ValueError("fit returned a cluster below min_cluster_size")

    run_row = {
        "row": specification.run_id,
        "source_prott5_row": specification.source_row,
        "dataset": DATASET,
        "embedding_model": specification.embedding_model,
        "run_group": specification.run_group,
        "records": specification.records,
        "n": len(dataset.ids),
        "subset_hash": dataset.subset_hash,
        "subset_seed": specification.subset_seed,
        "space": specification.space,
        "n_dims": matrix.shape[1],
        "metric": specification.metric,
        "min_cluster_size": specification.min_cluster_size,
        "min_samples": specification.min_samples,
        "cluster_selection_method": str(model.cluster_selection_method),
        "algorithm": str(model.algorithm),
        "n_clusters": len(cluster_ids),
        **dbcv_summary,
        **silhouette_summary,
        **noise_summary,
        **membership_summary,
        "git_sha": prott5_hdbscan.GIT_SHA,
        **{
            name: version
            for name, version in prott5_hdbscan.LIBRARY_VERSIONS.items()
            if name in {
                "hdbscan_version",
                "numpy_version",
                "sklearn_version",
                "pandas_version",
            }
        },
    }
    return FitResult(
        run_row=run_row,
        cluster_rows=cluster_rows,
        protein_ids=dataset.ids,
        labels=np.asarray(labels, dtype=np.int32),
        probabilities=probabilities,
    )


def _validate_bundle_directory(
    root: Path, specifications: tuple[RunSpecification, ...]
) -> dict[str, Any]:
    paths = OutputPaths(
        root=root,
        runs=root / "metrics/hdbscan_runs.csv",
        clusters=root / "metrics/hdbscan_clusters.csv",
        assignments=root / "assignments",
        manifest=root / "run_manifest.json",
    )
    try:
        manifest = json.loads(paths.manifest.read_text(encoding="utf-8"))
        runs = pd.read_csv(paths.runs)
        clusters = pd.read_csv(paths.clusters)
    except (OSError, json.JSONDecodeError, pd.errors.ParserError) as error:
        raise ValueError(f"invalid HDBSCAN output bundle: {root}") from error
    if manifest.get("schema") != BUNDLE_SCHEMA:
        raise ValueError("HDBSCAN output manifest schema mismatch")

    expected_ids = [item.run_id for item in specifications]
    if manifest.get("fit_count") != len(specifications):
        raise ValueError("HDBSCAN output manifest fit count mismatch")
    if manifest.get("run_ids") != expected_ids:
        raise ValueError("HDBSCAN output manifest run IDs mismatch")
    if set(runs.columns) != set(RUN_COLUMNS):
        raise ValueError("HDBSCAN run table columns do not match the output contract")
    if set(clusters.columns) != set(CLUSTER_COLUMNS):
        raise ValueError("HDBSCAN cluster table columns do not match the output contract")
    if runs["row"].duplicated().any() or runs["row"].tolist() != expected_ids:
        raise ValueError("HDBSCAN run table does not contain the expected unique rows")
    if not set(clusters["row"]).issubset(expected_ids):
        raise ValueError("HDBSCAN cluster table names an unknown run")

    expected_assignment_names = {f"{run_id}.npz" for run_id in expected_ids}
    actual_assignment_names = {
        path.name for path in paths.assignments.glob("*.npz") if path.is_file()
    }
    if actual_assignment_names != expected_assignment_names:
        raise ValueError("HDBSCAN assignment files do not match the expected runs")

    metadata = runs.set_index("row")
    for specification in specifications:
        row = metadata.loc[specification.run_id]
        if not isinstance(row, pd.Series):
            raise ValueError("HDBSCAN run metadata must have unique row IDs")
        expected_metadata = {
            "source_prott5_row": specification.source_row,
            "embedding_model": specification.embedding_model,
            "run_group": specification.run_group,
            "records": specification.records,
            "space": specification.space,
            "metric": specification.metric,
            "min_cluster_size": specification.min_cluster_size,
            "min_samples": specification.min_samples,
            "cluster_selection_method": specification.cluster_selection_method,
            "algorithm": specification.algorithm,
        }
        for field, expected in expected_metadata.items():
            if row[field] != expected:
                raise ValueError(
                    f"HDBSCAN run {specification.run_id} has wrong {field}"
                )
        archive_path = paths.assignments / f"{specification.run_id}.npz"
        with np.load(archive_path, allow_pickle=False) as archive:
            if set(archive.files) != {"protein_id", "label", "membership_probability"}:
                raise ValueError(f"assignment keys are invalid for {specification.run_id}")
            protein_ids = np.asarray(archive["protein_id"]).astype(str)
            labels = np.asarray(archive["label"])
            probabilities = np.asarray(archive["membership_probability"], dtype=float)
        if protein_ids.ndim != 1 or len(protein_ids) != int(row["n"]):
            raise ValueError(f"assignment ID count is invalid for {specification.run_id}")
        if len(set(protein_ids.tolist())) != len(protein_ids):
            raise ValueError(f"assignment IDs are not unique for {specification.run_id}")
        prott5_hdbscan._cluster_ids(labels)
        prott5_hdbscan._membership_probability(probabilities, labels)
        if int((labels != -1).sum()) != int(row["n_clustered"]):
            raise ValueError(f"assignment clustered count is invalid for {specification.run_id}")
        run_clusters = clusters.loc[clusters["row"] == specification.run_id]
        expected_cluster_sizes = {
            int(cluster_id): int((labels == cluster_id).sum())
            for cluster_id in np.unique(labels[labels != -1])
        }
        actual_cluster_sizes = {
            _integer_field(cluster.cluster, "cluster"): _integer_field(cluster.n, "n")
            for cluster in run_clusters.itertuples(index=False)
        }
        if actual_cluster_sizes != expected_cluster_sizes:
            raise ValueError(f"cluster sizes are invalid for {specification.run_id}")

    recorded_files = manifest.get("files")
    if not isinstance(recorded_files, dict):
        raise ValueError("HDBSCAN output manifest has no file digest map")
    actual_files = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path != paths.manifest
    }
    if set(recorded_files) != actual_files:
        raise ValueError("HDBSCAN output manifest file list mismatch")
    for relative_path, expected_digest in recorded_files.items():
        if esm2_embeddings.sha256_file(root / relative_path) != expected_digest:
            raise ValueError(f"HDBSCAN output digest mismatch: {relative_path}")
    return manifest


def validate_bundle(
    output_root: Path,
    specifications: tuple[RunSpecification, ...],
    *,
    run_group: str,
    model: str | None = None,
    repository_root: Path = REPO,
) -> dict[str, Any]:
    paths = output_paths(
        output_root,
        run_group=run_group,
        model=model,
        repository_root=repository_root,
    )
    expected = bundle_specifications(
        specifications, run_group=run_group, model=model
    )
    return _validate_bundle_directory(paths.root, expected)


def _write_bundle(
    root: Path,
    specifications: tuple[RunSpecification, ...],
    results: list[FitResult],
    *,
    source_grid_path: Path,
    manifest_context: dict[str, Any],
) -> None:
    paths = OutputPaths(
        root=root,
        runs=root / "metrics/hdbscan_runs.csv",
        clusters=root / "metrics/hdbscan_clusters.csv",
        assignments=root / "assignments",
        manifest=root / "run_manifest.json",
    )
    paths.runs.parent.mkdir(parents=True)
    paths.assignments.mkdir(parents=True)
    pd.DataFrame(
        [result.run_row for result in results], columns=RUN_COLUMNS
    ).to_csv(paths.runs, index=False, lineterminator="\n", float_format="%.12g")
    pd.DataFrame(
        [row for result in results for row in result.cluster_rows],
        columns=CLUSTER_COLUMNS,
    ).to_csv(paths.clusters, index=False, lineterminator="\n", float_format="%.12g")
    for specification, result in zip(specifications, results, strict=True):
        np.savez_compressed(
            paths.assignments / f"{specification.run_id}.npz",
            protein_id=np.asarray(result.protein_ids, dtype=str),
            label=result.labels,
            membership_probability=result.probabilities,
        )
    files = {
        path.relative_to(root).as_posix(): esm2_embeddings.sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }
    manifest = {
        "schema": BUNDLE_SCHEMA,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_group": specifications[0].run_group,
        "fit_count": len(specifications),
        "run_ids": [item.run_id for item in specifications],
        "source_grid": {
            "path": str(Path(source_grid_path)),
            "sha256": esm2_embeddings.sha256_file(Path(source_grid_path)),
        },
        "context": manifest_context,
        "files": files,
    }
    paths.manifest.write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n"
    )


def run_and_publish_bundle(
    specifications: tuple[RunSpecification, ...],
    datasets: dict[tuple[str, str], FitDataset],
    *,
    run_group: str,
    output_root: Path,
    model: str | None = None,
    source_grid_path: Path = SOURCE_RUNS_PATH,
    manifest_context: dict[str, Any] | None = None,
    repository_root: Path = REPO,
    fit_function: Callable[[RunSpecification, FitDataset], FitResult] = fit_one,
) -> dict[str, Any]:
    paths = output_paths(
        output_root,
        run_group=run_group,
        model=model,
        repository_root=repository_root,
    )
    selected = bundle_specifications(
        specifications, run_group=run_group, model=model
    )
    missing = {
        (item.embedding_model, item.records)
        for item in selected
        if (item.embedding_model, item.records) not in datasets
    }
    if missing:
        raise ValueError(f"fit datasets are missing: {sorted(missing)}")

    results = [
        fit_function(item, datasets[(item.embedding_model, item.records)])
        for item in selected
    ]
    paths.root.parent.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex
    staged_root = paths.root.with_name(f".{paths.root.name}.{token}.tmp")
    backup_root = paths.root.with_name(f".{paths.root.name}.{token}.bak")
    promoted = False
    backed_up = False
    try:
        _write_bundle(
            staged_root,
            selected,
            results,
            source_grid_path=source_grid_path,
            manifest_context=manifest_context or {},
        )
        _validate_bundle_directory(staged_root, selected)
        if paths.root.exists():
            os.replace(paths.root, backup_root)
            backed_up = True
        os.replace(staged_root, paths.root)
        promoted = True
        manifest = _validate_bundle_directory(paths.root, selected)
        if backed_up:
            shutil.rmtree(backup_root)
        return manifest
    except BaseException:
        if promoted and paths.root.exists():
            shutil.rmtree(paths.root)
        if backed_up and backup_root.exists():
            os.replace(backup_root, paths.root)
        raise
    finally:
        if staged_root.exists():
            shutil.rmtree(staged_root)
        if backup_root.exists() and paths.root.exists():
            shutil.rmtree(backup_root)


def _load_model(
    *,
    h5_path: Path,
    model_output_root: Path,
    annotation_path: Path,
    fasta_path: Path,
    reference_h5: Path,
    repository_root: Path,
) -> esm2_analysis.ModelDataset:
    return esm2_analysis.load_model_dataset(
        h5_path=h5_path,
        output_root=model_output_root,
        annotation_path=annotation_path,
        fasta_path=fasta_path,
        reference_h5=reference_h5,
        repository_root=repository_root,
    )


def _load_reference(
    *, reference_h5: Path, annotation_path: Path, fasta_path: Path
) -> esm2_analysis.AnalysisDataset:
    return esm2_analysis.load_aligned_dataset(
        name="prott5",
        h5_path=reference_h5,
        annotation_path=annotation_path,
        fasta_path=fasta_path,
    )


def run_main_bundle(
    *,
    model_key: str,
    h5_path: Path,
    model_output_root: Path,
    annotation_path: Path,
    fasta_path: Path,
    reference_h5: Path,
    source_grid_path: Path,
    output_root: Path,
    repository_root: Path = REPO,
) -> dict[str, Any]:
    source_grid = read_source_grid(source_grid_path)
    specifications = build_run_specifications(source_grid)
    model = _load_model(
        h5_path=h5_path,
        model_output_root=model_output_root,
        annotation_path=annotation_path,
        fasta_path=fasta_path,
        reference_h5=reference_h5,
        repository_root=repository_root,
    )
    if model.model_key != model_key:
        raise ValueError(
            f"requested model {model_key} but the validated manifest names {model.model_key}"
        )
    datasets = main_fit_datasets(model)
    context = input_manifest_context(
        reference_h5=reference_h5,
        annotation_path=annotation_path,
        fasta_path=fasta_path,
        models=[model],
        repository_root=repository_root,
    )
    return run_and_publish_bundle(
        specifications,
        datasets,
        run_group="main",
        model=model.stem,
        output_root=output_root,
        source_grid_path=source_grid_path,
        manifest_context=context,
        repository_root=repository_root,
    )


def _load_model_pair(
    embedding_inputs: list[tuple[Path, Path]],
    *,
    annotation_path: Path,
    fasta_path: Path,
    reference_h5: Path,
    repository_root: Path,
) -> list[esm2_analysis.ModelDataset]:
    if len(embedding_inputs) != 2:
        raise ValueError("length sensitivity requires exactly two ESM-2 embedding inputs")
    models = [
        _load_model(
            h5_path=h5_path,
            model_output_root=model_output_root,
            annotation_path=annotation_path,
            fasta_path=fasta_path,
            reference_h5=reference_h5,
            repository_root=repository_root,
        )
        for h5_path, model_output_root in embedding_inputs
    ]
    by_stem = {model.stem: model for model in models}
    if set(by_stem) != set(ESM2_MODELS) or len(by_stem) != len(models):
        raise ValueError("embedding inputs must contain each registered ESM-2 model once")
    return [by_stem[stem] for stem in ESM2_MODELS]


def run_length_bundle(
    *,
    embedding_inputs: list[tuple[Path, Path]],
    annotation_path: Path,
    fasta_path: Path,
    reference_h5: Path,
    source_grid_path: Path,
    output_root: Path,
    repository_root: Path = REPO,
) -> dict[str, Any]:
    selected_ids = length_subset_ids(fasta_path)
    source_grid = read_source_grid(source_grid_path)
    specifications = build_run_specifications(source_grid)
    reference = _load_reference(
        reference_h5=reference_h5,
        annotation_path=annotation_path,
        fasta_path=fasta_path,
    )
    models = _load_model_pair(
        embedding_inputs,
        annotation_path=annotation_path,
        fasta_path=fasta_path,
        reference_h5=reference_h5,
        repository_root=repository_root,
    )
    datasets = length_fit_datasets(reference, models, selected_ids)
    context = input_manifest_context(
        reference_h5=reference_h5,
        annotation_path=annotation_path,
        fasta_path=fasta_path,
        models=models,
        repository_root=repository_root,
    )
    context["maximum_sequence_length"] = LENGTH_MAXIMUM
    return run_and_publish_bundle(
        specifications,
        datasets,
        run_group="length_sensitivity",
        output_root=output_root,
        source_grid_path=source_grid_path,
        manifest_context=context,
        repository_root=repository_root,
    )


def run_control_bundle(
    *,
    annotation_path: Path,
    fasta_path: Path,
    reference_h5: Path,
    source_grid_path: Path,
    output_root: Path,
    repository_root: Path = REPO,
) -> dict[str, Any]:
    full_ids = canonical_fasta_ids(fasta_path)
    controls = matched_n_control_subsets(
        full_ids,
        subset_size=LENGTH_SUBSET_COUNT,
        seeds=CONTROL_SEEDS,
    )
    source_grid = read_source_grid(source_grid_path)
    specifications = build_run_specifications(source_grid)
    reference = _load_reference(
        reference_h5=reference_h5,
        annotation_path=annotation_path,
        fasta_path=fasta_path,
    )
    if reference.ids != full_ids:
        raise ValueError("ProtT5 IDs do not match the canonical FASTA IDs")
    datasets = control_fit_datasets(reference, controls)
    context = input_manifest_context(
        reference_h5=reference_h5,
        annotation_path=annotation_path,
        fasta_path=fasta_path,
        models=[],
        repository_root=repository_root,
    )
    context["control_seeds"] = list(CONTROL_SEEDS)
    context["sampling"] = "numpy.default_rng choice without replacement from canonical IDs"
    return run_and_publish_bundle(
        specifications,
        datasets,
        run_group="matched_n_control",
        output_root=output_root,
        source_grid_path=source_grid_path,
        manifest_context=context,
        repository_root=repository_root,
    )


def _add_shared_input_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--annotation", type=Path, required=True)
    parser.add_argument("--fasta", type=Path, required=True)
    parser.add_argument("--reference-h5", type=Path, required=True)
    parser.add_argument("--source-runs", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)


def draw_comparison_figures(
    *,
    source_grid_path: Path,
    esm2_8m_root: Path,
    esm2_3b_root: Path,
    prott5_assignments_dir: Path,
    output_root: Path,
    repository_root: Path = REPO,
) -> dict[str, Any]:
    source_grid = read_source_grid(source_grid_path)
    specifications = build_run_specifications(source_grid)
    roots = {
        "esm2_t6_8M_UR50D": Path(esm2_8m_root),
        "esm2_t36_3B_UR50D": Path(esm2_3b_root),
    }
    for model, root in roots.items():
        validate_bundle(
            root,
            specifications,
            run_group="main",
            model=model,
            repository_root=repository_root,
        )
    validate_comparison_outputs(output_root, repository_root=repository_root)
    table = load_full_figure_runs(
        source_grid_path=source_grid_path,
        esm2_roots=roots,
    )
    comparison_root = comparison_output_paths(
        output_root, repository_root=repository_root
    ).root
    agreement = load_full_cross_model_agreement(
        comparison_root / "esm2_partition_agreement.csv"
    )
    partitions = load_prott5_partitions(
        source_grid,
        prott5_assignments_dir,
        repository_root=repository_root,
    )
    for model, root in roots.items():
        partitions.update(
            load_bundle_partitions(
                root,
                specifications,
                run_group="main",
                model=model,
                repository_root=repository_root,
            )
        )
    frequencies = noise_frequency_table(partitions)
    noise_path = draw_noise_figure(
        table,
        comparison_root / COMPARISON_FIGURES_DIR / NOISE_FIGURE_NAME,
    )
    cluster_count_path = draw_cluster_count_figure(
        table,
        comparison_root / COMPARISON_FIGURES_DIR / CLUSTER_COUNT_FIGURE_NAME,
    )
    agreement_path = draw_partition_agreement_figure(
        agreement,
        comparison_root / COMPARISON_FIGURES_DIR / AGREEMENT_FIGURE_NAME,
    )
    noise_frequency_path = draw_noise_frequency_figure(
        frequencies,
        comparison_root / COMPARISON_FIGURES_DIR / NOISE_FREQUENCY_FIGURE_NAME,
    )
    return {
        "source_rows": len(table),
        "figures": [
            noise_path.relative_to(comparison_root).as_posix(),
            cluster_count_path.relative_to(comparison_root).as_posix(),
            agreement_path.relative_to(comparison_root).as_posix(),
            noise_frequency_path.relative_to(comparison_root).as_posix(),
        ],
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    main_parser = commands.add_parser("run-main", help="run one ESM-2 native grid")
    _add_shared_input_arguments(main_parser)
    main_parser.add_argument("--model", choices=tuple(esm2_embeddings.MODEL_SPECS), required=True)
    main_parser.add_argument("--h5", type=Path, required=True)
    main_parser.add_argument("--model-output-root", type=Path, required=True)

    length_parser = commands.add_parser(
        "run-length", help="run the three-model length sensitivity"
    )
    _add_shared_input_arguments(length_parser)
    length_parser.add_argument(
        "--embedding",
        nargs=2,
        action="append",
        required=True,
        metavar=("H5", "MODEL_OUTPUT_ROOT"),
    )

    control_parser = commands.add_parser(
        "run-controls", help="run the five matched-n ProtT5 controls"
    )
    _add_shared_input_arguments(control_parser)

    validate_parser = commands.add_parser(
        "validate", help="reopen one completed output bundle"
    )
    validate_parser.add_argument("--source-runs", type=Path, required=True)
    validate_parser.add_argument("--output-root", type=Path, required=True)
    validate_parser.add_argument(
        "--bundle",
        choices=("8m-main", "3b-main", "length", "controls"),
        required=True,
    )
    compare_parser = commands.add_parser(
        "compare", help="compare every matching persisted partition"
    )
    compare_parser.add_argument("--source-runs", type=Path, required=True)
    compare_parser.add_argument("--prott5-assignments", type=Path, required=True)
    compare_parser.add_argument("--esm2-8m-root", type=Path, required=True)
    compare_parser.add_argument("--esm2-3b-root", type=Path, required=True)
    compare_parser.add_argument("--length-root", type=Path, required=True)
    compare_parser.add_argument("--control-root", type=Path, required=True)
    compare_parser.add_argument("--output-root", type=Path, required=True)
    validate_comparison_parser = commands.add_parser(
        "validate-comparison", help="reopen the completed comparison tables"
    )
    validate_comparison_parser.add_argument("--output-root", type=Path, required=True)
    draw_parser = commands.add_parser(
        "draw-figures", help="draw figures from validated persisted outputs"
    )
    draw_parser.add_argument("--source-runs", type=Path, required=True)
    draw_parser.add_argument("--esm2-8m-root", type=Path, required=True)
    draw_parser.add_argument("--esm2-3b-root", type=Path, required=True)
    draw_parser.add_argument("--prott5-assignments", type=Path, required=True)
    draw_parser.add_argument("--output-root", type=Path, required=True)
    arguments = parser.parse_args(argv)

    if arguments.command == "run-main":
        manifest = run_main_bundle(
            model_key=arguments.model,
            h5_path=arguments.h5,
            model_output_root=arguments.model_output_root,
            annotation_path=arguments.annotation,
            fasta_path=arguments.fasta,
            reference_h5=arguments.reference_h5,
            source_grid_path=arguments.source_runs,
            output_root=arguments.output_root,
        )
    elif arguments.command == "run-length":
        manifest = run_length_bundle(
            embedding_inputs=[
                (Path(h5_path), Path(model_output_root))
                for h5_path, model_output_root in arguments.embedding
            ],
            annotation_path=arguments.annotation,
            fasta_path=arguments.fasta,
            reference_h5=arguments.reference_h5,
            source_grid_path=arguments.source_runs,
            output_root=arguments.output_root,
        )
    elif arguments.command == "run-controls":
        manifest = run_control_bundle(
            annotation_path=arguments.annotation,
            fasta_path=arguments.fasta,
            reference_h5=arguments.reference_h5,
            source_grid_path=arguments.source_runs,
            output_root=arguments.output_root,
        )
    elif arguments.command == "validate":
        source_grid = read_source_grid(arguments.source_runs)
        specifications = build_run_specifications(source_grid)
        bundle_contract = {
            "8m-main": ("main", "esm2_t6_8M_UR50D"),
            "3b-main": ("main", "esm2_t36_3B_UR50D"),
            "length": ("length_sensitivity", None),
            "controls": ("matched_n_control", None),
        }
        run_group, model = bundle_contract[arguments.bundle]
        manifest = validate_bundle(
            arguments.output_root,
            specifications,
            run_group=run_group,
            model=model,
        )
    elif arguments.command == "compare":
        manifest = run_comparison(
            source_grid_path=arguments.source_runs,
            prott5_assignments_dir=arguments.prott5_assignments,
            esm2_8m_root=arguments.esm2_8m_root,
            esm2_3b_root=arguments.esm2_3b_root,
            length_root=arguments.length_root,
            control_root=arguments.control_root,
            output_root=arguments.output_root,
        )
    elif arguments.command == "draw-figures":
        result = draw_comparison_figures(
            source_grid_path=arguments.source_runs,
            esm2_8m_root=arguments.esm2_8m_root,
            esm2_3b_root=arguments.esm2_3b_root,
            prott5_assignments_dir=arguments.prott5_assignments,
            output_root=arguments.output_root,
        )
        print(
            f"HDBSCAN DRAW-FIGURES PASS: {result['source_rows']} source rows, "
            f"{len(result['figures'])} figures in {arguments.output_root}"
        )
        return
    else:
        manifest = validate_comparison_outputs(arguments.output_root)
    print(
        f"HDBSCAN {arguments.command.upper()} PASS: "
        f"{manifest.get('fit_count', manifest.get('agreement_rows'))} rows in "
        f"{arguments.output_root}"
    )


if __name__ == "__main__":
    main()
