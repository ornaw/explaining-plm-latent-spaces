from __future__ import annotations

import argparse
import shutil
import sys
from collections import Counter
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize

from src import esm2_analysis
from src.algo.kmeans import kmeans_standalone as km
from src.data import DATASETS

REPO = km.REPO
DATASET = km.DATASET
MODEL_KEY = "esm2_3b"
MODEL_STEM = "esm2_t36_3B_UR50D"
H5_PATH = REPO / "data/phosphatase/phosphatase/esm2_t36_3B_UR50D_full_length.h5"
MANIFEST_ROOT = REPO / "results/esm2/esm2_t36_3B_UR50D"
PROTT5_H5, ANNOTATION_PATH, FASTA_PATH = DATASETS[DATASET]
EXPECTED_N = 1_587
WIDTH = 2_560
RECORDS = "full"
GEOMETRIES = ("raw2560", "l2_raw2560")
GEOMETRY_CODES = {"raw2560": "R2560", "l2_raw2560": "L2R2560"}
PROTT5_GEOMETRY = {"raw2560": "raw1024", "l2_raw2560": "l2_raw1024"}
ROLE_CODES = km.ROLE_CODES
PROTT5_RUNS = REPO / "results/kmeans/metrics/kmeans_standalone_runs.csv"
PROTT5_ASSIGNMENTS = REPO / "results/kmeans/assignments/kmeans_standalone"

EXPECTED_SWEEP_ROWS = 980
EXPECTED_SELECTION_ROWS = 60
EXPECTED_PROCEDURE_PARTITIONS = 20
if (EXPECTED_SWEEP_ROWS != len(GEOMETRIES) * len(km.SEEDS) * len(km.K_VALUES)
        or EXPECTED_SELECTION_ROWS != len(GEOMETRIES) * len(km.SEEDS) * len(km.ENDPOINTS)
        or EXPECTED_PROCEDURE_PARTITIONS != len(GEOMETRIES) * len(km.SEEDS)):
    raise RuntimeError("the ESM-2 k-means axes no longer match the planned counts")
if int(round(np.sqrt(EXPECTED_N))) != km.PRIMARY_ENDPOINT:
    raise RuntimeError("the primary endpoint no longer equals round(sqrt(n))")


@dataclass(frozen=True)
class OutputPaths:
    root: Path
    sweep: Path
    selection: Path
    runs: Path
    clusters: Path
    agreement: Path
    assignments: Path
    elbow_figure: Path
    frequency_figure: Path
    stability_figure: Path


def output_paths(root: Path | None = None) -> OutputPaths:
    result_root = Path(root) if root is not None else REPO / "results/kmeans" / MODEL_STEM
    metrics, figures = result_root / "metrics", result_root / "figures"
    prefix = f"kmeans_{MODEL_STEM}"
    return OutputPaths(
        root=result_root,
        sweep=metrics / f"{prefix}_elbow_sweep.csv",
        selection=metrics / f"{prefix}_elbow_selection.csv",
        runs=metrics / f"{prefix}_runs.csv",
        clusters=metrics / f"{prefix}_clusters.csv",
        agreement=metrics / f"{prefix}_partition_agreement.csv",
        assignments=result_root / "assignments",
        elbow_figure=figures / f"{prefix}_elbow.png",
        frequency_figure=figures / f"{prefix}_selected_k_frequency.png",
        stability_figure=figures / f"{prefix}_stability.png",
    )


def load_esm2() -> esm2_analysis.ModelDataset:
    model = esm2_analysis.load_model_dataset(
        h5_path=H5_PATH,
        output_root=MANIFEST_ROOT,
        annotation_path=ANNOTATION_PATH,
        fasta_path=FASTA_PATH,
        reference_h5=PROTT5_H5,
        repository_root=REPO,
    )
    data = model.data
    if model.model_key != MODEL_KEY or model.width != WIDTH:
        raise ValueError(f"expected {MODEL_KEY} at width {WIDTH}, got {model.model_key}, {model.width}")
    if data.X.shape != (EXPECTED_N, WIDTH) or not np.isfinite(data.X).all():
        raise ValueError(f"expected finite {(EXPECTED_N, WIDTH)} coordinates, got {data.X.shape}")
    if list(data.ids) != sorted(data.ids) or len(set(data.ids)) != EXPECTED_N:
        raise ValueError("ESM-2 rows must be unique and in protein-ID order")
    return model


def spaces(X: np.ndarray) -> dict[str, np.ndarray]:
    raw = np.ascontiguousarray(X, dtype=np.float64)
    result = {"raw2560": raw, "l2_raw2560": normalize(raw, norm="l2", axis=1, copy=True)}
    for geometry, coordinates in result.items():
        if coordinates.shape != (EXPECTED_N, WIDTH) or not np.isfinite(coordinates).all():
            raise ValueError(f"bad coordinates for {geometry}")
    if not np.allclose(np.linalg.norm(result["l2_raw2560"], axis=1), 1.0):
        raise ValueError("unit-length coordinates are not unit length")
    return result


def partition_id(geometry: str, seed: int, k: int, role: str) -> str:
    if seed < 0 or k < 2:
        raise ValueError("partition IDs need a non-negative seed and K of at least two")
    return f"KM-E3B-{GEOMETRY_CODES[geometry]}-F-S{seed:02d}-K{k:02d}-{ROLE_CODES[role]}"


def partition_specs(selection: pd.DataFrame) -> list[dict[str, object]]:
    primary = selection[selection["is_primary"].astype(bool)]
    if len(primary) != EXPECTED_PROCEDURE_PARTITIONS or primary.duplicated(["geometry", "seed"]).any():
        raise ValueError("primary selections must hold one row per geometry and seed")
    modal = km._modal_k(primary)
    specs = []
    for row in primary.itertuples():
        specs.append({"geometry": row.geometry, "seed": int(row.seed), "k": int(row.selected_k),
                      "role": "procedure_selected", "selection_id": row.selection_id})
    for row in primary.itertuples():
        if int(row.selected_k) != modal[row.geometry]:
            specs.append({"geometry": row.geometry, "seed": int(row.seed), "k": modal[row.geometry],
                          "role": "fixed_k_stability", "selection_id": row.selection_id})
    return specs


def fit_and_score(X: np.ndarray, ids: list[str], spec: dict[str, object]
                  ) -> tuple[dict[str, object], pd.DataFrame, dict[str, np.ndarray]]:
    geometry, seed, k, role = spec["geometry"], int(spec["seed"]), int(spec["k"]), spec["role"]
    model, labels, warning_text = km._fit_partition(X, ids, k, seed)
    realised_k = km._validate_partition(model, labels, len(X))
    if not 2 <= realised_k < len(X):
        raise ValueError(f"internal scores are undefined at requested K {k}, realised {realised_k}")
    row = partition_id(geometry, seed, k, role)
    partition = {"row": row, "geometry": geometry, "records": RECORDS, "seed": seed,
                 "requested_k": k, "role": role}
    run = {
        **partition,
        "selection_id": spec["selection_id"],
        "n": len(X),
        "realised_k": realised_k,
        "cluster_count_shortfall": realised_k != k,
        "n_iter": int(model.n_iter_),
        "convergence_warning": warning_text,
        "inertia": float(model.inertia_),
        "silhouette_n_scored": len(X),
        "calinski_harabasz_n_scored": len(X),
        "davies_bouldin_n_scored": len(X),
        **km._internal_scores(X, labels),
        **km._control_summaries(X, labels, seed),
    }
    clusters = km._cluster_table(X, labels, model, partition)
    payload = {"protein_id": np.asarray(ids, dtype=str), "label": np.asarray(labels, dtype=np.int32)}
    return run, clusters, payload


def build_partitions(coordinates: dict[str, np.ndarray], ids: list[str],
                     specs: list[dict[str, object]], fit=fit_and_score
                     ) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict[str, np.ndarray]]]:
    runs, clusters, payloads = [], [], {}
    for spec in specs:
        run, cluster_table, payload = fit(coordinates[spec["geometry"]], ids, spec)
        if run["row"] in payloads:
            raise ValueError(f"duplicate partition {run['row']}")
        runs.append(run)
        clusters.append(cluster_table)
        payloads[run["row"]] = payload
    return pd.DataFrame(runs), pd.concat(clusters, ignore_index=True), payloads


def _prott5_procedure_rows() -> dict[tuple[str, int], str]:
    runs = pd.read_csv(PROTT5_RUNS)
    rows = runs[(runs["role"] == "procedure_selected") & (runs["records"] == "full")]
    keyed = {(str(r.geometry), int(r.seed)): str(r.row) for r in rows.itertuples()}
    if len(keyed) != len(rows):
        raise ValueError("ProtT5 procedure rows are not unique by geometry and seed")
    return keyed


def agreement_table(runs: pd.DataFrame, payloads: dict[str, dict[str, np.ndarray]],
                    prott5_assignments: Path = PROTT5_ASSIGNMENTS) -> pd.DataFrame:
    rows = []

    def add(comparison: str, geometry: str, left: str, right: str,
            left_payload: dict, right_payload: dict) -> None:
        n_common, adjusted_rand = km._aligned_ari(left_payload, right_payload)
        rows.append({"comparison": comparison, "geometry": geometry, "records": RECORDS,
                     "left_row": left, "right_row": right, "n_common": n_common,
                     "adjusted_rand": adjusted_rand})

    procedure = runs[runs["role"] == "procedure_selected"]
    for geometry in GEOMETRIES:
        group = procedure[procedure["geometry"] == geometry].sort_values("seed")
        for left, right in combinations(group["row"], 2):
            add("procedure_selected_stability", geometry, left, right, payloads[left], payloads[right])
        counts = Counter(group["requested_k"].astype(int))
        if len(counts) > 1:
            highest = max(counts.values())
            modal = min(k for k, count in counts.items() if count == highest)
            at_modal = runs[(runs["geometry"] == geometry) & (runs["requested_k"] == modal)]
            chosen = []
            for seed in sorted(group["seed"].astype(int)):
                seed_rows = at_modal[at_modal["seed"] == seed]
                if len(seed_rows) != 1:
                    raise ValueError(f"fixed-K set needs one {geometry} seed-{seed} partition at K {modal}")
                chosen.append(str(seed_rows.iloc[0]["row"]))
            for left, right in combinations(chosen, 2):
                add("fixed_k_stability", geometry, left, right, payloads[left], payloads[right])
    raw = procedure[procedure["geometry"] == "raw2560"].set_index("seed")["row"]
    unit = procedure[procedure["geometry"] == "l2_raw2560"].set_index("seed")["row"]
    if set(raw.index) != set(unit.index):
        raise ValueError("raw and unit-length runs must cover the same seeds")
    for seed in sorted(raw.index):
        add("raw_vs_l2", "raw_vs_l2", raw[seed], unit[seed], payloads[raw[seed]], payloads[unit[seed]])
    prott5 = _prott5_procedure_rows()
    for row in procedure.itertuples():
        other = prott5[(PROTT5_GEOMETRY[row.geometry], int(row.seed))]
        with np.load(prott5_assignments / f"{other}.npz", allow_pickle=False) as loaded:
            other_payload = {"protein_id": loaded["protein_id"].astype(str), "label": loaded["label"]}
        add("prott5_vs_esm2", row.geometry, other, row.row, other_payload, payloads[row.row])
    agreement = pd.DataFrame(rows)
    if agreement["adjusted_rand"].isna().any() or not agreement["adjusted_rand"].between(-1, 1).all():
        raise ValueError("adjusted Rand scores must be finite and between -1 and one")
    return agreement


def tables_with_provenance(sweep, selection, runs, clusters, agreement,
                           model: esm2_analysis.ModelDataset) -> dict[str, pd.DataFrame]:
    common = {
        "dataset": DATASET,
        "embedding": MODEL_STEM,
        "embedding_checkpoint": model.checkpoint,
        "embedding_revision": model.revision,
        "embedding_sha256": model.manifest["artifact"]["sha256"],
        "records": RECORDS,
        "n": len(model.data),
        "subset_hash": model.data.subset_hash,
        "git_sha": km._git_sha(),
        **km.LIBRARY_VERSIONS,
    }
    out = {}
    for name, table in (("sweep", sweep), ("selection", selection)):
        frame = table.copy()
        frame["space"] = frame["geometry"]
        frame["n_dims"] = WIDTH
        frame["input_normalisation"] = frame["geometry"].map(lambda g: "l2" if g.startswith("l2") else "none")
        out[name] = frame.assign(**common)
    out["sweep"]["endpoints"] = "|".join(str(endpoint) for endpoint in km.ENDPOINTS)
    frame = runs.drop(columns=["records", "n"]).copy()
    frame["space"] = frame["geometry"]
    frame["n_dims"] = WIDTH
    frame["input_normalisation"] = frame["geometry"].map(lambda g: "l2" if g.startswith("l2") else "none")
    frame["clustering_metric"] = "euclidean"
    frame["scoring_metric"] = "euclidean"
    frame["cluster_seed"] = frame["seed"]
    frame["control_permutations"] = km.CONTROL_PERMUTATIONS
    frame["kmeans_n_clusters"] = frame["requested_k"]
    frame["kmeans_random_state"] = frame["seed"]
    for argument, value in km.KMEANS_ARGUMENTS.items():
        frame[f"kmeans_{argument}"] = value
    frame["archive"] = frame["row"] + ".npz"
    out["runs"] = frame.assign(**common)
    out["clusters"] = clusters.drop(columns=["records"]).assign(**common)
    out["agreement"] = agreement.drop(columns=["records"]).assign(**common)
    return out


def reset_outputs(paths: OutputPaths) -> None:
    for directory in (paths.sweep.parent, paths.elbow_figure.parent, paths.assignments):
        if directory.exists():
            shutil.rmtree(directory)
        directory.mkdir(parents=True)


def validate_reopened(tables: dict[str, pd.DataFrame], payloads: dict, paths: OutputPaths,
                      ids: list[str]) -> dict[str, int]:
    sweep, selection, runs = tables["sweep"], tables["selection"], tables["runs"]
    clusters, agreement = tables["clusters"], tables["agreement"]
    if len(sweep) != EXPECTED_SWEEP_ROWS or sweep.duplicated(["geometry", "seed", "k"]).any():
        raise ValueError(f"expected {EXPECTED_SWEEP_ROWS} unique sweep rows, got {len(sweep)}")
    if len(selection) != EXPECTED_SELECTION_ROWS or selection["selection_id"].duplicated().any():
        raise ValueError(f"expected {EXPECTED_SELECTION_ROWS} selection rows, got {len(selection)}")
    procedure = runs[runs["role"] == "procedure_selected"]
    if len(procedure) != EXPECTED_PROCEDURE_PARTITIONS or runs["row"].duplicated().any():
        raise ValueError("expected one procedure partition per geometry and seed")
    if set(runs["row"]) != set(payloads) or not (runs["archive"] == runs["row"] + ".npz").all():
        raise ValueError("run rows, payloads and archive names do not match")
    if (runs["realised_k"] != runs["requested_k"]).any():
        raise ValueError("a partition realised fewer clusters than requested")
    archives = sorted(paths.assignments.glob("*.npz"))
    if len(archives) != len(runs):
        raise ValueError(f"expected {len(runs)} archives, found {len(archives)}")
    for row, expected in payloads.items():
        with np.load(paths.assignments / f"{row}.npz", allow_pickle=False) as loaded:
            if set(loaded.files) != {"protein_id", "label"} or loaded["label"].dtype != np.int32:
                raise ValueError(f"{row}.npz breaks the partition contract")
            if list(loaded["protein_id"].astype(str)) != list(ids):
                raise ValueError(f"{row}.npz does not hold the record set in protein-ID order")
            if not np.array_equal(loaded["label"], expected["label"]):
                raise ValueError(f"{row}.npz labels differ from the fitted partition")
    sizes = clusters.groupby("row")["size"].sum()
    if not sizes.reindex(runs["row"]).eq(runs.set_index("row")["n"]).all():
        raise ValueError("cluster sizes do not add up to n")
    per_comparison = agreement["comparison"].value_counts().to_dict()
    pairs = len(km.SEEDS) * (len(km.SEEDS) - 1) // 2
    if per_comparison.get("procedure_selected_stability") != pairs * len(GEOMETRIES):
        raise ValueError(f"expected {pairs} seed pairs per geometry")
    if per_comparison.get("raw_vs_l2") != len(km.SEEDS) or per_comparison.get("prott5_vs_esm2") != EXPECTED_PROCEDURE_PARTITIONS:
        raise ValueError("raw against unit-length or ESM-2 against ProtT5 rows are missing")
    if (agreement["n_common"] != EXPECTED_N).any():
        raise ValueError("every comparison must cover all 1,587 records")
    for name, table in tables.items():
        for column in ("dataset", "embedding_sha256", "subset_hash", "git_sha", "sklearn_version"):
            if table[column].isna().any():
                raise ValueError(f"provenance column {column} has gaps in {name}")
    return {"sweep_rows": len(sweep), "selection_rows": len(selection), "partitions": len(runs),
            "fixed_k": int((runs["role"] == "fixed_k_stability").sum()), "agreement_rows": len(agreement)}


def persist(tables: dict[str, pd.DataFrame], payloads: dict, paths: OutputPaths,
            ids: list[str]) -> dict[str, int]:
    reset_outputs(paths)
    table_paths = {"sweep": paths.sweep, "selection": paths.selection, "runs": paths.runs,
                   "clusters": paths.clusters, "agreement": paths.agreement}
    for name, path in table_paths.items():
        tables[name].to_csv(path, index=False)
    for row, payload in payloads.items():
        np.savez_compressed(paths.assignments / f"{row}.npz",
                            protein_id=np.asarray(payload["protein_id"], dtype=str),
                            label=np.asarray(payload["label"], dtype=np.int32))
    reopened = {name: pd.read_csv(path) for name, path in table_paths.items()}
    counts = validate_reopened(reopened, payloads, paths, ids)
    km._plot_elbow(reopened["sweep"], reopened["selection"], paths.elbow_figure)
    km._plot_selection_frequency(reopened["selection"], paths.frequency_figure)
    km._plot_stability(reopened["agreement"], paths.stability_figure)
    return counts


def run_scientific(root: Path | None = None) -> dict[str, int]:
    km._require_sklearn_version()
    paths = output_paths(root)
    reset_outputs(paths)
    print(f"Cleared {paths.root}; loading {MODEL_STEM}.", flush=True)
    model = load_esm2()
    ids = list(model.data.ids)
    coordinates = spaces(model.data.X)
    print(f"Starting the {EXPECTED_SWEEP_ROWS}-fit inertia sweep.", flush=True)
    sweep = km._sweep_table(coordinates, ids, progress=True)
    selection = km._selection_table(sweep)
    specs = partition_specs(selection)
    print(f"Fitting and scoring {len(specs)} partitions.", flush=True)
    runs, clusters, payloads = build_partitions(coordinates, ids, specs)
    agreement = agreement_table(runs, payloads)
    tables = tables_with_provenance(sweep, selection, runs, clusters, agreement, model)
    counts = persist(tables, payloads, paths, ids)
    print(f"Output validation passed: {counts}", flush=True)
    primary = selection[selection["is_primary"]]
    print("Modal K at endpoint 40:", km._modal_k(primary), flush=True)
    print(agreement.groupby(["comparison", "geometry"])["adjusted_rand"]
          .agg(["count", "median", "min", "max"]).round(3).to_string(), flush=True)
    return counts


def preflight() -> None:
    km._require_sklearn_version()
    model = load_esm2()
    coordinates = spaces(model.data.X)
    timings = {}
    for geometry, X in coordinates.items():
        started = perf_counter()
        km._fit_kmeans(X, list(model.data.ids), km.PRIMARY_ENDPOINT, 0)
        timings[geometry] = perf_counter() - started
    per_seed = sum(timings.values()) * len(km.K_VALUES)
    print(f"{MODEL_STEM}: {model.data.X.shape}, subset {model.data.subset_hash}")
    print(f"one K=40 fit: {', '.join(f'{g} {t:.1f}s' for g, t in timings.items())}")
    print(f"rough sweep estimate from K=40 alone: {per_seed * len(km.SEEDS) / 60:.0f} minutes")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-scientific", action="store_true",
                        help="run the whole experiment and replace results/kmeans/esm2_t36_3B_UR50D/")
    arguments = parser.parse_args(argv)
    if arguments.run_scientific:
        run_scientific()
    else:
        preflight()


if __name__ == "__main__":
    main()
