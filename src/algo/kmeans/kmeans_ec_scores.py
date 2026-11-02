from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import pandas as pd
from sklearn.metrics import (
    adjusted_mutual_info_score, adjusted_rand_score, normalized_mutual_info_score,
)

from src.algo.kmeans import esm2_kmeans_standalone as esm2_km
from src.algo.kmeans import kmeans_standalone as km
from src.data import EC_LEVELS, EC_PROTEINS, ec_table, external_scores
from src.length_baseline import length_partition, sequence_lengths

ROLE = "procedure_selected"
SHUFFLE_SEEDS = tuple(range(5))
SHUFFLE_MEASURES = ("ari", "nmi", "ami")
SUMMARY_MEASURES = ("ari", "nmi", "ami", "homogeneity", "completeness")
SHUFFLE_SCORE = {"ari": adjusted_rand_score, "nmi": normalized_mutual_info_score,
                 "ami": adjusted_mutual_info_score}
KNOWN_MEDOIDS = {
    ("prott5", "raw1024", "full"): 9,
    ("prott5", "l2_raw1024", "full"): 9,
    (esm2_km.MODEL_STEM, "raw2560", "full"): 5,
    (esm2_km.MODEL_STEM, "l2_raw2560", "full"): 0,
}


@dataclass(frozen=True)
class Source:
    embedding: str
    prefix: str
    runs: Path
    clusters: Path
    agreement: Path
    assignments: Path
    out: Path
    expected_partitions: int


def sources(prott5_root: Path | None = None, esm2_root: Path | None = None) -> tuple[Source, ...]:
    p = km._output_paths(prott5_root)
    e = esm2_km.output_paths(esm2_root)
    return (
        Source("prott5", "kmeans", p.runs, p.clusters, p.agreement, p.assignments,
               p.root / "ec_scores", 100),
        Source(esm2_km.MODEL_STEM, f"kmeans_{esm2_km.MODEL_STEM}", e.runs, e.clusters, e.agreement,
               e.assignments, e.root / "ec_scores", esm2_km.EXPECTED_PROCEDURE_PARTITIONS),
    )


def subset_hash(ids) -> str:
    return hashlib.sha1("\n".join(sorted(ids)).encode()).hexdigest()[:12]


def partition_hash(ids: np.ndarray, labels: np.ndarray) -> str:
    order = np.argsort(ids, kind="stable")
    text = "\n".join(f"{i}\t{int(c)}" for i, c in zip(ids[order], labels[order]))
    return hashlib.sha256(text.encode()).hexdigest()


def load_partitions(source: Source, table: pd.DataFrame) -> list[tuple[dict, np.ndarray, np.ndarray]]:
    runs = pd.read_csv(source.runs)
    runs = runs[runs["role"] == ROLE].sort_values(["geometry", "records", "seed"], kind="stable")
    if len(runs) != source.expected_partitions or runs["row"].duplicated().any():
        raise ValueError(f"{source.embedding}: {len(runs)} {ROLE} rows, expected {source.expected_partitions}")
    clusters = pd.read_csv(source.clusters)
    loaded = []
    for run in runs.to_dict("records"):
        with np.load(source.assignments / f"{run['row']}.npz", allow_pickle=False) as archive:
            if sorted(archive.files) != ["label", "protein_id"]:
                raise ValueError(f"{run['row']}: unexpected arrays {archive.files}")
            ids, labels = archive["protein_id"].astype(str), archive["label"].astype(np.int64)
        if ids.ndim != 1 or ids.shape != labels.shape or len(set(ids.tolist())) != len(ids):
            raise ValueError(f"{run['row']}: protein IDs and labels do not align one to one")
        if len(ids) != int(run["n"]) or subset_hash(ids) != str(run["subset_hash"]):
            raise ValueError(f"{run['row']}: archive IDs are not the recorded record set")
        if not set(ids.tolist()) <= set(table.index):
            raise ValueError(f"{run['row']}: protein IDs missing from the EC table")
        values, counts = np.unique(labels, return_counts=True)
        if len(values) != int(run["realised_k"]):
            raise ValueError(f"{run['row']}: {len(values)} clusters, {run['realised_k']} recorded")
        recorded = clusters.loc[clusters["row"] == run["row"]]
        if dict(zip(values.tolist(), counts.tolist())) != dict(zip(recorded["cluster"], recorded["size"])):
            raise ValueError(f"{run['row']}: cluster sizes differ from the clusters table")
        loaded.append((run, ids, labels))
    return loaded


def medoid_seeds(source: Source, runs: pd.DataFrame) -> pd.DataFrame:
    agreement = pd.read_csv(source.agreement)
    agreement = agreement[agreement["comparison"] == "procedure_selected_stability"]
    seed_of = dict(zip(runs["row"], runs["seed"].astype(int)))
    rows = []
    for (geometry, records), group in runs.groupby(["geometry", "records"], sort=True):
        pairs = agreement[agreement["left_row"].isin(group["row"]) & agreement["right_row"].isin(group["row"])]
        means = {}
        for seed in sorted(group["seed"].astype(int)):
            mine = pairs[(pairs["left_row"].map(seed_of) == seed) | (pairs["right_row"].map(seed_of) == seed)]
            if len(mine) != len(group) - 1:
                raise ValueError(f"{geometry}/{records} seed {seed}: {len(mine)} ARI pairs")
            means[seed] = float(mine["adjusted_rand"].mean())
        best = sorted(means, key=lambda s: (-means[s], s))[0]
        known = KNOWN_MEDOIDS.get((source.embedding, geometry, records))
        if known is not None and known != best:
            raise ValueError(f"{geometry}/{records}: medoid seed {best}, recorded {known}")
        rows.append({"embedding": source.embedding, "geometry": geometry, "records": records,
                     "medoid_seed": best, "medoid_mean_ari": means[best]})
    return pd.DataFrame(rows)


def ec_rows(ids: np.ndarray, labels: np.ndarray, base: dict, table: pd.DataFrame) -> list[dict]:
    rows = []
    for level in EC_LEVELS:
        row = {**base, **external_scores(ids, labels, level, table=table)}
        scored = table.loc[ids, f"ec{level}_n"].to_numpy() == 1
        classes = table.loc[ids[scored], f"ec{level}"].to_numpy(dtype=str)
        clusters = labels[scored]
        shares = pd.Series(classes).value_counts(normalize=True)
        row["reference_largest_class"] = str(shares.index[0]) if len(shares) else ""
        row["reference_largest_share"] = float(shares.iloc[0]) if len(shares) else np.nan
        for measure in SHUFFLE_MEASURES:
            if row["score_status"] != "scored":
                row[f"{measure}_shuffled_mean"] = np.nan
                continue
            row[f"{measure}_shuffled_mean"] = float(np.mean([
                SHUFFLE_SCORE[measure](np.random.default_rng(seed).permutation(classes), clusters)
                for seed in SHUFFLE_SEEDS
            ]))
        rows.append(row)
    return rows


def score(source: Source, table: pd.DataFrame, loaded=None,
          lengths: pd.Series | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    loaded = load_partitions(source, table) if loaded is None else loaded
    lengths = sequence_lengths() if lengths is None else lengths
    baselines: dict[tuple[str, int], list[dict]] = {}
    runs = pd.DataFrame([run for run, _, _ in loaded])
    medoids = medoid_seeds(source, runs)
    medoid = {(m["geometry"], m["records"]): m["medoid_seed"] for m in medoids.to_dict("records")}
    rows = []
    for run, ids, labels in loaded:
        base = {"embedding": source.embedding, "row": run["row"], "geometry": run["geometry"],
                "records": run["records"], "seed": int(run["seed"]), "k": int(run["realised_k"]),
                "n": int(run["n"]), "subset_hash": run["subset_hash"],
                "medoid_seed": int(run["seed"]) == medoid[(run["geometry"], run["records"])],
                "partition_sha256": partition_hash(ids, labels), "partition_git_sha": run["git_sha"]}
        key = (str(run["subset_hash"]), int(run["realised_k"]))
        if key not in baselines:
            baselines[key] = ec_rows(ids, length_partition(ids, key[1], lengths), {}, table)
        for row, baseline in zip(ec_rows(ids, labels, base, table), baselines[key]):
            if row["level"] != baseline["level"] or row["n_scored"] != baseline["n_scored"]:
                raise ValueError(f"{run['row']}: baseline scored on other proteins")
            row.update({f"length_baseline_{m}": baseline[m] for m in SUMMARY_MEASURES})
            rows.append(row)
    return pd.DataFrame(rows), medoids


def check_scores(scores: pd.DataFrame, source: Source) -> None:
    if len(scores) != len(EC_LEVELS) * source.expected_partitions:
        raise ValueError(f"{source.embedding}: {len(scores)} score rows")
    if (scores["score_status"] != "scored").any():
        raise ValueError(f"{source.embedding}: a partition was not scorable")
    total = scores["n_scored"] + scores["n_no_class"] + scores["n_several_classes"]
    if not (total == scores["n_records"]).all() or not (scores["n_records"] == scores["n"]).all():
        raise ValueError(f"{source.embedding}: scored and excluded proteins do not add up to n")
    if not (scores["n_clusters"] <= scores["k"]).all():
        raise ValueError(f"{source.embedding}: more clusters among scored proteins than K")
    columns = [*SUMMARY_MEASURES, *(f"{m}_shuffled_mean" for m in SHUFFLE_MEASURES),
               *(f"length_baseline_{m}" for m in SUMMARY_MEASURES)]
    if scores[columns].isna().any().any():
        raise ValueError(f"{source.embedding}: a score is missing")


def summarise(scores: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["embedding", "geometry", "records", "level"]
    for key, group in scores.groupby(keys, sort=True):
        medoid = group[group["medoid_seed"]]
        if len(medoid) != 1:
            raise ValueError(f"{key}: {len(medoid)} medoid rows")
        for measure in SUMMARY_MEASURES:
            values = group[measure]
            row = dict(zip(keys, key), measure=measure, n_partitions=len(group),
                       median=float(values.median()), minimum=float(values.min()),
                       maximum=float(values.max()), medoid_seed=int(medoid["seed"].iloc[0]),
                       medoid_value=float(medoid[measure].iloc[0]),
                       k_values=" ".join(str(k) for k in sorted(group["k"])),
                       n_scored_min=int(group["n_scored"].min()), n_scored_max=int(group["n_scored"].max()),
                       reference_classes=int(group["n_reference_classes"].iloc[0]),
                       reference_largest_class=group["reference_largest_class"].iloc[0],
                       reference_largest_share=float(group["reference_largest_share"].iloc[0]))
            baseline = group[f"length_baseline_{measure}"]
            row.update(length_baseline_median=float(baseline.median()),
                       length_baseline_minimum=float(baseline.min()),
                       length_baseline_maximum=float(baseline.max()),
                       n_at_or_below_length_baseline=int((values <= baseline).sum()))
            if measure in SHUFFLE_MEASURES:
                shuffled = group[f"{measure}_shuffled_mean"]
                row["shuffled_mean_largest_magnitude"] = float(shuffled.abs().max())
                row["n_at_or_below_shuffled"] = int((values <= shuffled).sum())
            rows.append(row)
    return pd.DataFrame(rows)


def with_provenance(frame: pd.DataFrame, git_sha: str, ec_sha: str) -> pd.DataFrame:
    return frame.assign(dataset=km.DATASET, ec_table=EC_PROTEINS.relative_to(km.REPO).as_posix(),
                        ec_table_sha256=ec_sha, shuffle_seeds="0-4", git_sha=git_sha,
                        **km.LIBRARY_VERSIONS)


def write(source: Source, scores: pd.DataFrame, summary: pd.DataFrame) -> None:
    if source.out.exists():
        shutil.rmtree(source.out)
    source.out.mkdir(parents=True)
    paths = {source.out / f"{source.prefix}_ec_scores.csv": scores,
             source.out / f"{source.prefix}_ec_summary.csv": summary}
    for path, frame in paths.items():
        frame.to_csv(path, index=False)
    for path, frame in paths.items():
        text = [c for c in frame.columns if c not in frame.select_dtypes(include=np.number).columns]
        back = pd.read_csv(path, dtype={c: str for c in text if frame[c].dtype != bool})
        if back.shape != frame.shape or list(back.columns) != list(frame.columns):
            raise ValueError(f"{path.name} did not reopen as written")
        numeric = frame.select_dtypes(include=np.number).columns
        if not np.allclose(back[numeric].to_numpy(float), frame[numeric].to_numpy(float),
                           rtol=0, atol=1e-12, equal_nan=True):
            raise ValueError(f"{path.name}: numbers changed on reopening")
        other = [c for c in frame.columns if c not in numeric]
        if not (back[other].fillna("").astype(str) == frame[other].fillna("").astype(str)).all().all():
            raise ValueError(f"{path.name}: text changed on reopening")


def primary_lines(summary: pd.DataFrame) -> pd.DataFrame:
    shown = summary[summary["measure"].isin(["ari", "ami"])]
    return shown[["embedding", "geometry", "records", "level", "measure", "median", "minimum", "maximum",
                  "medoid_value", "shuffled_mean_largest_magnitude", "n_at_or_below_shuffled",
                  "length_baseline_median", "n_at_or_below_length_baseline", "n_scored_min",
                  "reference_largest_share"]].round(3)


def run_scientific(prott5_root: Path | None = None, esm2_root: Path | None = None) -> dict[str, int]:
    km._require_sklearn_version()
    table = ec_table()
    ec_sha = hashlib.sha256(EC_PROTEINS.read_bytes()).hexdigest()
    git_sha = km._git_sha()
    counts, shown = {}, []
    for source in sources(prott5_root, esm2_root):
        scores, _ = score(source, table)
        check_scores(scores, source)
        summary = summarise(scores)
        write(source, with_provenance(scores, git_sha, ec_sha), with_provenance(summary, git_sha, ec_sha))
        counts[source.embedding] = len(scores)
        shown.append(primary_lines(summary))
    print(f"git {git_sha}; EC table sha256 {ec_sha[:12]}; rows {counts}")
    with pd.option_context("display.width", 220, "display.max_rows", 200):
        print(pd.concat(shown).to_string(index=False))
    return counts


def preflight(prott5_root: Path | None = None, esm2_root: Path | None = None) -> None:
    table = ec_table()
    for source in sources(prott5_root, esm2_root):
        loaded = load_partitions(source, table)
        runs = pd.DataFrame([run for run, _, _ in loaded])
        medoids = medoid_seeds(source, runs)
        run, ids, labels = loaded[0]
        rows = ec_rows(ids, labels, {"row": run["row"]}, table)
        rows += ec_rows(ids, length_partition(ids, int(run["realised_k"])), {}, table)
        print(f"{source.embedding}: {len(loaded)} partitions checked; medoid seeds "
              f"{dict(zip(medoids['geometry'] + '/' + medoids['records'], medoids['medoid_seed']))}; "
              f"{run['row']} and its length baseline scored in memory ({len(rows)} rows). Nothing written.")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").partition("\n")[0])
    parser.add_argument("--run-scientific", action="store_true",
                        help="score all 120 partitions and write the EC tables")
    arguments = parser.parse_args(argv)
    if arguments.run_scientific:
        run_scientific()
    else:
        preflight()


if __name__ == "__main__":
    main()
