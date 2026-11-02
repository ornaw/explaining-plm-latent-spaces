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

from src.algo.agglomerative import agglomerative_standalone as ag
from src.algo.agglomerative import esm2_agglomerative_standalone as esm2_ag
from src.algo.kmeans.kmeans_ec_scores import (
    SUMMARY_MEASURES, ec_rows, partition_hash, subset_hash,
)
from src.data import EC_LEVELS, EC_PROTEINS, ec_table
from src.length_baseline import length_partition, sequence_lengths

ELIGIBLE_TREE = "AG-R1024-COC-F"
ELIGIBLE_ROW = "AG-R1024-COC-F-K15"
BEST_MATCH_MEASURE = "ami"
EXPECTED_SWEEP_CUTS = 49


@dataclass(frozen=True)
class Source:
    embedding: str
    prefix: str
    runs: Path
    clusters: Path
    assignments: Path
    out: Path
    expected_reported: int


def sources(prott5_root: Path | None = None, esm2_root: Path | None = None) -> tuple[Source, ...]:
    p = ag._output_paths(prott5_root)
    e = esm2_ag.output_paths(esm2_root)
    return (
        Source("prott5", "agglomerative", p.runs, p.clusters, p.assignments, p.root / "ec_scores", 18),
        Source(esm2_ag.MODEL_STEM, f"agglomerative_{esm2_ag.MODEL_STEM}", e.runs, e.clusters,
               e.assignments, e.root / "ec_scores", 3),
    )


def load_rows(source: Source, runs: pd.DataFrame, table: pd.DataFrame) -> list[tuple[dict, np.ndarray, np.ndarray]]:
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
        if len(values) != int(run["realised_k"]) or int(run["realised_k"]) != int(run["requested_k"]):
            raise ValueError(f"{run['row']}: {len(values)} clusters, {run['requested_k']} requested")
        recorded = clusters.loc[clusters["row"] == run["row"]]
        if dict(zip(values.tolist(), counts.tolist())) != dict(zip(recorded["cluster"], recorded["size"])):
            raise ValueError(f"{run['row']}: cluster sizes differ from the clusters table")
        loaded.append((run, ids, labels))
    return loaded


def reported_runs(source: Source) -> pd.DataFrame:
    runs = pd.read_csv(source.runs)
    runs = runs[runs["reported_partition"].astype(bool)].sort_values("row", kind="stable")
    if len(runs) != source.expected_reported or runs["row"].duplicated().any():
        raise ValueError(f"{source.embedding}: {len(runs)} reported rows, expected {source.expected_reported}")
    return runs


def sweep_runs(source: Source) -> pd.DataFrame:
    runs = pd.read_csv(source.runs)
    runs = runs[(runs["tree"] == ELIGIBLE_TREE) & (runs["role"] == "cut_sweep")].sort_values("requested_k")
    if list(runs["requested_k"]) != list(range(2, 2 + EXPECTED_SWEEP_CUTS)):
        raise ValueError(f"{ELIGIBLE_TREE}: the saved cuts are not K = 2 to 50")
    if ELIGIBLE_ROW not in set(runs["row"]):
        raise ValueError(f"{ELIGIBLE_ROW} is not in the sweep")
    return runs


def score_rows(source: Source, loaded, table: pd.DataFrame, lengths: pd.Series) -> pd.DataFrame:
    rows, baselines = [], {}
    for run, ids, labels in loaded:
        k = int(run["realised_k"])
        base = {"embedding": source.embedding, "row": run["row"], "tree": run["tree"],
                "geometry": run["geometry"], "linkage": run["linkage"], "distance": run["distance"],
                "records": run.get("records", "full"), "k": k, "n": int(run["n"]),
                "subset_hash": run["subset_hash"],
                "eligible_cross_method": bool(run["eligible_cross_method"]),
                "partition_sha256": partition_hash(ids, labels), "partition_git_sha": run["git_sha"]}
        key = (str(run["subset_hash"]), k)
        if key not in baselines:
            baselines[key] = ec_rows(ids, length_partition(ids, k, lengths), {}, table)
        for row, baseline in zip(ec_rows(ids, labels, base, table), baselines[key]):
            if row["level"] != baseline["level"] or row["n_scored"] != baseline["n_scored"]:
                raise ValueError(f"{run['row']}: baseline scored on other proteins")
            row.update({f"length_baseline_{m}": baseline[m] for m in SUMMARY_MEASURES})
            rows.append(row)
    return pd.DataFrame(rows)


def check_scores(scores: pd.DataFrame, expected_partitions: int) -> None:
    if len(scores) != len(EC_LEVELS) * expected_partitions:
        raise ValueError(f"{len(scores)} score rows, expected {len(EC_LEVELS) * expected_partitions}")
    if (scores["score_status"] != "scored").any():
        raise ValueError("a partition was not scorable")
    total = scores["n_scored"] + scores["n_no_class"] + scores["n_several_classes"]
    if not (total == scores["n_records"]).all() or not (scores["n_records"] == scores["n"]).all():
        raise ValueError("scored and excluded proteins do not add up to n")
    measures = [c for c in scores.columns if c in SUMMARY_MEASURES or c.endswith("_shuffled_mean")
                or c.startswith("length_baseline_")]
    if scores[measures].isna().any().any():
        raise ValueError("a score is missing")


def best_match(sweep: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for level, group in sweep.groupby("level", sort=True):
        if len(group) != EXPECTED_SWEEP_CUTS:
            raise ValueError(f"{level}: {len(group)} cuts")
        best = group.sort_values([BEST_MATCH_MEASURE, "k"], ascending=[False, True], kind="stable").iloc[0]
        rows.append({**best.to_dict(), "chosen_by": BEST_MATCH_MEASURE, "cuts_searched": len(group),
                     "chosen_with_labels": True,
                     "reported_k_ami": float(group.loc[group["row"] == ELIGIBLE_ROW, "ami"].iloc[0])})
    return pd.DataFrame(rows)


def ec3_tables(ids: np.ndarray, labels: np.ndarray, table: pd.DataFrame,
               lengths: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    ec3_n = table.loc[ids, "ec3_n"].to_numpy()
    ec3 = table.loc[ids, "ec3"].to_numpy(dtype=str)
    length = lengths.loc[ids].to_numpy()
    single = ec3_n == 1
    confusion = (pd.DataFrame({"cluster": labels[single], "ec3": ec3[single]})
                 .value_counts().rename("count").reset_index()
                 .sort_values(["cluster", "count", "ec3"], ascending=[True, False, True]))
    profile = []
    for cluster in np.unique(labels):
        member = labels == cluster
        classes = pd.Series(ec3[member & single]).value_counts()
        profile.append({
            "cluster": int(cluster), "size": int(member.sum()),
            "n_ec3_single_class": int((member & single).sum()),
            "n_no_ec3_class": int((member & (ec3_n == 0)).sum()),
            "n_several_ec3_classes": int((member & (ec3_n > 1)).sum()),
            "largest_ec3": str(classes.index[0]) if len(classes) else "",
            "largest_ec3_share": float(classes.iloc[0] / classes.sum()) if len(classes) else np.nan,
            "n_ec3_classes": int(len(classes)),
            "length_min": int(length[member].min()), "length_median": float(np.median(length[member])),
            "length_max": int(length[member].max()),
        })
    profile = pd.DataFrame(profile)
    if int(profile["size"].sum()) != len(ids) or int(confusion["count"].sum()) != int(single.sum()):
        raise ValueError("the EC3 tables do not add up to the partition")
    return confusion, profile


def with_provenance(frame: pd.DataFrame, git_sha: str, ec_sha: str) -> pd.DataFrame:
    return frame.assign(dataset=ag.DATASET, ec_table=EC_PROTEINS.relative_to(ag.REPO).as_posix(),
                        ec_table_sha256=ec_sha, shuffle_seeds="0-4", git_sha=git_sha,
                        **ag.LIBRARY_VERSIONS)


def write(out: Path, frames: dict[str, pd.DataFrame]) -> None:
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    for name, frame in frames.items():
        frame.to_csv(out / name, index=False)
    for name, frame in frames.items():
        numeric = frame.select_dtypes(include=np.number).columns
        text = [c for c in frame.columns if c not in numeric]
        back = pd.read_csv(out / name, dtype={c: str for c in text if frame[c].dtype != bool})
        if back.shape != frame.shape or list(back.columns) != list(frame.columns):
            raise ValueError(f"{name} did not reopen as written")
        if not np.allclose(back[numeric].to_numpy(float), frame[numeric].to_numpy(float),
                           rtol=0, atol=1e-12, equal_nan=True):
            raise ValueError(f"{name}: numbers changed on reopening")
        if not (back[text].fillna("").astype(str).to_numpy() == frame[text].fillna("").astype(str).to_numpy()).all():
            raise ValueError(f"{name}: text changed on reopening")


def build(table: pd.DataFrame, lengths: pd.Series, prott5_root: Path | None = None,
          esm2_root: Path | None = None) -> dict[str, dict[str, pd.DataFrame]]:
    out = {}
    for source in sources(prott5_root, esm2_root):
        scores = score_rows(source, load_rows(source, reported_runs(source), table), table, lengths)
        check_scores(scores, source.expected_reported)
        frames = {f"{source.prefix}_ec_scores.csv": scores}
        if source.embedding == "prott5":
            loaded = load_rows(source, sweep_runs(source), table)
            sweep = score_rows(source, loaded, table, lengths)
            check_scores(sweep, EXPECTED_SWEEP_CUTS)
            _, ids, labels = next(item for item in loaded if item[0]["row"] == ELIGIBLE_ROW)
            confusion, profile = ec3_tables(ids, labels, table, lengths)
            frames.update({f"{source.prefix}_ec_best_match_sweep.csv": sweep,
                           f"{source.prefix}_ec_best_match.csv": best_match(sweep),
                           f"{source.prefix}_ec3_confusion.csv": confusion.assign(row=ELIGIBLE_ROW),
                           f"{source.prefix}_ec3_cluster_profile.csv": profile.assign(row=ELIGIBLE_ROW)})
        out[source.embedding] = frames
    return out


def shown(tables: dict[str, dict[str, pd.DataFrame]]) -> str:
    lines = []
    for frames in tables.values():
        scores = next(f for n, f in frames.items() if n.endswith("_ec_scores.csv"))
        lines.append(scores[["embedding", "row", "level", "ami", "ari", "ami_shuffled_mean",
                             "length_baseline_ami", "length_baseline_ari", "n_scored",
                             "reference_largest_share"]].round(3).to_string(index=False))
        best = [f for n, f in frames.items() if n.endswith("_ec_best_match.csv")]
        if best:
            lines.append(best[0][["level", "row", "k", "ami", "ari", "length_baseline_ami",
                                  "reported_k_ami", "cuts_searched"]].round(3).to_string(index=False))
    return "\n".join(lines)


def run_scientific(prott5_root: Path | None = None, esm2_root: Path | None = None) -> dict[str, int]:
    ag._require_versions()
    table, lengths = ec_table(), sequence_lengths()
    ec_sha = hashlib.sha256(EC_PROTEINS.read_bytes()).hexdigest()
    git_sha = ag._git_sha()
    tables = build(table, lengths, prott5_root, esm2_root)
    counts = {}
    for source in sources(prott5_root, esm2_root):
        frames = {n: with_provenance(f, git_sha, ec_sha) for n, f in tables[source.embedding].items()}
        write(source.out, frames)
        counts.update({n: len(f) for n, f in frames.items()})
    print(f"git {git_sha}; EC table sha256 {ec_sha[:12]}; rows {counts}")
    with pd.option_context("display.width", 220):
        print(shown(tables))
    return counts


def preflight() -> None:
    table, lengths = ec_table(), sequence_lengths()
    for source in sources():
        loaded = load_rows(source, reported_runs(source), table)
        if source.embedding == "prott5":
            loaded_sweep = load_rows(source, sweep_runs(source), table)
            print(f"prott5: {len(loaded_sweep)} cuts of {ELIGIBLE_TREE} checked")
        rows = score_rows(source, loaded[:1], table, lengths)
        print(f"{source.embedding}: {len(loaded)} reported partitions checked; {loaded[0][0]['row']} "
              f"and its length baseline scored in memory ({len(rows)} rows). Nothing written.")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").partition("\n")[0])
    parser.add_argument("--run-scientific", action="store_true",
                        help="score the reported partitions and the best-match sweep, write the EC tables")
    arguments = parser.parse_args(argv)
    if arguments.run_scientific:
        run_scientific()
    else:
        preflight()


if __name__ == "__main__":
    main()
