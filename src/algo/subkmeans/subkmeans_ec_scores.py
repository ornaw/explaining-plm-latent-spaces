from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import pandas as pd

from src.algo.kmeans.kmeans_ec_scores import (
    SUMMARY_MEASURES, check_scores, ec_rows, partition_hash, primary_lines, subset_hash, summarise,
    write,
)
from src.data import EC_PROTEINS, REPO, ec_table
from src.length_baseline import length_partition, sequence_lengths

ROOT = REPO / "results/subkmeans"
ESM2_STEM = "esm2_t36_3B_UR50D"
ROLE = "procedure_selected"
EXPECTED_SKLEARN_VERSION = "1.9.0"
KNOWN_MEDOIDS = {
    ("prott5", "raw1024", "full"): 1,
    ("prott5", "l2_raw1024", "full"): 6,
    ("prott5", "raw1024", "deduplicated"): 0,
    ("prott5", "l2_raw1024", "deduplicated"): 7,
    (ESM2_STEM, "raw2560", "full"): 9,
}
LIBRARY_VERSIONS = {f"{name.replace('scikit-learn', 'sklearn')}_version": importlib.metadata.version(name)
                    for name in ("numpy", "pandas", "scikit-learn")}


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
    comparison: str
    left: str
    right: str


def sources(prott5_root: Path | None = None, esm2_root: Path | None = None) -> tuple[Source, ...]:
    p = ROOT if prott5_root is None else Path(prott5_root)
    e = ROOT / ESM2_STEM if esm2_root is None else Path(esm2_root)
    metrics = e / "metrics"
    return (
        Source("prott5", "subkmeans", p / "metrics/subkmeans_standalone_runs.csv",
               p / "metrics/subkmeans_standalone_clusters.csv",
               p / "metrics/subkmeans_standalone_partition_agreement.csv", p / "assignments",
               p / "ec_scores", 40, "procedure_selected_stability", "left_run", "right_run"),
        Source(ESM2_STEM, f"subkmeans_{ESM2_STEM}", metrics / f"subkmeans_{ESM2_STEM}_runs.csv",
               metrics / f"subkmeans_{ESM2_STEM}_clusters.csv",
               metrics / f"subkmeans_{ESM2_STEM}_partition_agreement.csv", e / "assignments",
               e / "ec_scores", 10, "subkmeans_seed_pairs", "left", "right"),
    )


def reported_runs(source: Source) -> pd.DataFrame:
    runs = pd.read_csv(source.runs)
    if "roles" in runs.columns:
        runs = runs[runs["roles"].astype(str).str.split().apply(lambda roles: ROLE in roles)]
    runs = runs.rename(columns={"run_id": "row", "outer_seed": "seed", "project_revision": "git_sha"})
    runs = runs.sort_values(["geometry", "records", "seed"], kind="stable").reset_index(drop=True)
    if len(runs) != source.expected_partitions or runs["row"].duplicated().any():
        raise ValueError(f"{source.embedding}: {len(runs)} reported rows, expected {source.expected_partitions}")
    if not (runs["returned_k"] == runs["requested_k"]).all():
        raise ValueError(f"{source.embedding}: a fit returned fewer clusters than requested")
    if runs.groupby(["geometry", "records"])["seed"].apply(lambda s: sorted(s) != list(range(10))).any():
        raise ValueError(f"{source.embedding}: an arm does not hold seeds 0 to 9 once each")
    return runs.assign(realised_k=runs["returned_k"])


def load_partitions(source: Source, table: pd.DataFrame) -> list[tuple[dict, np.ndarray, np.ndarray]]:
    runs = reported_runs(source)
    clusters = pd.read_csv(source.clusters)
    loaded = []
    for run in runs.to_dict("records"):
        with np.load(source.assignments / run["row"] / "partition.npz", allow_pickle=False) as archive:
            if not {"protein_id", "label"} <= set(archive.files):
                raise ValueError(f"{run['row']}: archive has no protein_id or label")
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
        recorded = clusters.loc[clusters["run_id"] == run["row"]]
        if dict(zip(values.tolist(), counts.tolist())) != dict(zip(recorded["label"], recorded["n"])):
            raise ValueError(f"{run['row']}: cluster sizes differ from the clusters table")
        loaded.append((run, ids, labels))
    return loaded


def medoid_seeds(source: Source, runs: pd.DataFrame) -> pd.DataFrame:
    agreement = pd.read_csv(source.agreement)
    agreement = agreement[agreement["comparison"] == source.comparison]
    seed_of = dict(zip(runs["row"], runs["seed"].astype(int)))
    rows = []
    for (geometry, records), group in runs.groupby(["geometry", "records"], sort=True):
        inside = agreement[source.left].isin(group["row"]) & agreement[source.right].isin(group["row"])
        pairs = agreement[inside]
        means = {}
        for seed in sorted(group["seed"].astype(int)):
            mine = pairs[(pairs[source.left].map(seed_of) == seed) | (pairs[source.right].map(seed_of) == seed)]
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


def git_sha() -> str:
    try:
        revision = subprocess.run(["git", "describe", "--always"], cwd=REPO, capture_output=True,
                                  text=True, check=True).stdout.strip()
        status = subprocess.run(["git", "status", "--porcelain", "--", ".", ":(exclude)results/subkmeans/**"],
                                cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return revision + ("-dirty" if status else "")


def require_sklearn_version() -> None:
    installed = importlib.metadata.version("scikit-learn")
    if installed != EXPECTED_SKLEARN_VERSION:
        raise RuntimeError(f"EC scores require scikit-learn {EXPECTED_SKLEARN_VERSION}, found {installed}")


def with_provenance(frame: pd.DataFrame, revision: str, ec_sha: str) -> pd.DataFrame:
    return frame.assign(dataset="phosphatase", ec_table=EC_PROTEINS.relative_to(REPO).as_posix(),
                        ec_table_sha256=ec_sha, shuffle_seeds="0-4", git_sha=revision,
                        **LIBRARY_VERSIONS)


def run_scientific(prott5_root: Path | None = None, esm2_root: Path | None = None) -> dict[str, int]:
    require_sklearn_version()
    table = ec_table()
    ec_sha = hashlib.sha256(EC_PROTEINS.read_bytes()).hexdigest()
    revision = git_sha()
    counts, shown = {}, []
    for source in sources(prott5_root, esm2_root):
        scores, _ = score(source, table)
        check_scores(scores, source)
        summary = summarise(scores)
        write(source, with_provenance(scores, revision, ec_sha), with_provenance(summary, revision, ec_sha))
        counts[source.embedding] = len(scores)
        shown.append(primary_lines(summary))
    print(f"git {revision}; EC table sha256 {ec_sha[:12]}; rows {counts}")
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
                        help="score all 50 reported partitions and write the EC tables")
    arguments = parser.parse_args(argv)
    if arguments.run_scientific:
        run_scientific()
    else:
        preflight()


if __name__ == "__main__":
    main()
