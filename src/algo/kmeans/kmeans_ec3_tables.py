from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import pandas as pd

from src.algo.agglomerative.agglomerative_ec_scores import ec3_tables, write
from src.algo.kmeans import kmeans_standalone as km
from src.algo.kmeans.kmeans_ec_scores import (
    KNOWN_MEDOIDS, load_partitions, medoid_seeds, partition_hash, sources,
)
from src.data import EC_PROTEINS, REPO, ec_table
from src.length_baseline import sequence_lengths

EMBEDDING, GEOMETRY, RECORDS = "prott5", "raw1024", "full"
OUT = REPO / "results/kmeans/ec3_tables"


def eligible(table: pd.DataFrame) -> tuple[dict, object, object]:
    source = next(s for s in sources() if s.embedding == EMBEDDING)
    loaded = load_partitions(source, table)
    medoids = medoid_seeds(source, pd.DataFrame([run for run, _, _ in loaded]))
    found = medoids.loc[(medoids["geometry"] == GEOMETRY) & (medoids["records"] == RECORDS), "medoid_seed"]
    if len(found) != 1 or int(found.iloc[0]) != KNOWN_MEDOIDS[(EMBEDDING, GEOMETRY, RECORDS)]:
        raise ValueError(f"medoid seed {found.tolist()}, recorded {KNOWN_MEDOIDS[(EMBEDDING, GEOMETRY, RECORDS)]}")
    seed = int(found.iloc[0])
    matches = [item for item in loaded if item[0]["geometry"] == GEOMETRY
               and item[0]["records"] == RECORDS and int(item[0]["seed"]) == seed]
    if len(matches) != 1:
        raise ValueError(f"{len(matches)} partitions for {GEOMETRY}/{RECORDS} seed {seed}")
    return matches[0]


def build(table: pd.DataFrame, lengths: pd.Series) -> dict[str, pd.DataFrame]:
    run, ids, labels = eligible(table)
    confusion, profile = ec3_tables(ids, labels, table, lengths)
    keys = {"row": run["row"], "seed": int(run["seed"]), "k": int(run["realised_k"]),
            "partition_sha256": partition_hash(ids, labels)}
    return {"kmeans_ec3_confusion.csv": confusion.assign(**keys),
            "kmeans_ec3_cluster_profile.csv": profile.assign(**keys)}


def run_scientific(out: Path = OUT) -> dict[str, int]:
    table, lengths = ec_table(), sequence_lengths()
    ec_sha = hashlib.sha256(EC_PROTEINS.read_bytes()).hexdigest()
    git = km._git_sha()
    frames = {name: frame.assign(ec_table_sha256=ec_sha, git_sha=git)
              for name, frame in build(table, lengths).items()}
    write(out, frames)
    print(f"git {git}; EC table sha256 {ec_sha[:12]}; rows { {n: len(f) for n, f in frames.items()} }")
    with pd.option_context("display.width", 220, "display.max_rows", 100):
        print(frames["kmeans_ec3_cluster_profile.csv"].drop(columns=["ec_table_sha256", "git_sha", "partition_sha256"]).to_string(index=False))
    return {n: len(f) for n, f in frames.items()}


def preflight() -> None:
    frames = build(ec_table(), sequence_lengths())
    profile = frames["kmeans_ec3_cluster_profile.csv"]
    print(f"{profile['row'].iloc[0]}: {len(profile)} clusters tabled in memory. Nothing written.")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").partition("\n")[0])
    parser.add_argument("--run-scientific", action="store_true", help="write the two EC3 tables")
    if parser.parse_args(argv).run_scientific:
        run_scientific()
    else:
        preflight()


if __name__ == "__main__":
    main()
