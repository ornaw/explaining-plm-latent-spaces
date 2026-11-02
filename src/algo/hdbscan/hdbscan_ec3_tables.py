from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import pandas as pd

from src.algo.agglomerative.agglomerative_ec_scores import ec3_tables, write
from src.algo.hdbscan.hdbscan_ec_scores import (
    GRID, HD, OUT as EC_SCORES, check, git_sha, read_archive, subset_hash,
)
from src.algo.kmeans.kmeans_ec_scores import partition_hash
from src.data import EC_PROTEINS, ec_table
from src.length_baseline import sequence_lengths

ROW = "CA0505"
ARCHIVE = GRID / "CA0505_full_raw1024.npz"
RUNS = HD / "metrics/hdbscan_standalone_runs.csv"
NOISE_CLASSES = EC_SCORES / "hdbscan_ec_noise_classes.csv"
OUT = HD / "ec3_tables"


def reported() -> tuple[np.ndarray, np.ndarray]:
    run = pd.read_csv(RUNS).set_index("row").loc[ROW]
    ids, labels = read_archive(ARCHIVE)
    check(ids, labels, int(run.n_clusters), int(run.n_noise), int(run.n), ROW)
    if subset_hash(ids) != str(run.subset_hash):
        raise ValueError(f"{ROW}: archive IDs are not the recorded record set")
    return ids, labels


def check_noise(confusion: pd.DataFrame, profile: pd.DataFrame) -> None:
    other = pd.read_csv(NOISE_CLASSES)
    other = other[(other["partition"] == ROW) & (other["level"] == "EC3")].set_index("class")["n_noise"]
    noise = profile.loc[profile["cluster"] == -1].iloc[0]
    if int(other.get("no class", 0)) != int(noise["n_no_ec3_class"]) \
            or int(other.get("several classes", 0)) != int(noise["n_several_ec3_classes"]):
        raise ValueError("noise row: no-class or several-class counts differ from the EC scorer's")
    single = other.drop(["no class", "several classes"], errors="ignore")
    single = single[single > 0].sort_index()
    mine = confusion.loc[confusion["cluster"] == -1].set_index("ec3")["count"].sort_index()
    if not single.astype(int).equals(mine.astype(int)):
        raise ValueError("noise row: EC3 class counts differ from the EC scorer's")


def build(table: pd.DataFrame, lengths: pd.Series) -> dict[str, pd.DataFrame]:
    ids, labels = reported()
    confusion, profile = ec3_tables(ids, labels, table, lengths)
    check_noise(confusion, profile)
    keys = {"row": ROW, "partition_sha256": partition_hash(ids, labels)}
    return {"hdbscan_ec3_confusion.csv": confusion.assign(noise=confusion["cluster"] == -1, **keys),
            "hdbscan_ec3_cluster_profile.csv": profile.assign(noise=profile["cluster"] == -1, **keys)}


def run_scientific(out: Path = OUT) -> dict[str, int]:
    table, lengths = ec_table(), sequence_lengths()
    ec_sha = hashlib.sha256(EC_PROTEINS.read_bytes()).hexdigest()
    git = git_sha()
    frames = {name: frame.assign(ec_table_sha256=ec_sha, git_sha=git)
              for name, frame in build(table, lengths).items()}
    write(out, frames)
    print(f"git {git}; EC table sha256 {ec_sha[:12]}; rows { {n: len(f) for n, f in frames.items()} }")
    with pd.option_context("display.width", 220, "display.max_rows", 120):
        print(frames["hdbscan_ec3_cluster_profile.csv"].drop(columns=["ec_table_sha256", "git_sha", "partition_sha256"]).to_string(index=False))
    return {n: len(f) for n, f in frames.items()}


def preflight() -> None:
    frames = build(ec_table(), sequence_lengths())
    profile = frames["hdbscan_ec3_cluster_profile.csv"]
    print(f"{ROW}: {int((~profile['noise']).sum())} clusters and a noise row tabled in memory. Nothing written.")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").partition("\n")[0])
    parser.add_argument("--run-scientific", action="store_true", help="write the two EC3 tables")
    if parser.parse_args(argv).run_scientific:
        run_scientific()
    else:
        preflight()


if __name__ == "__main__":
    main()
