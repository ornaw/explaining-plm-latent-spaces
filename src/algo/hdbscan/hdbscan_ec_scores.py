from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import pandas as pd
from sklearn.metrics import (
    adjusted_mutual_info_score, adjusted_rand_score, normalized_mutual_info_score,
)

from src.algo.kmeans.kmeans_ec_scores import partition_hash, subset_hash
from src.data import EC_LEVELS, EC_PROTEINS, REPO, ec_table, external_scores
from src.length_baseline import length_partition, sequence_lengths

DATASET = "phosphatase"
HD = REPO / "results/hdbscan"
GRID = HD / "assignments/hdbscan_standalone_grid"
ESM = HD / "esm2_t36_3B_UR50D"
ORDERS = HD / "reported_orders"
OUT = HD / "ec_scores"
SHUFFLE_SEEDS = tuple(range(5))
SHUFFLE = {"ari": adjusted_rand_score, "nmi": normalized_mutual_info_score, "ami": adjusted_mutual_info_score}
MEASURES = ("ari", "nmi", "ami", "homogeneity", "completeness")
E_COS = "esm2_t36_3B_UR50D_full_raw2560_cosine_mcs05_ms05"
E_EUC = "esm2_t36_3B_UR50D_full_raw2560_euclidean_mcs05_ms05"
SETTINGS = (
    ("prott5", "CA0505", "reported", "full", GRID / "CA0505_full_raw1024.npz", HD / "metrics/hdbscan_standalone_runs.csv", True),
    ("prott5", "EA0505", "sensitivity", "full", GRID / "EA0505_full_raw1024.npz", HD / "metrics/hdbscan_standalone_runs.csv", True),
    ("prott5", "CE0505", "sensitivity, 1,480 records", "deduplicated", GRID / "CE0505_deduplicated_raw1024.npz", HD / "metrics/hdbscan_standalone_runs.csv", False),
    ("prott5", "EE0505", "sensitivity, 1,480 records", "deduplicated", GRID / "EE0505_deduplicated_raw1024.npz", HD / "metrics/hdbscan_standalone_runs.csv", False),
    ("esm2_t36_3B_UR50D", E_COS, "reported", "full", ESM / f"assignments/{E_COS}.npz", ESM / "metrics/hdbscan_runs.csv", True),
    ("esm2_t36_3B_UR50D", E_EUC, "sensitivity", "full", ESM / f"assignments/{E_EUC}.npz", ESM / "metrics/hdbscan_runs.csv", True),
)
EXPECTED_PARTITIONS = 86
LIBRARY_VERSIONS = {f"{name}_version": importlib.metadata.version(pkg) for pkg, name in
                    (("numpy", "numpy"), ("scikit-learn", "sklearn"), ("pandas", "pandas"))}


def git_sha() -> str:
    try:
        sha = subprocess.run(["git", "describe", "--always"], cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()
        status = subprocess.run(["git", "status", "--porcelain", "--", ".", ":(exclude)results/hdbscan/**"],
                                cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return f"{sha}-dirty" if status else sha


def read_archive(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        return z["protein_id"].astype(str), z["label"].astype(np.int64)


def check(ids: np.ndarray, labels: np.ndarray, n_clusters: int, n_noise: int, n: int, name: str) -> None:
    if ids.ndim != 1 or ids.shape != labels.shape or len(ids) != n or len(set(ids.tolist())) != n:
        raise ValueError(f"{name}: IDs and labels do not align, or n is not {n}")
    if list(ids) != sorted(ids):
        raise ValueError(f"{name}: rows are not in protein-ID order")
    k = int(labels.max()) + 1
    if not set(np.unique(labels).tolist()) <= set(range(-1, k)) or len(np.unique(labels[labels >= 0])) != k:
        raise ValueError(f"{name}: labels are not -1 and 0 to K-1")
    if k != n_clusters or int((labels == -1).sum()) != n_noise:
        raise ValueError(f"{name}: {k} clusters and {(labels == -1).sum()} noise, recorded {n_clusters} and {n_noise}")


def partitions() -> list[tuple[dict, np.ndarray, np.ndarray]]:
    orders = pd.read_csv(ORDERS / "hdbscan_reported_orders.csv")
    out = []
    for embedding, row, role, records, archive, runs_path, has_orders in SETTINGS:
        runs = pd.read_csv(runs_path).set_index("row")
        r = runs.loc[row]
        ids, labels = read_archive(archive)
        check(ids, labels, int(r.n_clusters), int(r.n_noise), int(r.n), row)
        if subset_hash(ids) != str(r.subset_hash):
            raise ValueError(f"{row}: archive IDs are not the recorded record set")
        base = {"embedding": embedding, "stored_row": row, "role": role, "records": records,
                "subset_hash": str(r.subset_hash), "git_sha_partition": str(r.git_sha)}
        out.append(({**base, "partition": row, "order_seed": -1}, ids, labels))
        if has_orders:
            mine = orders[(orders.stored_row == row) & (orders.order_seed >= 0)].sort_values("order_seed")
            if len(mine) != 20 or not mine.canonical_refit_same_as_stored.all():
                raise ValueError(f"{row}: reorders missing or not anchored to the stored partition")
            for o in mine.itertuples():
                oids, olab = read_archive(ORDERS / "assignments" / f"{o.partition}.npz")
                check(oids, olab, int(o.n_clusters), int(o.n_noise), len(ids), o.partition)
                if not np.array_equal(oids, ids):
                    raise ValueError(f"{o.partition}: protein IDs differ from the stored partition")
                out.append(({**base, "partition": o.partition, "order_seed": int(o.order_seed),
                             "git_sha_partition": str(o.git_sha)}, oids, olab))
    if len(out) != EXPECTED_PARTITIONS:
        raise ValueError(f"{len(out)} partitions, expected {EXPECTED_PARTITIONS}")
    return out


def baseline_labels(ids: np.ndarray, labels: np.ndarray, lengths: pd.Series) -> np.ndarray:
    clustered = labels != -1
    out = np.full(len(labels), -1, dtype=np.int64)
    out[clustered] = length_partition(ids[clustered], int(labels.max()) + 1, lengths)
    return out


def shuffled_means(classes: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    return {m: float(np.mean([f(np.random.default_rng(s).permutation(classes), predicted) for s in SHUFFLE_SEEDS]))
            for m, f in SHUFFLE.items()}


def ec_rows(ids: np.ndarray, labels: np.ndarray, base: dict, table: pd.DataFrame, lengths: pd.Series) -> list[dict]:
    rows = []
    blabels = baseline_labels(ids, labels, lengths)
    for level in EC_LEVELS:
        row = {**base, **external_scores(ids, labels, level, noise_label=-1, table=table)}
        bl = external_scores(ids, blabels, level, noise_label=-1, table=table)
        if bl["n_scored"] != row["n_scored"]:
            raise ValueError(f"{base['partition']}: baseline scored on other proteins")
        row.update({f"length_baseline_{m}": bl[m] for m in MEASURES})
        row.update({f"length_baseline_{m}_singletons": bl[f"{m}_singletons"] for m in MEASURES})
        labelled = table.loc[ids, f"ec{level}_n"].to_numpy() == 1
        classes = table.loc[ids[labelled], f"ec{level}"].to_numpy(dtype=str)
        pred = labels[labelled]
        clustered = pred != -1
        row.update({f"{m}_shuffled_mean": v for m, v in shuffled_means(classes[clustered], pred[clustered]).items()})
        single = pred.astype(object).copy()
        single[~clustered] = [f"noise_{i}" for i in range(int((~clustered).sum()))]
        row.update({f"{m}_singletons_shuffled_mean": v for m, v in shuffled_means(classes, single.astype(str)).items()})
        shares = pd.Series(classes[clustered]).value_counts(normalize=True)
        row["reference_largest_class"] = str(shares.index[0]); row["reference_largest_share"] = float(shares.iloc[0])
        row["k"] = int(labels.max()) + 1; row["n_noise"] = int((labels == -1).sum())
        row["partition_sha256"] = partition_hash(ids, labels)
        rows.append(row)
    return rows


def noise_classes(ids: np.ndarray, labels: np.ndarray, base: dict, table: pd.DataFrame) -> list[dict]:
    rows = []
    noise = labels == -1
    for level in EC_LEVELS:
        n_col = table.loc[ids, f"ec{level}_n"].to_numpy(); cls = table.loc[ids, f"ec{level}"].to_numpy(dtype=str)
        key = np.where(n_col == 1, cls, np.where(n_col == 0, "no class", "several classes"))
        for value in sorted(set(key.tolist())):
            members = key == value
            rows.append({**base, "level": f"EC{level}", "class": value, "n_all": int(members.sum()),
                         "n_noise": int((members & noise).sum()),
                         "noise_share_of_class": float((members & noise).sum() / members.sum())})
    return rows


def check_scores(scores: pd.DataFrame) -> None:
    if len(scores) != len(EC_LEVELS) * EXPECTED_PARTITIONS:
        raise ValueError(f"{len(scores)} score rows")
    if (scores["score_status"] != "scored").any():
        raise ValueError("a partition was not scorable")
    total = scores["n_labelled"] + scores["n_no_class"] + scores["n_several_classes"]
    if not (total == scores["n_records"]).all():
        raise ValueError("labelled and excluded proteins do not add up to n")
    cols = [c for c in scores.columns if any(c.startswith(m) for m in MEASURES) or c.startswith("length_baseline_")]
    if scores[cols].isna().any().any():
        raise ValueError("a score is missing")


def summarise(scores: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["embedding", "stored_row", "role", "records", "level"]
    for key, g in scores.groupby(keys, sort=True):
        stored = g[g.order_seed == -1].iloc[0]
        for policy, suffix in (("primary", ""), ("check, noise as singletons", "_singletons")):
            for m in MEASURES:
                v = g[f"{m}{suffix}"]; b = g[f"length_baseline_{m}{suffix}"]
                row = dict(zip(keys, key), policy=policy, measure=m, n_partitions=len(g),
                           stored=float(stored[f"{m}{suffix}"]), median=float(v.median()), minimum=float(v.min()),
                           maximum=float(v.max()), length_baseline_stored=float(stored[f"length_baseline_{m}{suffix}"]),
                           length_baseline_median=float(b.median()), n_at_or_below_length_baseline=int((v <= b).sum()),
                           clustered_fraction_all_stored=float(stored.clustered_fraction_all),
                           clustered_fraction_all_median=float(g.clustered_fraction_all.median()),
                           clustered_fraction_labelled_stored=float(stored.clustered_fraction_labelled),
                           n_scored_stored=int(stored.n_scored), n_labelled=int(stored.n_labelled),
                           k_stored=int(stored.k), k_values=" ".join(str(k) for k in sorted(g.k)),
                           reference_largest_share_stored=float(stored.reference_largest_share))
                if m in SHUFFLE:
                    sh = g[f"{m}{suffix}_shuffled_mean"]
                    row["shuffled_mean_largest_magnitude"] = float(sh.abs().max())
                    row["n_at_or_below_shuffled"] = int((v <= sh).sum())
                rows.append(row)
    return pd.DataFrame(rows)


def with_provenance(frame: pd.DataFrame, git: str, ec_sha: str) -> pd.DataFrame:
    return frame.assign(dataset=DATASET, ec_table=EC_PROTEINS.relative_to(REPO).as_posix(), ec_table_sha256=ec_sha,
                        shuffle_seeds="0-4", git_sha=git, **LIBRARY_VERSIONS)


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
        if not np.allclose(back[numeric].to_numpy(float), frame[numeric].to_numpy(float), rtol=0, atol=1e-12, equal_nan=True):
            raise ValueError(f"{name}: numbers changed on reopening")
        if not (back[text].fillna("").astype(str).to_numpy() == frame[text].fillna("").astype(str).to_numpy()).all():
            raise ValueError(f"{name}: text changed on reopening")


def build(table: pd.DataFrame, lengths: pd.Series) -> dict[str, pd.DataFrame]:
    scores, noise = [], []
    for base, ids, labels in partitions():
        scores.extend(ec_rows(ids, labels, base, table, lengths))
        if base["order_seed"] == -1:
            noise.extend(noise_classes(ids, labels, base, table))
    scores = pd.DataFrame(scores)
    check_scores(scores)
    return {"hdbscan_ec_scores.csv": scores, "hdbscan_ec_summary.csv": summarise(scores),
            "hdbscan_ec_noise_classes.csv": pd.DataFrame(noise)}


def shown(frames: dict[str, pd.DataFrame]) -> str:
    s = frames["hdbscan_ec_summary.csv"]
    s = s[s.measure.isin(["ami", "ari"])]
    return s[["stored_row", "records", "level", "policy", "measure", "stored", "median", "minimum", "maximum",
              "length_baseline_stored", "n_at_or_below_length_baseline", "shuffled_mean_largest_magnitude",
              "n_at_or_below_shuffled", "clustered_fraction_all_stored", "n_scored_stored"]].round(3).to_string(index=False)


def run_scientific(out: Path = OUT) -> dict[str, int]:
    table, lengths = ec_table(), sequence_lengths()
    ec_sha = hashlib.sha256(EC_PROTEINS.read_bytes()).hexdigest()
    git = git_sha()
    frames = {n: with_provenance(f, git, ec_sha) for n, f in build(table, lengths).items()}
    write(out, frames)
    print(f"git {git}; EC table sha256 {ec_sha[:12]}; rows { {n: len(f) for n, f in frames.items()} }")
    with pd.option_context("display.width", 250):
        print(shown(frames))
    return {n: len(f) for n, f in frames.items()}


def preflight() -> None:
    table, lengths = ec_table(), sequence_lengths()
    parts = partitions()
    base, ids, labels = parts[0]
    rows = ec_rows(ids, labels, base, table, lengths)
    print(f"{len(parts)} partitions checked; {base['partition']} scored in memory ({len(rows)} rows). Nothing written.")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").partition("\n")[0])
    parser.add_argument("--run-scientific", action="store_true", help="score the 86 partitions and write the EC tables")
    if parser.parse_args(argv).run_scientific:
        run_scientific()
    else:
        preflight()


if __name__ == "__main__":
    main()
