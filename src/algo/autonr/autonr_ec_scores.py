from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import pandas as pd

from src.algo.hdbscan.hdbscan_ec_scores import (
    LIBRARY_VERSIONS, ec_rows, noise_classes, write,
)
from src.data import EC_LEVELS, EC_PROTEINS, REPO, ec_table
from src.length_baseline import sequence_lengths

DATASET = "phosphatase"
AN = REPO / "results/autonr"
LOCAL = AN / "assignments/local_pca_seeds"
ESM = AN / "esm2_t36_3B_UR50D/attempts"
RAW = AN / "attempts/raw_full_seed0_20260925"
OUT = AN / "ec_scores"
N_FULL, N_DEDUP = 1587, 1480
EXPECTED_FITS = 120 + 10 + 1
LEVEL3 = "EC3"
BEST_CHOSEN_BY = "highest EC3 AMI (label-informed)"


def git_sha() -> str:
    try:
        sha = subprocess.run(["git", "describe", "--always"], cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()
        status = subprocess.run(["git", "status", "--porcelain", "--", ".", ":(exclude)results/autonr/ec_scores/**"],
                                cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return f"{sha}-dirty" if status else sha


def read_fit(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        return z["protein_id"].astype(str), z["label"].astype(np.int64), z["m"].astype(np.int64)


def check_fit(ids: np.ndarray, labels: np.ndarray, dims: np.ndarray, run: pd.Series, subspaces: pd.DataFrame,
              name: str) -> None:
    n = int(run.n)
    if ids.ndim != 1 or len(ids) != n or len(set(ids.tolist())) != n or list(ids) != sorted(ids):
        raise ValueError(f"{name}: IDs are not {n} unique sorted proteins")
    if hashlib.sha256("\n".join(ids).encode()).hexdigest() != str(run.subset_hash):
        raise ValueError(f"{name}: IDs are not the recorded record set")
    if labels.shape != (n, len(subspaces)) or len(dims) != len(subspaces):
        raise ValueError(f"{name}: {labels.shape[1]} label columns, {len(subspaces)} subspaces recorded")
    for j, s in enumerate(subspaces.sort_values("subspace").itertuples()):
        col = labels[:, j]
        k = int(col.max()) + 1
        if set(np.unique(col).tolist()) - set(range(-1, k)) or len(np.unique(col[col >= 0])) != k:
            raise ValueError(f"{name} subspace {j}: labels are not -1 and 0 to K-1")
        if (k, int((col == -1).sum()), int(dims[j])) != (int(s.n_clusters), int(s.n_outliers), int(s.dimensions)):
            raise ValueError(f"{name} subspace {j}: archive does not match its subspace row")
        if bool(s.noise_space) != (k == 1):
            raise ValueError(f"{name} subspace {j}: noise flag disagrees with one cluster")


def fits() -> list[tuple[dict, np.ndarray, np.ndarray, np.ndarray]]:
    out = []
    runs = pd.read_csv(AN / "metrics/local_pca_runs.csv").set_index("run_id")
    subs = pd.read_csv(AN / "metrics/local_pca_subspaces.csv")
    for path in sorted(LOCAL.glob("*.npz")):
        run_id = path.stem
        run = runs.loc[run_id]
        ids, labels, dims = read_fit(path)
        check_fit(ids, labels, dims, run, subs[subs.run_id == run_id], run_id)
        out.append(({"embedding": "prott5", "arm": f"{run.geometry}_{run.records}", "geometry": str(run.geometry),
                     "records": str(run.records), "seed": int(run.seed), "run_id": run_id,
                     "git_sha_partition": str(run.git_sha)}, ids, labels, dims))
    for folder in sorted(ESM.glob("esm2_3b_pca90_full_seed*")):
        run = pd.read_csv(folder / "run.csv").iloc[0]
        ids, labels, dims = read_fit(folder / "model.npz")
        check_fit(ids, labels, dims, run, pd.read_csv(folder / "subspaces.csv"), folder.name)
        out.append(({"embedding": "esm2_t36_3B_UR50D", "arm": "pca45_full", "geometry": "pca90", "records": "full",
                     "seed": int(run.seed), "run_id": folder.name, "git_sha_partition": str(run.git_sha)},
                    ids, labels, dims))
    run = pd.read_csv(RAW / "run.csv").iloc[0]
    ids, labels, dims = read_fit(RAW / "model.npz")
    check_fit(ids, labels, dims, run, pd.read_csv(RAW / "subspaces.csv"), RAW.name)
    out.append(({"embedding": "prott5", "arm": "raw_full_single_fit", "geometry": "raw", "records": "full",
                 "seed": int(run.seed), "run_id": RAW.name, "git_sha_partition": str(run.git_sha)}, ids, labels, dims))
    if len(out) != EXPECTED_FITS:
        raise ValueError(f"{len(out)} fits, expected {EXPECTED_FITS}")
    for base, ids, _, _ in out:
        if len(ids) != (N_FULL if base["records"] == "full" else N_DEDUP):
            raise ValueError(f"{base['run_id']}: {len(ids)} records")
    return out


def subspace_parts(base: dict, labels: np.ndarray, dims: np.ndarray):
    n_clustered = sum(int(labels[:, j].max()) + 1 > 1 for j in range(labels.shape[1]))
    for j in range(labels.shape[1]):
        col = labels[:, j]
        if int(col.max()) + 1 == 1:
            continue
        yield {**base, "subspace": j, "dimensions": int(dims[j]), "n_subspaces": int(labels.shape[1]),
               "n_clustered_subspaces": n_clustered, "partition": f"{base['run_id']}_s{j}"}, col


def best_match(scores: pd.DataFrame) -> pd.DataFrame:
    keys = ["embedding", "arm", "seed", "run_id"]
    ec3 = scores[(scores.level == LEVEL3) & (scores.score_status == "scored")]
    rows = []
    for key, g in ec3.groupby(keys, sort=True):
        top = g.sort_values(["ami", "subspace"], ascending=[False, True]).iloc[0]
        for level in [f"EC{x}" for x in EC_LEVELS]:
            r = scores[(scores.run_id == top.run_id) & (scores.subspace == top.subspace) & (scores.level == level)].iloc[0]
            rows.append({**dict(zip(keys, key)), "level": level, "chosen_by": BEST_CHOSEN_BY,
                         "n_candidates": int(len(g)), "subspace": int(top.subspace), "dimensions": int(r.dimensions),
                         "k": int(r.k), "n_outliers": int(r.n_noise), "n_scored": int(r.n_scored),
                         "clustered_fraction_all": float(r.clustered_fraction_all),
                         "ami": float(r.ami), "ari": float(r.ari), "nmi": float(r.nmi),
                         "ami_singletons": float(r.ami_singletons), "length_baseline_ami": float(r.length_baseline_ami),
                         "ami_shuffled_mean": float(r.ami_shuffled_mean)})
    return pd.DataFrame(rows)


def summarise(scores: pd.DataFrame, best: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (emb, arm, level), g in scores.groupby(["embedding", "arm", "level"], sort=True):
        s = g[g.score_status == "scored"]
        b = best[(best.embedding == emb) & (best.arm == arm) & (best.level == level)]
        rows.append({"embedding": emb, "arm": arm, "level": level, "n_fits": int(g.run_id.nunique()),
                     "n_subspaces": int(len(g)), "n_scored": int(len(s)),
                     "clustered_subspaces_per_fit_min": int(g.groupby("run_id").subspace.nunique().min()),
                     "clustered_subspaces_per_fit_max": int(g.groupby("run_id").subspace.nunique().max()),
                     "k_min": int(s.k.min()), "k_max": int(s.k.max()),
                     "ami_median": float(s.ami.median()), "ami_min": float(s.ami.min()), "ami_max": float(s.ami.max()),
                     "length_baseline_ami_median": float(s.length_baseline_ami.median()),
                     "n_at_or_below_length_baseline": int((s.ami <= s.length_baseline_ami).sum()),
                     "n_at_or_below_shuffled": int((s.ami <= s.ami_shuffled_mean).sum()),
                     "shuffled_ami_largest_magnitude": float(s.ami_shuffled_mean.abs().max()),
                     "clustered_fraction_all_median": float(s.clustered_fraction_all.median()),
                     "clustered_fraction_all_min": float(s.clustered_fraction_all.min()),
                     "best_chosen_by": "; ".join(sorted(set(b.chosen_by))),
                     "best_ami_median": float(b.ami.median()), "best_ami_min": float(b.ami.min()),
                     "best_ami_max": float(b.ami.max()),
                     "best_length_baseline_ami_median": float(b.length_baseline_ami.median()),
                     "best_n_at_or_below_length_baseline": int((b.ami <= b.length_baseline_ami).sum()),
                     "best_n_candidates_min": int(b.n_candidates.min()), "best_n_candidates_max": int(b.n_candidates.max()),
                     "best_k_min": int(b.k.min()), "best_k_max": int(b.k.max())})
    return pd.DataFrame(rows)


def check_scores(scores: pd.DataFrame, n_expected: int) -> None:
    if len(scores) != len(EC_LEVELS) * n_expected:
        raise ValueError(f"{len(scores)} score rows, expected {len(EC_LEVELS) * n_expected}")
    total = scores["n_labelled"] + scores["n_no_class"] + scores["n_several_classes"]
    if not (total == scores["n_records"]).all():
        raise ValueError("labelled and excluded proteins do not add up to n")
    bad = scores[scores.score_status != "scored"]
    if len(bad):
        raise ValueError(f"{len(bad)} rows not scorable: {bad.score_status.value_counts().to_dict()}")
    cols = [c for c in scores.columns if c.startswith(("ari", "nmi", "ami", "homogeneity", "completeness", "length_baseline_"))]
    if scores[cols].isna().any().any():
        raise ValueError("a score is missing")


def build(table: pd.DataFrame, lengths: pd.Series) -> dict[str, pd.DataFrame]:
    scores, outliers, n = [], [], 0
    for base, ids, labels, dims in fits():
        for sbase, col in subspace_parts(base, labels, dims):
            n += 1
            scores.extend(ec_rows(ids, col, sbase, table, lengths))
            if (col == -1).any():
                outliers.extend(noise_classes(ids, col, sbase, table))
    scores = pd.DataFrame(scores).rename(columns={"n_noise": "n_outliers"}).assign(n_noise=lambda d: d.n_outliers)
    check_scores(scores, n)
    best = best_match(scores)
    return {"autonr_ec_scores.csv": scores.drop(columns="n_noise"), "autonr_ec_best_match.csv": best,
            "autonr_ec_summary.csv": summarise(scores, best),
            "autonr_ec_outlier_classes.csv": pd.DataFrame(outliers).rename(columns={"n_noise": "n_outliers",
                                                                                    "noise_share_of_class": "outlier_share_of_class"})}


def with_provenance(frame: pd.DataFrame, git: str, ec_sha: str) -> pd.DataFrame:
    return frame.assign(dataset=DATASET, ec_table=EC_PROTEINS.relative_to(REPO).as_posix(), ec_table_sha256=ec_sha,
                        shuffle_seeds="0-4", git_sha=git, **LIBRARY_VERSIONS)


def shown(frames: dict[str, pd.DataFrame]) -> str:
    s = frames["autonr_ec_summary.csv"]
    return s[["embedding", "arm", "level", "n_fits", "n_subspaces", "k_min", "k_max", "ami_median", "ami_max",
              "length_baseline_ami_median", "n_at_or_below_length_baseline", "n_at_or_below_shuffled",
              "best_ami_median", "best_ami_min", "best_ami_max", "best_length_baseline_ami_median",
              "best_n_at_or_below_length_baseline"]].round(3).to_string(index=False)


def run_scientific(out: Path = OUT) -> dict[str, int]:
    table, lengths = ec_table(), sequence_lengths()
    ec_sha = hashlib.sha256(EC_PROTEINS.read_bytes()).hexdigest()
    git = git_sha()
    frames = {n: with_provenance(f, git, ec_sha) for n, f in build(table, lengths).items()}
    write(out, frames)
    print(f"git {git}; EC table sha256 {ec_sha[:12]}; rows { {n: len(f) for n, f in frames.items()} }")
    with pd.option_context("display.width", 250, "display.max_columns", 40):
        print(shown(frames))
    return {n: len(f) for n, f in frames.items()}


def preflight() -> None:
    table, lengths = ec_table(), sequence_lengths()
    all_fits = fits()
    n = sum(sum(1 for _ in subspace_parts(b, lab, d)) for b, _, lab, d in all_fits)
    base, ids, labels, dims = all_fits[0]
    sbase, col = next(subspace_parts(base, labels, dims))
    rows = ec_rows(ids, col, sbase, table, lengths)
    print(f"{len(all_fits)} fits checked, {n} clustered subspaces; {sbase['partition']} scored in memory "
          f"({len(rows)} rows). Nothing written.")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").partition("\n")[0])
    parser.add_argument("--run-scientific", action="store_true", help="score every clustered subspace and write the EC tables")
    if parser.parse_args(argv).run_scientific:
        run_scientific()
    else:
        preflight()


if __name__ == "__main__":
    main()
