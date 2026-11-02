from __future__ import annotations

import hashlib
import inspect
import io
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors

PARQUET_DELIMITER = b"---PARQUET_DELIMITER---"

REPO = Path(__file__).resolve().parents[1]

SEED = 42

DATASETS = {
    "phosphatase": (
        REPO / "data/phosphatase/phosphatase/tmp/prot_t5.h5",
        REPO / "data/phosphatase/phosphatase/data_standard.parquetbundle",
        REPO / "data/phosphatase/phosphatase/tmp/sequences.fasta",
    ),
}

DEFAULT_DATASET = "phosphatase"


DEDUP_SUFFIX = "_dedup"


def out_paths(dedup: bool = False) -> tuple[Path, Path, Path]:
    s = DEDUP_SUFFIX if dedup else ""
    return (REPO / f"results/metrics/embedding_profile{s}.json",
            REPO / f"results/embedding_profile{s}.md",
            REPO / f"results/metrics/duplicate_groups{s}.csv")


@dataclass
class Dataset:
    name: str
    ids: list[str]
    X: np.ndarray
    ann: pd.DataFrame

    def __len__(self) -> int:
        return len(self.ids)

    @property
    def subset_hash(self) -> str:
        return hashlib.sha1("\n".join(sorted(self.ids)).encode()).hexdigest()[:12]


def load_embeddings(path: Path) -> tuple[list[str], np.ndarray]:
    with h5py.File(path, "r") as f:
        ids = list(f.keys())
        X = np.stack([f[k][:] for k in ids])
    return ids, X


def read_fasta(path: Path) -> dict[str, str]:
    seqs: dict[str, str] = {}
    key, buf = None, []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line.startswith(">"):
            if key:
                seqs[key] = "".join(buf)
            head = line[1:].split()[0]
            key, buf = (head.split("|")[1] if "|" in head else head), []
        elif line:
            buf.append(line)
    if key:
        seqs[key] = "".join(buf)
    return seqs


def read_bundle_part(path: Path, index: int = 0) -> pd.DataFrame:
    parts = path.read_bytes().split(PARQUET_DELIMITER)
    return pd.read_parquet(io.BytesIO(parts[index]))


def load_dataset(name: str = DEFAULT_DATASET) -> Dataset:
    if name not in DATASETS:
        raise KeyError(f"unknown dataset '{name}'. Known: {', '.join(DATASETS)}")
    h5_path, bundle_path, _ = DATASETS[name]

    ids, X = load_embeddings(h5_path)
    ann = read_bundle_part(bundle_path, 0)

    missing = set(ids) - set(ann["protein_id"])
    extra = set(ann["protein_id"]) - set(ids)
    if missing or extra:
        raise ValueError(
            f"ID mismatch between embeddings and annotations: "
            f"{len(missing)} only in .h5, {len(extra)} only in the bundle"
        )

    ann = ann.set_index("protein_id").loc[ids].reset_index()
    return Dataset(name=name, ids=ids, X=X, ann=ann)


def lengths(ds: Dataset) -> np.ndarray:
    return pd.to_numeric(ds.ann["length"], errors="coerce").to_numpy(dtype=float)


def embedding_qc(ds: Dataset) -> dict:
    norms = np.linalg.norm(ds.X, axis=1)
    lens = lengths(ds)
    ok = np.isfinite(lens)
    return {
        "n_proteins": len(ds),
        "n_dims": int(ds.X.shape[1]),
        "dtype": str(ds.X.dtype),
        "total_residues": int(np.nansum(lens)),
        "norm_min": round(float(norms.min()), 4),
        "norm_median": round(float(np.median(norms)), 4),
        "norm_max": round(float(norms.max()), 4),
        "norm_sd": round(float(norms.std(ddof=1)), 4),
        "all_unit_norm": bool(np.allclose(norms, 1.0)),
        "corr_norm_length": round(float(np.corrcoef(norms[ok], lens[ok])[0, 1]), 4),
        "corr_norm_log_length": round(float(np.corrcoef(norms[ok], np.log(lens[ok]))[0, 1]), 4),
    }


VAR_THRESHOLDS = (0.5, 0.8, 0.9, 0.95, 0.99)


def effective_dim(X: np.ndarray) -> dict:
    p = PCA(random_state=SEED).fit(X)
    cum = np.cumsum(p.explained_variance_ratio_)
    ev = p.explained_variance_
    out = {f"pc_for_{int(t * 100)}pct": int(np.searchsorted(cum, t)) + 1
           for t in VAR_THRESHOLDS}
    out["pc1_variance_pct"] = round(float(p.explained_variance_ratio_[0] * 100), 2)
    out["participation_ratio"] = round(float(ev.sum() ** 2 / (ev**2).sum()), 2)
    return out


def distance_concentration(flat: np.ndarray) -> dict:
    return {
        "dist_min": round(float(flat.min()), 4),
        "dist_mean": round(float(flat.mean()), 4),
        "dist_max": round(float(flat.max()), 4),
        "relative_contrast": round(float((flat.max() - flat.min()) / flat.mean()), 4),
        "coef_variation": round(float(flat.std() / flat.mean()), 4),
    }


def metric_agreement(X: np.ndarray, flat: np.ndarray, iu) -> dict:
    C = pairwise_distances(X, metric="cosine")[iu]
    return {"spearman_euclid_cosine": round(float(spearmanr(flat, C).statistic), 4)}


def anisotropy(X: np.ndarray) -> dict:
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2 or X.shape[0] < 2 or X.shape[1] < 1:
        raise ValueError("anisotropy requires a 2-D matrix with at least two rows and one column")
    if not np.isfinite(X).all():
        raise ValueError("anisotropy requires finite embeddings")

    iu = np.triu_indices(len(X), 1)
    expected_random = 1.0 / np.sqrt(X.shape[1])

    avg = float((1.0 - pairwise_distances(X, metric="cosine"))[iu].mean())
    X_centred = X - X.mean(axis=0)
    avg_centred = float(
        (1.0 - pairwise_distances(X_centred, metric="cosine"))[iu].mean()
    )

    return {
        "avg_pairwise_cosine_sim": round(avg, 4),
        "expected_random": round(float(expected_random), 4),
        "anisotropy_score": round(float(avg / expected_random), 4),
        "avg_pairwise_cosine_sim_centred": round(avg_centred, 4),
        "anisotropy_score_centred": round(float(avg_centred / expected_random), 4),
    }


KNN_KS = (4, 10, 20)


def density_spread(X: np.ndarray) -> dict:
    out = {}
    for k in KNN_KS:
        dk = NearestNeighbors(n_neighbors=k + 1).fit(X).kneighbors(X)[0][:, k]
        q25, q75 = np.percentile(dk, [25, 75])
        out[f"knn{k}_min"] = round(float(dk.min()), 4)
        out[f"knn{k}_p25"] = round(float(q25), 4)
        out[f"knn{k}_median"] = round(float(np.median(dk)), 4)
        out[f"knn{k}_p75"] = round(float(q75), 4)
        out[f"knn{k}_max"] = round(float(dk.max()), 4)
        out[f"knn{k}_p75_over_p25"] = round(float(q75 / max(q25, 1e-9)), 2)
    return out


HUB_KS = (5, 10, 20)


def _knn_without_self(X: np.ndarray, k: int) -> np.ndarray:
    distances = pairwise_distances(X, metric="euclidean")
    np.fill_diagonal(distances, np.inf)
    neighbours = np.argsort(distances, axis=1, kind="stable")[:, :k]

    assert neighbours.shape == (len(X), k)
    assert not np.any(neighbours == np.arange(len(X))[:, None])
    return neighbours


def k_occurrence(neighbours: np.ndarray, k: int) -> np.ndarray:
    return np.bincount(neighbours[:, :k].ravel(), minlength=len(neighbours))


def hubness(X: np.ndarray) -> dict:
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2 or X.shape[1] < 1:
        raise ValueError("hubness requires a 2-D matrix with at least one column")
    if len(X) <= max(HUB_KS):
        raise ValueError(f"hubness requires more than {max(HUB_KS)} rows")
    if not np.isfinite(X).all():
        raise ValueError("hubness requires finite embeddings")

    out = {}
    neighbours = _knn_without_self(X, max(HUB_KS))
    for k in HUB_KS:
        occurrence = k_occurrence(neighbours, k)
        assert int(occurrence.sum()) == len(X) * k
        assert float(occurrence.mean()) == float(k)

        excess = (occurrence[occurrence > k] - k).sum()
        rhi = float(excess / occurrence.sum())
        out[f"hub{k}_rhi"] = round(rhi, 4)
        out[f"hub{k}_kocc_max"] = int(occurrence.max())
        out[f"hub{k}_kocc_zero"] = int(np.count_nonzero(occurrence == 0))
    return out


DUP_DECIMALS = 6

DUP_CONTEXT = ("uniprot_kb_id", "gene_name", "protein_name", "species", "length", "reviewed",
               "fragment")


def _duplicate_labels(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    _, inverse, counts = np.unique(
        X.round(DUP_DECIMALS), axis=0, return_inverse=True, return_counts=True
    )
    return np.asarray(inverse).ravel(), counts


def duplicates(X: np.ndarray) -> dict:
    _, counts = _duplicate_labels(X)
    dup = counts[counts > 1]
    return {
        "n_unique_vectors": int(len(counts)),
        "n_duplicate_groups": int(len(dup)),
        "n_proteins_in_groups": int(dup.sum()),
        "pct_proteins_in_groups": round(float(dup.sum() / len(X) * 100), 2),
        "largest_group": int(dup.max()) if len(dup) else 0,
    }


def duplicate_groups(ds: Dataset) -> pd.DataFrame:
    Xf = np.asarray(ds.X)
    labels, counts = _duplicate_labels(np.asarray(Xf, dtype=np.float64))
    keep = counts[labels] > 1
    kept = np.flatnonzero(keep)

    cols = [c for c in DUP_CONTEXT if c in ds.ann.columns]
    out = ds.ann.loc[keep, ["protein_id", *cols]].copy()
    out.insert(0, "_label", labels[keep])
    out.insert(0, "group_size", counts[labels][keep])
    out["vector_sha1"] = [hashlib.sha1(Xf[i].tobytes()).hexdigest()[:12] for i in kept]

    agree = {}
    for lab in np.unique(labels[keep]):
        rows = Xf[np.flatnonzero(labels == lab)]
        agree[lab] = int((rows == rows[0]).sum(axis=1).min())
    out["dims_equal_in_group"] = out["_label"].map(agree)

    order = (out.groupby("_label")["group_size"].first()
                .sort_values(ascending=False, kind="stable").index)
    out["group_id"] = out["_label"].map({lab: i for i, lab in enumerate(order, start=1)})

    for col, name in (("species", "n_species"), ("length", "n_lengths")):
        if col in out.columns:
            out[name] = out.groupby("_label")[col].transform("nunique")

    out = out.drop(columns="_label").sort_values(["group_id", "protein_id"], kind="stable")
    front = ["group_id", "group_size", "dims_equal_in_group", "vector_sha1",
             *[c for c in ("n_species", "n_lengths") if c in out.columns]]
    return out[[*front, *[c for c in out.columns if c not in front]]].reset_index(drop=True)


def deduplicate(ds: Dataset) -> Dataset:
    labels, _ = _duplicate_labels(np.asarray(ds.X, dtype=np.float64))
    keep = np.zeros(len(ds), dtype=bool)
    seen: set[int] = set()
    for i in np.argsort(np.asarray(ds.ids), kind="stable"):
        lab = int(labels[i])
        if lab not in seen:
            seen.add(lab)
            keep[i] = True
    idx = np.flatnonzero(keep)
    return Dataset(name=ds.name, ids=[ds.ids[i] for i in idx], X=ds.X[idx],
                   ann=ds.ann.iloc[idx].reset_index(drop=True))


_SEQ_KEYS = ("n_sequences_in_fasta", "ids_missing_from_fasta", "duplicate_groups_checked",
             "groups_with_one_sequence", "groups_with_more_than_one_sequence",
             "distinct_sequences_across_groups", "fasta_length_matches_annotation")


def sequence_check(ds: Dataset) -> dict:
    fasta = DATASETS.get(ds.name, (None, None, None))[2]
    if fasta is None or not Path(fasta).exists():
        return {"fasta_registered": False, **{k: None for k in _SEQ_KEYS}}

    seqs = read_fasta(Path(fasta))
    labels, counts = _duplicate_labels(np.asarray(ds.X, dtype=np.float64))
    ann_len = pd.to_numeric(ds.ann["length"], errors="coerce").to_numpy(dtype=float)

    one, many, firsts, missing, len_ok = 0, 0, [], 0, True
    for lab in np.unique(labels[counts[labels] > 1]):
        members = np.flatnonzero(labels == lab)
        found = [ds.ids[i] for i in members if ds.ids[i] in seqs]
        missing += len(members) - len(found)
        if not found:
            continue
        distinct = {seqs[p] for p in found}
        one, many = one + (len(distinct) == 1), many + (len(distinct) > 1)
        firsts.append(seqs[found[0]])
        for i in members:
            if ds.ids[i] in seqs and np.isfinite(ann_len[i]):
                len_ok &= len(seqs[ds.ids[i]]) == int(ann_len[i])

    return {
        "fasta_registered": True,
        "n_sequences_in_fasta": len(seqs),
        "ids_missing_from_fasta": int(missing),
        "duplicate_groups_checked": one + many,
        "groups_with_one_sequence": int(one),
        "groups_with_more_than_one_sequence": int(many),
        "distinct_sequences_across_groups": len(set(firsts)),
        "fasta_length_matches_annotation": bool(len_ok),
    }


def embedding_profile(ds: Dataset) -> dict[str, dict]:
    X = np.asarray(ds.X, dtype=np.float64)
    D = pairwise_distances(X)
    iu = np.triu_indices(len(X), 1)
    flat = D[iu]
    return {
        "embedding_qc": embedding_qc(ds),
        "duplicates": duplicates(X),
        "sequence_check": sequence_check(ds),
        "effective_dim": effective_dim(X),
        "distance_concentration": distance_concentration(flat),
        "metric_agreement": metric_agreement(X, flat, iu),
        "anisotropy": anisotropy(X),
        "density_spread": density_spread(X),
        "hubness": hubness(X),
    }


_FUNCS = {"embedding_qc": embedding_qc, "duplicates": duplicates,
          "sequence_check": sequence_check, "effective_dim": effective_dim,
          "distance_concentration": distance_concentration, "metric_agreement": metric_agreement,
          "anisotropy": anisotropy, "density_spread": density_spread, "hubness": hubness}


def write_profile_md(name: str, groups: dict[str, dict], path: Path,
                     *, dedup: bool = False) -> None:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    which = " — deduplicated, one record per duplicate group" if dedup else ""
    cmd = f"src/data.py{' --dedup' if dedup else ''}"
    out = [f"# Embedding profile — {name}{which}", "",
           f"Generated by `{cmd}` on {stamp}. Seed {SEED}. Do not edit by hand.",
           ""]
    for group, values in groups.items():
        out += [f"## {group}", "", inspect.cleandoc(_FUNCS[group].__doc__ or ""), "",
                "| metric | value |", "|---|---|"]
        out += [f"| `{k}` | {v} |" for k, v in values.items()]
        out.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(out), encoding="utf-8")


def main() -> None:
    argv = sys.argv[1:]
    dedup = "--dedup" in argv
    unknown = [a for a in argv if a.startswith("-") and a != "--dedup"]
    if unknown:
        raise SystemExit(f"unknown flag '{unknown[0]}'. Usage: python src/data.py [dataset] [--dedup]")
    names = [a for a in argv if not a.startswith("-")]
    name = names[0] if names else DEFAULT_DATASET

    ds = load_dataset(name)
    n_loaded = len(ds)
    if dedup:
        ds = deduplicate(ds)
    groups = embedding_profile(ds)
    json_path, md_path, csv_path = out_paths(dedup)

    flat = {f"{g}.{k}": v for g, vals in groups.items() for k, v in vals.items()}
    header = {"dataset": name, "seed": SEED, "dedup": dedup, "n": len(ds),
              "subset_hash": ds.subset_hash}
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps({**header, **flat}, indent=2) + "\n")
    write_profile_md(name, groups, md_path, dedup=dedup)

    dups = duplicate_groups(ds)
    dups.to_csv(csv_path, index=False)

    for group, values in groups.items():
        print(f"\n### {group}")
        for k, v in values.items():
            print(f"  {k:<26} {v}")
    if dedup:
        print(f"\ndedup: {n_loaded} records in, {len(ds)} out, {n_loaded - len(ds)} dropped")
    print(f"{len(dups)} proteins in {dups['group_id'].nunique() if len(dups) else 0} duplicate "
          f"groups -> {csv_path.relative_to(REPO)}")
    print(f"written to {json_path.relative_to(REPO)} and {md_path.relative_to(REPO)}")


EC_PROTEINS = REPO / "results/ec_audit/proteins.csv"
EC_LEVELS = (3, 4)


def ec_table(path: Path = EC_PROTEINS) -> pd.DataFrame:
    table = pd.read_csv(path, dtype=str, keep_default_na=False).set_index("protein_id")
    for level in (1, 2, 3, 4):
        table[f"ec{level}_n"] = table[f"ec{level}_n"].astype(int)
    if table.index.duplicated().any():
        raise ValueError(f"{path} lists a protein twice")
    return table


def ec_reference(level: int, table: pd.DataFrame | None = None) -> pd.Series:
    if level not in (1, 2, 3, 4):
        raise ValueError("EC level must be 1 to 4")
    table = ec_table() if table is None else table
    return table.loc[table[f"ec{level}_n"] == 1, f"ec{level}"].rename(f"ec{level}")


def external_scores(protein_ids, labels, level: int, *, noise_label: int | None = None,
                    table: pd.DataFrame | None = None) -> dict:
    from sklearn.metrics import (adjusted_mutual_info_score, adjusted_rand_score,
                                 completeness_score, homogeneity_score,
                                 normalized_mutual_info_score)

    ids = np.asarray(protein_ids, dtype=str)
    predicted = np.asarray(labels)
    if ids.ndim != 1 or ids.shape != predicted.shape or len(set(ids.tolist())) != len(ids):
        raise ValueError("need one label per unique protein ID")
    table = ec_table() if table is None else table
    unknown = sorted(set(ids.tolist()) - set(table.index))
    if unknown:
        raise ValueError(f"{len(unknown)} protein IDs are not in the EC table, e.g. {unknown[:3]}")
    counts = table.loc[ids, f"ec{level}_n"].to_numpy()
    keep = counts == 1
    classes = table.loc[ids[keep], f"ec{level}"].to_numpy(dtype=str)
    all_predicted = predicted
    predicted = predicted[keep]
    names = ("ari", "nmi", "ami", "homogeneity", "completeness")

    def largest_share(pred):
        if len(pred) == 0:
            return float("nan")
        return float(pd.Series(pred).value_counts(normalize=True).iloc[0])

    def scores(true, pred, suffix=""):
        if len(set(true.tolist())) < 2:
            return {f"{name}{suffix}": float("nan") for name in names}
        return {
            f"ari{suffix}": float(adjusted_rand_score(true, pred)),
            f"nmi{suffix}": float(normalized_mutual_info_score(true, pred)),
            f"ami{suffix}": float(adjusted_mutual_info_score(true, pred)),
            f"homogeneity{suffix}": float(homogeneity_score(true, pred)),
            f"completeness{suffix}": float(completeness_score(true, pred)),
        }

    row = {"level": f"EC{level}", "n_records": int(len(ids)), "n_labelled": int(keep.sum()),
           "n_no_class": int((counts == 0).sum()), "n_several_classes": int((counts > 1).sum()),
           "n_reference_classes": int(len(set(classes.tolist())))}
    if noise_label is None:
        if not keep.any():
            status = "no_labelled_proteins"
        elif len(set(classes.tolist())) < 2:
            status = "one_reference_class"
        else:
            status = "scored"
        row.update(score_status=status, n_scored=int(keep.sum()),
                   n_clusters=int(len(set(predicted.tolist()))),
                   largest_cluster_share=largest_share(predicted),
                   clustered_fraction_all=1.0, clustered_fraction_labelled=1.0)
        row.update(scores(classes, predicted))
        return row
    clustered = predicted != noise_label
    if not keep.any():
        status = "no_labelled_proteins"
    elif not clustered.any():
        status = "all_labelled_proteins_are_noise"
    elif len(set(classes[clustered].tolist())) < 2:
        status = "one_reference_class"
    else:
        status = "scored"
    row.update(score_status=status, n_scored=int(clustered.sum()),
               n_clusters=int(len(set(predicted[clustered].tolist()))),
               largest_cluster_share=largest_share(predicted[clustered]),
               clustered_fraction_all=float((all_predicted != noise_label).mean())
               if len(all_predicted) else float("nan"),
               clustered_fraction_labelled=float(clustered.mean()) if keep.any() else float("nan"))
    row.update(scores(classes[clustered], predicted[clustered]))
    singletons = predicted.astype(str).astype(object)
    singletons[~clustered] = [f"noise_{i}" for i in range(int((~clustered).sum()))]
    row.update(scores(classes, singletons.astype(str), "_singletons"))
    return row

if __name__ == "__main__":
    main()
