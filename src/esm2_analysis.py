from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors

from src import data as profile_metrics
from src import esm2_embeddings


REPO = Path(__file__).resolve().parents[1]
PROFILE_SCHEMA = "esm2-embedding-profile-v1"
REQUIRED_NEIGHBOUR_KS = (20, 50)
COMPARISON_FULL_COUNT = 1587
COMPARISON_MAXIMUM_LENGTH = 1022
COMPARISON_LENGTH_COUNT = 1443
DistanceMetric = Literal["euclidean", "cosine"]


@dataclass(frozen=True)
class AnalysisDataset:
    name: str
    ids: tuple[str, ...]
    X: np.ndarray
    ann: pd.DataFrame
    sequences: tuple[str, ...]

    def __len__(self) -> int:
        return len(self.ids)

    @property
    def subset_hash(self) -> str:
        return hashlib.sha1("\n".join(self.ids).encode("ascii")).hexdigest()[:12]


@dataclass(frozen=True)
class ModelDataset:
    model_key: str
    checkpoint: str
    revision: str
    stem: str
    width: int
    h5_path: Path
    output_root: Path
    annotation_path: Path
    fasta_path: Path
    reference_h5: Path
    repository_root: Path
    manifest: dict[str, Any]
    data: AnalysisDataset


def _require_same_ids(
    embedding_ids: list[str], annotation_ids: list[str], fasta_ids: list[str]
) -> None:
    sources = {
        "embeddings": set(embedding_ids),
        "annotations": set(annotation_ids),
        "FASTA": set(fasta_ids),
    }
    for name, ids in sources.items():
        source_ids = {
            "embeddings": embedding_ids,
            "annotations": annotation_ids,
            "FASTA": fasta_ids,
        }[name]
        if len(ids) != len(source_ids):
            raise ValueError(f"duplicate protein IDs in {name}")

    expected = sources["embeddings"]
    for name in ("annotations", "FASTA"):
        missing = expected - sources[name]
        extra = sources[name] - expected
        if missing or extra:
            raise ValueError(
                f"ID mismatch between embeddings and {name}: "
                f"{len(missing)} missing, {len(extra)} extra"
            )


def load_aligned_dataset(
    *,
    name: str,
    h5_path: Path,
    annotation_path: Path,
    fasta_path: Path,
) -> AnalysisDataset:
    embedding_ids, matrix = profile_metrics.load_embeddings(Path(h5_path))
    annotations = profile_metrics.read_bundle_part(Path(annotation_path), 0)
    if "protein_id" not in annotations:
        raise ValueError("annotation table has no protein_id column")
    annotation_ids = annotations["protein_id"].astype(str).tolist()
    sequence_map = esm2_embeddings.read_fasta(Path(fasta_path))
    _require_same_ids(embedding_ids, annotation_ids, list(sequence_map))

    embedding_index = {protein_id: index for index, protein_id in enumerate(embedding_ids)}
    canonical_ids = tuple(sorted(embedding_ids))
    indices = [embedding_index[protein_id] for protein_id in canonical_ids]
    aligned_annotations = (
        annotations.assign(protein_id=annotation_ids)
        .set_index("protein_id")
        .loc[list(canonical_ids)]
        .reset_index()
    )
    return AnalysisDataset(
        name=name,
        ids=canonical_ids,
        X=np.asarray(matrix)[indices],
        ann=aligned_annotations,
        sequences=tuple(sequence_map[protein_id] for protein_id in canonical_ids),
    )


def sequence_digest(sequence: str) -> str:
    return hashlib.sha256(sequence.encode("ascii")).hexdigest()


def sequence_groups(dataset: AnalysisDataset) -> dict[str, tuple[int, ...]]:
    grouped: dict[str, list[int]] = {}
    sequences_by_digest: dict[str, str] = {}
    for index, sequence in enumerate(dataset.sequences):
        digest = sequence_digest(sequence)
        previous = sequences_by_digest.setdefault(digest, sequence)
        if previous != sequence:
            raise ValueError("SHA256 collision between distinct sequences")
        grouped.setdefault(digest, []).append(index)
    return {digest: tuple(indices) for digest, indices in grouped.items()}


def deduplicate_by_sequence(dataset: AnalysisDataset) -> AnalysisDataset:
    keep = sorted(min(indices, key=lambda index: dataset.ids[index])
                  for indices in sequence_groups(dataset).values())
    return AnalysisDataset(
        name=f"{dataset.name}_dedup",
        ids=tuple(dataset.ids[index] for index in keep),
        X=dataset.X[keep],
        ann=dataset.ann.iloc[keep].reset_index(drop=True),
        sequences=tuple(dataset.sequences[index] for index in keep),
    )


def _checkpoint_stem(checkpoint: str) -> str:
    parts = checkpoint.split("/")
    if len(parts) != 2 or not all(parts) or parts[0] != "facebook":
        raise ValueError(f"unexpected ESM-2 checkpoint name: {checkpoint!r}")
    return parts[1]


def load_model_dataset(
    *,
    h5_path: Path,
    output_root: Path,
    annotation_path: Path,
    fasta_path: Path,
    reference_h5: Path,
    repository_root: Path = REPO,
) -> ModelDataset:
    root = Path(repository_root).resolve()
    h5_path = Path(h5_path).resolve()
    output_root = Path(output_root).resolve()
    manifest_path = output_root / "embedding_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        model = manifest["model"]
        artifact = manifest["artifact"]
        generation = manifest["generation"]
        model_key = str(model["key"])
        checkpoint = str(model["checkpoint"])
        revision = str(model["revision"])
        width = int(model["width"])
        expected_count = int(artifact["record_count"])
        route = str(generation["route"])
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid ESM-2 embedding manifest: {manifest_path}") from error

    spec = esm2_embeddings.model_spec(model_key)
    stem = _checkpoint_stem(checkpoint)
    expected_root = (root / spec.result_root).resolve()
    expected_h5 = (root / spec.final_h5).resolve()
    if output_root != expected_root:
        raise ValueError(f"output root is not allowlisted for {model_key}")
    if h5_path != expected_h5:
        raise ValueError(f"embedding HDF5 path is not allowlisted for {model_key}")
    if (checkpoint, revision, width) != (spec.checkpoint, spec.revision, spec.width):
        raise ValueError("manifest model fields do not match the registered model")
    if (root / Path(str(artifact.get("path", "")))).resolve() != h5_path:
        raise ValueError("manifest artifact path does not match the explicit HDF5 path")

    checked_manifest = esm2_embeddings.validate_persisted(
        model_key=model_key,
        fasta_path=Path(fasta_path),
        reference_h5=Path(reference_h5),
        final_h5=h5_path,
        output_root=output_root,
        route=route,
        expected_count=expected_count,
        expected_width=width,
        repository_root=root,
    )
    data = load_aligned_dataset(
        name=stem,
        h5_path=h5_path,
        annotation_path=annotation_path,
        fasta_path=fasta_path,
    )
    if len(data) != expected_count or data.X.shape != (expected_count, width):
        raise ValueError("aligned matrix shape does not match the embedding manifest")
    return ModelDataset(
        model_key=model_key,
        checkpoint=checkpoint,
        revision=revision,
        stem=stem,
        width=width,
        h5_path=h5_path,
        output_root=output_root,
        annotation_path=Path(annotation_path).resolve(),
        fasta_path=Path(fasta_path).resolve(),
        reference_h5=Path(reference_h5).resolve(),
        repository_root=root,
        manifest=checked_manifest,
        data=data,
    )


def profile_output_paths(model: ModelDataset, *, deduplicated: bool) -> tuple[Path, Path, Path]:
    spec = esm2_embeddings.model_spec(model.model_key)
    expected_root = (model.repository_root / spec.result_root).resolve()
    expected_stem = _checkpoint_stem(spec.checkpoint)
    if model.output_root.resolve() != expected_root or model.stem != expected_stem:
        raise ValueError(f"profile output root is not allowlisted for {model.model_key}")
    suffix = f"{model.stem}_dedup" if deduplicated else model.stem
    return (
        model.output_root / f"embedding_profile_{suffix}.json",
        model.output_root / f"embedding_profile_{suffix}.md",
        model.output_root / f"duplicate_groups_{suffix}.csv",
    )


def validate_comparison_output_root(
    output_root: Path, *, repository_root: Path = REPO
) -> Path:
    resolved = Path(output_root).resolve()
    expected = (Path(repository_root).resolve() / "results/esm2/comparison").resolve()
    if resolved != expected:
        raise ValueError("comparison output root is not allowlisted")
    return resolved


def _profile_dataset(dataset: AnalysisDataset) -> profile_metrics.Dataset:
    return profile_metrics.Dataset(
        name=dataset.name,
        ids=list(dataset.ids),
        X=dataset.X,
        ann=dataset.ann,
    )


def sequence_deduplication(dataset: AnalysisDataset) -> dict[str, int]:
    groups = sequence_groups(dataset)
    duplicates = [indices for indices in groups.values() if len(indices) > 1]
    return {
        "n_unique_sequences": len(groups),
        "n_duplicate_sequence_groups": len(duplicates),
        "n_records_in_duplicate_sequence_groups": sum(map(len, duplicates)),
        "largest_duplicate_sequence_group": max(map(len, duplicates), default=0),
    }


def vector_determinism(dataset: AnalysisDataset) -> dict[str, int | float]:
    matrix = np.asarray(dataset.X)
    _, counts = np.unique(matrix, axis=0, return_counts=True)
    duplicate_counts = counts[counts > 1]
    maximum_difference = 0.0
    for indices in sequence_groups(dataset).values():
        if len(indices) < 2:
            continue
        rows = np.asarray(matrix[list(indices)], dtype=np.float64)
        maximum_difference = max(maximum_difference, float(np.max(np.ptp(rows, axis=0))))
    return {
        "n_unique_exact_vectors": len(counts),
        "n_exact_vector_duplicate_groups": len(duplicate_counts),
        "n_records_in_exact_vector_duplicate_groups": int(duplicate_counts.sum()),
        "largest_exact_vector_duplicate_group": int(duplicate_counts.max())
        if len(duplicate_counts)
        else 0,
        "max_within_sequence_group_absolute_difference": maximum_difference,
    }


def duplicate_groups_table(dataset: AnalysisDataset) -> pd.DataFrame:
    columns = [
        "sequence_group_id",
        "group_size",
        "protein_id",
        "sequence_sha256",
        "representative",
        "vector_sha256",
        "exact_match_to_representative",
        "max_absolute_difference_to_representative",
    ]
    rows: list[dict[str, Any]] = []
    groups = [
        (digest, indices)
        for digest, indices in sequence_groups(dataset).items()
        if len(indices) > 1
    ]
    groups.sort(key=lambda item: (-len(item[1]), dataset.ids[item[1][0]]))
    for group_id, (digest, indices) in enumerate(groups, start=1):
        representative = min(indices, key=lambda index: dataset.ids[index])
        reference = np.asarray(dataset.X[representative])
        for index in sorted(indices, key=lambda member: dataset.ids[member]):
            vector = np.asarray(dataset.X[index])
            rows.append(
                {
                    "sequence_group_id": group_id,
                    "group_size": len(indices),
                    "protein_id": dataset.ids[index],
                    "sequence_sha256": digest,
                    "representative": index == representative,
                    "vector_sha256": hashlib.sha256(vector.tobytes()).hexdigest(),
                    "exact_match_to_representative": bool(np.array_equal(vector, reference)),
                    "max_absolute_difference_to_representative": float(
                        np.max(
                            np.abs(
                                np.asarray(vector, dtype=np.float64)
                                - np.asarray(reference, dtype=np.float64)
                            )
                        )
                    ),
                }
            )
    return pd.DataFrame(rows, columns=columns)


def scale_profile(dataset: AnalysisDataset) -> dict[str, Any]:
    return profile_metrics.embedding_qc(_profile_dataset(dataset))


def effective_dimension_profile(dataset: AnalysisDataset) -> dict[str, Any]:
    return profile_metrics.effective_dim(np.asarray(dataset.X, dtype=np.float64))


def distance_concentration_profile(
    dataset: AnalysisDataset, euclidean_distances: np.ndarray
) -> dict[str, Any]:
    upper = np.triu_indices(len(dataset), 1)
    return profile_metrics.distance_concentration(euclidean_distances[upper])


def metric_agreement_profile(
    dataset: AnalysisDataset, euclidean_distances: np.ndarray
) -> dict[str, Any]:
    upper = np.triu_indices(len(dataset), 1)
    return profile_metrics.metric_agreement(dataset.X, euclidean_distances[upper], upper)


def anisotropy_profile(dataset: AnalysisDataset) -> dict[str, Any]:
    return profile_metrics.anisotropy(dataset.X)


def density_spread_profile(dataset: AnalysisDataset) -> dict[str, Any]:
    return profile_metrics.density_spread(dataset.X)


def hubness_profile(dataset: AnalysisDataset) -> dict[str, Any]:
    return profile_metrics.hubness(dataset.X)


PROFILE_FUNCTIONS = {
    "scale": scale_profile,
    "sequence_deduplication": sequence_deduplication,
    "vector_determinism": vector_determinism,
    "effective_dimension": effective_dimension_profile,
    "distance_concentration": distance_concentration_profile,
    "metric_agreement": metric_agreement_profile,
    "anisotropy": anisotropy_profile,
    "density_spread": density_spread_profile,
    "hubness": hubness_profile,
}


def embedding_profile(dataset: AnalysisDataset) -> dict[str, dict[str, Any]]:
    euclidean_distances = pairwise_distances(dataset.X, metric="euclidean")
    return {
        "scale": scale_profile(dataset),
        "sequence_deduplication": sequence_deduplication(dataset),
        "vector_determinism": vector_determinism(dataset),
        "effective_dimension": effective_dimension_profile(dataset),
        "distance_concentration": distance_concentration_profile(
            dataset, euclidean_distances
        ),
        "metric_agreement": metric_agreement_profile(dataset, euclidean_distances),
        "anisotropy": anisotropy_profile(dataset),
        "density_spread": density_spread_profile(dataset),
        "hubness": hubness_profile(dataset),
    }


def _relative_to_root(path: Path, root: Path) -> str:
    try:
        return Path(path).resolve().relative_to(Path(root).resolve()).as_posix()
    except ValueError as error:
        raise ValueError(f"analysis input is outside the repository root: {path}") from error


def profile_payload(
    model: ModelDataset,
    dataset: AnalysisDataset,
    groups: dict[str, dict[str, Any]],
    *,
    deduplicated: bool,
    generated_at_utc: str,
) -> dict[str, Any]:
    return {
        "schema": PROFILE_SCHEMA,
        "generated_at_utc": generated_at_utc,
        "model": {
            "key": model.model_key,
            "checkpoint": model.checkpoint,
            "revision": model.revision,
            "stem": model.stem,
            "width": model.width,
        },
        "input": {
            "h5_path": _relative_to_root(model.h5_path, model.repository_root),
            "h5_sha256": model.manifest["artifact"]["sha256"],
            "annotation_path": _relative_to_root(
                model.annotation_path, model.repository_root
            ),
            "fasta_path": _relative_to_root(model.fasta_path, model.repository_root),
            "reference_h5": _relative_to_root(
                model.reference_h5, model.repository_root
            ),
        },
        "record_set": "sequence_deduplicated" if deduplicated else "full",
        "deduplication_rule": "lowest_protein_id_per_exact_sequence_sha256"
        if deduplicated
        else "none",
        "n": len(dataset),
        "subset_hash": dataset.subset_hash,
        "profile": groups,
    }


def render_profile_markdown(payload: dict[str, Any]) -> str:
    model = payload["model"]
    lines = [
        f"# ESM-2 embedding profile - {model['stem']}",
        "",
        f"Generated by `src/esm2_analysis.py profile` on {payload['generated_at_utc']}. "
        "Do not edit by hand.",
        "",
        f"Checkpoint: `{model['checkpoint']}` at revision `{model['revision']}`.",
        f"Record set: `{payload['record_set']}`, n = {payload['n']}, "
        f"subset hash `{payload['subset_hash']}`.",
        "",
        "The values describe this model's native coordinates. Raw internal scores are not used "
        "to rank embedding widths.",
        "",
    ]
    for group_name, values in payload["profile"].items():
        function = PROFILE_FUNCTIONS[group_name]
        lines.extend(
            [
                f"## {group_name}",
                "",
                inspect.cleandoc(function.__doc__ or ""),
                "",
                "| metric | value |",
                "|---|---:|",
                *[f"| `{name}` | {value} |" for name, value in values.items()],
                "",
            ]
        )
    return "\n".join(lines)


def _publish_text_outputs(contents: dict[Path, str]) -> None:
    token = uuid.uuid4().hex
    staged: dict[Path, Path] = {}
    backups: dict[Path, Path] = {}
    promoted: list[Path] = []
    ordered_targets = sorted(contents, key=lambda path: path.as_posix())
    try:
        for target in ordered_targets:
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(f".{target.name}.{token}.tmp")
            temporary.write_text(contents[target], encoding="utf-8", newline="\n")
            staged[target] = temporary
        for target in ordered_targets:
            if target.exists():
                backup = target.with_name(f".{target.name}.{token}.bak")
                os.replace(target, backup)
                backups[target] = backup
        for target in ordered_targets:
            os.replace(staged[target], target)
            promoted.append(target)
    except BaseException:
        for target in promoted:
            target.unlink(missing_ok=True)
        for target, backup in backups.items():
            if backup.exists():
                os.replace(backup, target)
        raise
    finally:
        for temporary in staged.values():
            temporary.unlink(missing_ok=True)
        for backup in backups.values():
            backup.unlink(missing_ok=True)


def write_model_profiles(model: ModelDataset) -> dict[str, dict[str, Any]]:
    generated_at_utc = datetime.now(timezone.utc).isoformat()
    outputs: dict[Path, str] = {}
    payloads: dict[str, dict[str, Any]] = {}
    datasets = {
        "full": model.data,
        "sequence_deduplicated": deduplicate_by_sequence(model.data),
    }
    for record_set, dataset in datasets.items():
        deduplicated = record_set == "sequence_deduplicated"
        groups = embedding_profile(dataset)
        payload = profile_payload(
            model,
            dataset,
            groups,
            deduplicated=deduplicated,
            generated_at_utc=generated_at_utc,
        )
        json_path, markdown_path, duplicate_path = profile_output_paths(
            model, deduplicated=deduplicated
        )
        outputs[json_path] = json.dumps(payload, indent=2) + "\n"
        outputs[markdown_path] = render_profile_markdown(payload)
        outputs[duplicate_path] = duplicate_groups_table(dataset).to_csv(
            index=False, lineterminator="\n"
        )
        payloads[record_set] = payload
    _publish_text_outputs(outputs)
    return payloads


def _canonicalise_datasets(
    datasets: list[AnalysisDataset],
) -> list[AnalysisDataset]:
    if len(datasets) < 2:
        raise ValueError("native geometry requires at least two embedding datasets")
    names = [dataset.name for dataset in datasets]
    if len(names) != len(set(names)):
        raise ValueError("embedding dataset names must be unique")
    expected_ids = set(datasets[0].ids)
    expected_sequences = dict(zip(datasets[0].ids, datasets[0].sequences, strict=True))
    canonical_ids = tuple(sorted(expected_ids))
    aligned: list[AnalysisDataset] = []
    for dataset in datasets:
        if len(dataset.ids) != len(set(dataset.ids)):
            raise ValueError(f"duplicate protein IDs in {dataset.name}")
        ids = set(dataset.ids)
        if ids != expected_ids:
            raise ValueError(
                f"ID mismatch for {dataset.name}: "
                f"{len(expected_ids - ids)} missing, {len(ids - expected_ids)} extra"
            )
        sequence_map = dict(zip(dataset.ids, dataset.sequences, strict=True))
        if sequence_map != expected_sequences:
            raise ValueError(f"sequence mismatch for {dataset.name}")
        index = {protein_id: row for row, protein_id in enumerate(dataset.ids)}
        rows = [index[protein_id] for protein_id in canonical_ids]
        aligned.append(
            AnalysisDataset(
                name=dataset.name,
                ids=canonical_ids,
                X=dataset.X[rows],
                ann=dataset.ann.set_index("protein_id")
                .loc[list(canonical_ids)]
                .reset_index(),
                sequences=tuple(sequence_map[protein_id] for protein_id in canonical_ids),
            )
        )
    return aligned


def _length_subset(dataset: AnalysisDataset, maximum_length: int) -> AnalysisDataset:
    keep = [
        index
        for index, sequence in enumerate(dataset.sequences)
        if len(sequence) <= maximum_length
    ]
    return AnalysisDataset(
        name=dataset.name,
        ids=tuple(dataset.ids[index] for index in keep),
        X=dataset.X[keep],
        ann=dataset.ann.iloc[keep].reset_index(drop=True),
        sequences=tuple(dataset.sequences[index] for index in keep),
    )


def _neighbour_indices(
    matrix: np.ndarray, *, k: int, metric: DistanceMetric
) -> np.ndarray:
    neighbours = (
        NearestNeighbors(n_neighbors=k + 1, metric=metric)
        .fit(matrix)
        .kneighbors(matrix, return_distance=False)
    )
    without_self = np.empty((len(matrix), k), dtype=np.int64)
    for row in range(len(matrix)):
        selected = [index for index in neighbours[row] if index != row][:k]
        if len(selected) != k:
            raise ValueError("nearest-neighbour query did not return k non-self records")
        without_self[row] = selected
    return without_self


def neighbour_overlap(
    first: np.ndarray, second: np.ndarray, *, k: int, metric: DistanceMetric
) -> float:
    if first.shape[0] != second.shape[0]:
        raise ValueError("neighbour overlap requires equal row counts")
    if not 0 < k < first.shape[0]:
        raise ValueError("neighbour k must be positive and smaller than n")
    first_neighbours = _neighbour_indices(first, k=k, metric=metric)
    second_neighbours = _neighbour_indices(second, k=k, metric=metric)
    shared = sum(
        len(set(first_neighbours[row]).intersection(second_neighbours[row]))
        for row in range(first.shape[0])
    )
    return float(shared / (first.shape[0] * k))


def _distance_vector(matrix: np.ndarray, metric: DistanceMetric) -> np.ndarray:
    upper = np.triu_indices(matrix.shape[0], 1)
    return pairwise_distances(matrix, metric=metric)[upper]


def _distance_rank_from_vectors(first: np.ndarray, second: np.ndarray) -> float:
    statistic = float(
        np.asarray(
            getattr(spearmanr(first, second), "statistic"),
            dtype=np.float64,
        ).item()
    )
    if not np.isfinite(statistic):
        raise ValueError("pairwise-distance rank agreement is undefined")
    return statistic


def distance_rank_agreement(
    first: np.ndarray, second: np.ndarray, *, metric: DistanceMetric
) -> float:
    if first.shape[0] != second.shape[0]:
        raise ValueError("distance rank agreement requires equal row counts")
    return _distance_rank_from_vectors(
        _distance_vector(first, metric), _distance_vector(second, metric)
    )


def native_geometry_table(
    datasets: list[AnalysisDataset],
    *,
    neighbour_ks: tuple[int, ...],
    maximum_length: int,
) -> pd.DataFrame:
    if tuple(neighbour_ks) != REQUIRED_NEIGHBOUR_KS:
        raise ValueError(
            f"neighbour_ks must be explicit and equal to {REQUIRED_NEIGHBOUR_KS}"
        )
    if maximum_length <= 0:
        raise ValueError("maximum_length must be positive")
    full = _canonicalise_datasets(datasets)
    record_sets = {
        "full": full,
        f"length_le_{maximum_length}": [
            _length_subset(dataset, maximum_length) for dataset in full
        ],
    }
    rows: list[dict[str, Any]] = []
    for record_set, aligned in record_sets.items():
        subset_ids = aligned[0].ids
        if not subset_ids or any(dataset.ids != subset_ids for dataset in aligned[1:]):
            raise ValueError(f"record set {record_set} is not aligned across models")
        subset_hash = aligned[0].subset_hash
        distance_vectors = {
            (dataset.name, metric): _distance_vector(dataset.X, metric)
            for dataset in aligned
            for metric in ("euclidean", "cosine")
        }
        neighbour_indices = {
            (dataset.name, metric, k): _neighbour_indices(
                dataset.X, k=k, metric=metric
            )
            for dataset in aligned
            for metric in ("euclidean", "cosine")
            for k in neighbour_ks
        }
        for first_index, first in enumerate(aligned[:-1]):
            for second in aligned[first_index + 1:]:
                for metric in ("euclidean", "cosine"):
                    row = {
                        "record_set": record_set,
                        "maximum_sequence_length": maximum_length
                        if record_set != "full"
                        else "",
                        "n": len(first),
                        "subset_hash": subset_hash,
                        "model_a": first.name,
                        "model_b": second.name,
                        "metric": metric,
                        "spearman_distance": _distance_rank_from_vectors(
                            distance_vectors[(first.name, metric)],
                            distance_vectors[(second.name, metric)],
                        ),
                    }
                    for k in neighbour_ks:
                        first_neighbours = neighbour_indices[(first.name, metric, k)]
                        second_neighbours = neighbour_indices[(second.name, metric, k)]
                        shared = sum(
                            len(set(first_neighbours[index]).intersection(
                                second_neighbours[index]
                            ))
                            for index in range(len(first))
                        )
                        row[f"neighbour_overlap_k{k}"] = float(
                            shared / (len(first) * k)
                        )
                    rows.append(row)
    return pd.DataFrame(rows)


def geometry_output_paths(
    output_root: Path, *, repository_root: Path = REPO
) -> tuple[Path, Path]:
    root = validate_comparison_output_root(
        output_root, repository_root=repository_root
    )
    return root / "native_geometry_comparison.csv", root / "native_geometry_comparison.md"


def render_geometry_markdown(
    table: pd.DataFrame,
    *,
    sources: list[dict[str, Any]],
    generated_at_utc: str,
    neighbour_ks: tuple[int, ...],
    maximum_length: int,
) -> str:
    lines = [
        "# ESM-2 native-geometry comparison",
        "",
        f"Generated by `src/esm2_analysis.py compare` on {generated_at_utc}. "
        "Do not edit by hand.",
        "",
        "## Inputs",
        "",
        "| model | checkpoint | width | revision | HDF5 | SHA256 |",
        "|---|---|---:|---|---|---|",
    ]
    for source in sources:
        lines.append(
            f"| `{source['name']}` | `{source['checkpoint']}` | {source['width']} | "
            f"`{source['revision']}` | `{source['h5_path']}` | `{source['h5_sha256']}` |"
        )
    lines.extend(
        [
            "",
            "## Method",
            "",
            inspect.cleandoc(distance_rank_agreement.__doc__ or ""),
            "",
            inspect.cleandoc(neighbour_overlap.__doc__ or ""),
            "",
            f"Both measures use each row's named metric. Neighbour overlap reports explicit "
            f"k values {', '.join(map(str, neighbour_ks))}. The sensitivity subset keeps "
            f"sequences at or below {maximum_length} residues, measured from the registered "
            "FASTA. No raw internal score is compared across embedding widths.",
            "",
            "These are measurements only.",
            "",
            "## Results",
            "",
            "| records | n | model A | model B | metric | Spearman | overlap k=20 | "
            "overlap k=50 |",
            "|---|---:|---|---|---|---:|---:|---:|",
        ]
    )
    for row in table.itertuples(index=False):
        lines.append(
            f"| `{row.record_set}` | {row.n} | `{row.model_a}` | `{row.model_b}` | "
            f"`{row.metric}` | {row.spearman_distance:.6f} | "
            f"{row.neighbour_overlap_k20:.6f} | {row.neighbour_overlap_k50:.6f} |"
        )
    lines.append("")
    return "\n".join(lines)


def write_native_geometry_comparison(
    *,
    reference: AnalysisDataset,
    reference_h5: Path,
    models: list[ModelDataset],
    output_root: Path,
    neighbour_ks: tuple[int, ...],
    maximum_length: int,
    expected_full_count: int,
    expected_length_count: int,
    repository_root: Path = REPO,
) -> pd.DataFrame:
    if len(models) != 2:
        raise ValueError("native geometry requires exactly two ESM-2 models")
    requested_contract = (
        expected_full_count,
        maximum_length,
        expected_length_count,
    )
    frozen_contract = (
        COMPARISON_FULL_COUNT,
        COMPARISON_MAXIMUM_LENGTH,
        COMPARISON_LENGTH_COUNT,
    )
    if requested_contract != frozen_contract:
        raise ValueError(
            "comparison counts and length must equal the frozen contract "
            f"{frozen_contract}"
        )
    if len(reference) != expected_full_count:
        raise ValueError(
            f"full record count is {len(reference)}, expected {expected_full_count}"
        )
    for model in models:
        if len(model.data) != expected_full_count:
            raise ValueError(
                f"full record count for {model.stem} is {len(model.data)}, "
                f"expected {expected_full_count}"
            )
    table = native_geometry_table(
        [reference, *(model.data for model in models)],
        neighbour_ks=neighbour_ks,
        maximum_length=maximum_length,
    )
    measured_counts = table.groupby("record_set")["n"].unique().to_dict()
    if measured_counts.get("full", np.array([])).tolist() != [expected_full_count]:
        raise ValueError("full comparison rows do not have the expected count")
    subset_name = f"length_le_{maximum_length}"
    if measured_counts.get(subset_name, np.array([])).tolist() != [expected_length_count]:
        actual = measured_counts.get(subset_name, np.array([])).tolist()
        raise ValueError(
            f"length-subset comparison count is {actual}, expected {expected_length_count}"
        )

    root = Path(repository_root).resolve()
    reference_h5 = Path(reference_h5).resolve()
    sources = [
        {
            "name": reference.name,
            "checkpoint": "unverified",
            "width": int(reference.X.shape[1]),
            "revision": "unverified",
            "h5_path": _relative_to_root(reference_h5, root),
            "h5_sha256": esm2_embeddings.sha256_file(reference_h5),
        }
    ]
    for model in models:
        sources.append(
            {
                "name": model.stem,
                "checkpoint": model.checkpoint,
                "width": model.width,
                "revision": model.revision,
                "h5_path": _relative_to_root(model.h5_path, root),
                "h5_sha256": model.manifest["artifact"]["sha256"],
            }
        )
    source_by_name = {source["name"]: source for source in sources}
    table.insert(
        list(table.columns).index("model_a") + 1,
        "model_a_h5_sha256",
        table["model_a"].map(
            lambda name: source_by_name[str(name)]["h5_sha256"]
        ),
    )
    table.insert(
        list(table.columns).index("model_b") + 1,
        "model_b_h5_sha256",
        table["model_b"].map(
            lambda name: source_by_name[str(name)]["h5_sha256"]
        ),
    )
    generated_at_utc = datetime.now(timezone.utc).isoformat()
    csv_path, markdown_path = geometry_output_paths(
        output_root, repository_root=repository_root
    )
    _publish_text_outputs(
        {
            csv_path: table.to_csv(index=False, lineterminator="\n", float_format="%.12g"),
            markdown_path: render_geometry_markdown(
                table,
                sources=sources,
                generated_at_utc=generated_at_utc,
                neighbour_ks=neighbour_ks,
                maximum_length=maximum_length,
            ),
        }
    )
    return table


def _add_explicit_input_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--annotation", type=Path, required=True)
    parser.add_argument("--fasta", type=Path, required=True)
    parser.add_argument("--reference-h5", type=Path, required=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    profile_parser = commands.add_parser("profile", help="profile one validated ESM-2 matrix")
    _add_explicit_input_arguments(profile_parser)
    profile_parser.add_argument("--h5", type=Path, required=True)
    profile_parser.add_argument("--output-root", type=Path, required=True)

    compare_parser = commands.add_parser("compare", help="compare the three native spaces")
    _add_explicit_input_arguments(compare_parser)
    compare_parser.add_argument(
        "--embedding",
        nargs=2,
        action="append",
        required=True,
        metavar=("H5", "MODEL_OUTPUT_ROOT"),
    )
    compare_parser.add_argument("--output-root", type=Path, required=True)
    compare_parser.add_argument("--neighbour-k", type=int, action="append", required=True)
    compare_parser.add_argument("--maximum-length", type=int, required=True)
    compare_parser.add_argument("--expected-full-count", type=int, required=True)
    compare_parser.add_argument("--expected-length-count", type=int, required=True)
    arguments = parser.parse_args(argv)

    if arguments.command == "profile":
        model = load_model_dataset(
            h5_path=arguments.h5,
            output_root=arguments.output_root,
            annotation_path=arguments.annotation,
            fasta_path=arguments.fasta,
            reference_h5=arguments.reference_h5,
        )
        payloads = write_model_profiles(model)
        print(
            f"PROFILE PASS: {model.stem}; full n={payloads['full']['n']}; "
            f"sequence-deduplicated n={payloads['sequence_deduplicated']['n']}"
        )
        return

    if len(arguments.embedding) != 2:
        parser.error("compare requires exactly two --embedding H5 MODEL_OUTPUT_ROOT pairs")
    models = [
        load_model_dataset(
            h5_path=Path(h5_path),
            output_root=Path(model_output_root),
            annotation_path=arguments.annotation,
            fasta_path=arguments.fasta,
            reference_h5=arguments.reference_h5,
        )
        for h5_path, model_output_root in arguments.embedding
    ]
    reference = load_aligned_dataset(
        name="prott5",
        h5_path=arguments.reference_h5,
        annotation_path=arguments.annotation,
        fasta_path=arguments.fasta,
    )
    table = write_native_geometry_comparison(
        reference=reference,
        reference_h5=arguments.reference_h5,
        models=models,
        output_root=arguments.output_root,
        neighbour_ks=tuple(arguments.neighbour_k),
        maximum_length=arguments.maximum_length,
        expected_full_count=arguments.expected_full_count,
        expected_length_count=arguments.expected_length_count,
    )
    print(
        f"COMPARISON PASS: {len(table)} rows; "
        f"record sets={','.join(table['record_set'].drop_duplicates())}"
    )


if __name__ == "__main__":
    main()