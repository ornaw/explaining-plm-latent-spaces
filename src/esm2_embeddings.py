from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import numpy as np


REPO = Path(__file__).resolve().parents[1]
GENERATION_ROUTE = "protspace-local"
EXPECTED_RECORD_COUNT = 1587
MAX_LENGTH = 2486
H5_SCHEMA = "esm2-embedding-h5-v1"
MANIFEST_SCHEMA = "esm2-embedding-manifest-v1"
DUPLICATE_ATOL = 1e-6
DUPLICATE_RTOL = 1e-5
PROTSPACE_REVISION = "1bcbec518df16c6964df351d3626c3961a23f73e"
PROTSPACE_SOURCE_SHA256 = {
    "__init__.py": "e876bc62434facba15e69e409f92bea74bbba668cd44efe1472593e548d868b4",
    "biocentral.py": "ff6f20c2ed8e9db0189749d0e8c992138d2a4e28814255908e3d9d01f83691a3",
    "local.py": "085a9c3a70d445b92a418cfef0662e8f67b4f1e438d02abe393b92bdb2301151",
}


@dataclass(frozen=True)
class ModelSpec:
    key: str
    checkpoint: str
    revision: str
    width: int
    final_h5: Path
    result_root: Path


@dataclass(frozen=True)
class ValidationSummary:
    record_count: int
    vector_width: int
    dtype: str
    duplicate_sequence_groups: int
    duplicate_sequence_records: int
    max_duplicate_absolute_difference: float


MODEL_SPECS = {
    "esm2_8m": ModelSpec(
        "esm2_8m",
        "facebook/esm2_t6_8M_UR50D",
        "c731040fcd8d73dceaa04b0a8e6329b345b0f5df",
        320,
        Path("data/phosphatase/phosphatase/esm2_t6_8M_UR50D_full_length.h5"),
        Path("results/esm2/esm2_t6_8M_UR50D"),
    ),
    "esm2_3b": ModelSpec(
        "esm2_3b",
        "facebook/esm2_t36_3B_UR50D",
        "476b639933c8baad5ad09a60ac1a87f987b656fc",
        2560,
        Path("data/phosphatase/phosphatase/esm2_t36_3B_UR50D_full_length.h5"),
        Path("results/esm2/esm2_t36_3B_UR50D"),
    ),
}


def model_spec(model_key: str) -> ModelSpec:
    try:
        return MODEL_SPECS[model_key]
    except KeyError as error:
        raise ValueError(
            f"unknown model key {model_key!r}; choose {', '.join(MODEL_SPECS)}"
        ) from error


def sha256_file(path: Path) -> str:
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read_fasta(path: Path) -> dict[str, str]:
    records: dict[str, str] = {}
    protein_id: str | None = None
    residues: list[str] = []

    def store_record() -> None:
        if protein_id is None:
            return
        sequence = "".join(residues)
        if protein_id in records:
            raise ValueError(f"duplicate FASTA protein ID: {protein_id}")
        if not sequence or not sequence.isalpha() or not sequence.isupper():
            raise ValueError(f"invalid protein sequence: {protein_id}")
        records[protein_id] = sequence

    for raw_line in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line.startswith(">"):
            store_record()
            header_parts = line[1:].split()
            if not header_parts:
                raise ValueError("empty FASTA header")
            header = header_parts[0]
            protein_id = header.split("|")[1] if "|" in header else header
            if "/" in protein_id:
                raise ValueError(f"FASTA protein ID contains '/': {protein_id}")
            residues = []
        elif line:
            if protein_id is None:
                raise ValueError("FASTA sequence appears before its header")
            residues.append(line)
    store_record()
    if not records:
        raise ValueError("FASTA contains no records")
    return records


def sequence_digest(sequences: dict[str, str]) -> str:
    payload = json.dumps(
        sorted(sequences.items()), ensure_ascii=True, separators=(",", ":")
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def validate_request(
    *,
    model_key: str,
    final_h5: Path,
    output_root: Path,
    route: str,
    expected_count: int,
    expected_width: int,
    repository_root: Path = REPO,
) -> ModelSpec:
    spec = model_spec(model_key)
    root = Path(repository_root).resolve()
    if route != GENERATION_ROUTE:
        raise ValueError(f"generation route must be {GENERATION_ROUTE!r}")
    if expected_count != EXPECTED_RECORD_COUNT:
        raise ValueError(f"expected count must be {EXPECTED_RECORD_COUNT}")
    if expected_width != spec.width:
        raise ValueError(f"expected width for {model_key} must be {spec.width}")
    if Path(final_h5).resolve() != (root / spec.final_h5).resolve():
        raise ValueError(f"final HDF5 path is not allowlisted for {model_key}")
    if Path(output_root).resolve() != (root / spec.result_root).resolve():
        raise ValueError(f"result root is not allowlisted for {model_key}")
    return spec


def _input_ids(
    sequences: dict[str, str], reference_h5: Path, expected_count: int
) -> set[str]:
    if len(sequences) != expected_count:
        raise ValueError(
            f"FASTA record count is {len(sequences)}, expected {expected_count}"
        )
    with h5py.File(reference_h5, "r") as store:
        reference_ids = set(store.keys())
        non_datasets = [name for name in store if not isinstance(store[name], h5py.Dataset)]
    if non_datasets:
        raise ValueError(f"reference HDF5 has non-dataset entries: {non_datasets[:5]}")
    fasta_ids = set(sequences)
    if reference_ids != fasta_ids:
        raise ValueError(
            "FASTA and ProtT5 reference IDs differ: "
            f"{len(reference_ids - fasta_ids)} only in reference, "
            f"{len(fasta_ids - reference_ids)} only in FASTA"
        )
    return reference_ids


def _inspect_vectors(
    path: Path,
    sequences: dict[str, str],
    expected_ids: set[str],
    expected_width: int,
) -> ValidationSummary:
    if not Path(path).is_file():
        raise ValueError(f"embedding HDF5 does not exist: {path}")
    with h5py.File(path, "r") as store:
        stored_ids = set(store.keys())
        if stored_ids != expected_ids:
            raise ValueError(
                "embedding IDs differ from the registered set: "
                f"{len(expected_ids - stored_ids)} missing, "
                f"{len(stored_ids - expected_ids)} extra"
            )
        vectors = {}
        for protein_id in sorted(stored_ids):
            dataset = store[protein_id]
            if not isinstance(dataset, h5py.Dataset):
                raise ValueError(f"HDF5 entry is not a dataset: {protein_id}")
            vector = np.asarray(dataset[()])
            if vector.shape != (expected_width,):
                raise ValueError(
                    f"wrong vector width for {protein_id}: {vector.shape}, "
                    f"expected ({expected_width},)"
                )
            if vector.dtype != np.dtype("float32"):
                raise ValueError(
                    f"wrong vector dtype for {protein_id}: {vector.dtype}, expected float32"
                )
            if not np.isfinite(vector).all():
                raise ValueError(f"non-finite vector: {protein_id}")
            if not np.any(vector):
                raise ValueError(f"all-zero vector: {protein_id}")
            vectors[protein_id] = vector

    groups: dict[str, list[str]] = {}
    for protein_id in sorted(stored_ids):
        groups.setdefault(sequences[protein_id], []).append(protein_id)
    duplicate_groups = [members for members in groups.values() if len(members) > 1]
    maximum_difference = 0.0
    for members in duplicate_groups:
        reference = vectors[members[0]]
        for protein_id in members[1:]:
            difference = float(np.max(np.abs(reference - vectors[protein_id])))
            maximum_difference = max(maximum_difference, difference)
            if not np.allclose(
                reference,
                vectors[protein_id],
                atol=DUPLICATE_ATOL,
                rtol=DUPLICATE_RTOL,
            ):
                raise ValueError(
                    "repeated complete sequence exceeds the determinism tolerance: "
                    f"{members[0]} versus {protein_id}, max absolute difference {difference}"
                )
    return ValidationSummary(
        len(stored_ids),
        expected_width,
        "float32",
        len(duplicate_groups),
        sum(len(members) for members in duplicate_groups),
        maximum_difference,
    )


def validate_vectors(
    path: Path,
    fasta_path: Path,
    reference_h5: Path,
    expected_count: int,
    expected_width: int,
) -> tuple[dict[str, str], ValidationSummary]:
    sequences = read_fasta(fasta_path)
    expected_ids = _input_ids(sequences, reference_h5, expected_count)
    return sequences, _inspect_vectors(path, sequences, expected_ids, expected_width)


def generation_metadata(
    spec: ModelSpec,
    *,
    fasta_path: Path,
    sequences: dict[str, str],
    route: str,
    generation_seconds: float,
    software_versions: dict[str, str],
    gpu: str,
) -> dict[str, Any]:
    return {
        "schema": H5_SCHEMA,
        "model_key": spec.key,
        "checkpoint": spec.checkpoint,
        "checkpoint_revision": spec.revision,
        "vector_width": spec.width,
        "dtype": "float32",
        "generation_route": route,
        "protspace_revision": PROTSPACE_REVISION,
        "protspace_source_sha256_json": json.dumps(
            PROTSPACE_SOURCE_SHA256, sort_keys=True, separators=(",", ":")
        ),
        "representation_layer": "final_hidden_layer",
        "pooling": "residue_mean_amino_acid_tokens_only",
        "special_tokens": "excluded",
        "max_length": MAX_LENGTH,
        "sequence_policy": "complete_sequences",
        "record_count": len(sequences),
        "fasta_sha256": sha256_file(fasta_path),
        "sequence_digest": sequence_digest(sequences),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "generation_seconds": float(generation_seconds),
        "software_versions_json": json.dumps(
            software_versions, sort_keys=True, separators=(",", ":")
        ),
        "gpu": gpu,
    }


def write_h5_metadata(path: Path, metadata: dict[str, Any]) -> None:
    with h5py.File(path, "r+") as store:
        for name, value in metadata.items():
            store.attrs[name] = value
        store.flush()


def validate_h5_metadata(
    path: Path,
    spec: ModelSpec,
    *,
    fasta_path: Path,
    sequences: dict[str, str],
    route: str,
) -> dict[str, Any]:
    with h5py.File(path, "r") as store:
        metadata = {
            name: value.item() if isinstance(value, np.generic) else value
            for name, value in store.attrs.items()
        }
    try:
        generation_seconds = float(metadata["generation_seconds"])
        generated_at = datetime.fromisoformat(str(metadata["generated_at_utc"]))
        source_hashes = json.loads(metadata["protspace_source_sha256_json"])
        software_versions = json.loads(metadata["software_versions_json"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError("HDF5 provenance metadata is invalid") from error
    expected = generation_metadata(
        spec,
        fasta_path=fasta_path,
        sequences=sequences,
        route=route,
        generation_seconds=generation_seconds,
        software_versions=software_versions,
        gpu=str(metadata.get("gpu", "")),
    )
    expected["generated_at_utc"] = metadata["generated_at_utc"]
    mismatches = {
        name: (metadata.get(name), value)
        for name, value in expected.items()
        if metadata.get(name) != value
    }
    if mismatches:
        raise ValueError(f"HDF5 metadata mismatch: {mismatches}")
    if (
        generated_at.utcoffset() != timezone.utc.utcoffset(generated_at)
        or not np.isfinite(generation_seconds)
        or generation_seconds < 0
        or not str(metadata.get("gpu", "")).strip()
    ):
        raise ValueError("HDF5 generation time metadata is invalid")
    if source_hashes != PROTSPACE_SOURCE_SHA256:
        raise ValueError("HDF5 ProtSpace source hashes do not match the pinned route")
    if not {"torch", "transformers", "h5py", "numpy"} <= set(software_versions):
        raise ValueError("HDF5 software metadata is incomplete")
    metadata["protspace_source_sha256"] = source_hashes
    metadata["software_versions"] = software_versions
    return metadata


def _validate_temporary_path(path: Path, final_h5: Path, repository_root: Path) -> None:
    temporary = Path(path).resolve()
    root = Path(repository_root).resolve()
    if temporary == Path(final_h5).resolve():
        raise ValueError("temporary HDF5 must differ from the final HDF5")
    if temporary.is_relative_to(root) and not temporary.is_relative_to(root / "tmp"):
        raise ValueError("a repository-local temporary HDF5 must be under tmp/")


def _generation_manifest(metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        "route": metadata["generation_route"],
        "generated_at_utc": metadata["generated_at_utc"],
        "generation_seconds": float(metadata["generation_seconds"]),
        "gpu": metadata["gpu"],
        "software_versions": metadata["software_versions"],
        "protspace_revision": metadata["protspace_revision"],
        "protspace_source_sha256": metadata["protspace_source_sha256"],
        "representation_layer": metadata["representation_layer"],
        "pooling": metadata["pooling"],
        "special_tokens": metadata["special_tokens"],
        "sequence_policy": metadata["sequence_policy"],
        "max_length": int(metadata["max_length"]),
    }


def _manifest_payload(
    *,
    spec: ModelSpec,
    h5_path: Path,
    fasta_path: Path,
    reference_h5: Path,
    metadata: dict[str, Any],
    summary: ValidationSummary,
    repository_root: Path,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    return {
        "schema": MANIFEST_SCHEMA,
        "model": {
            "key": spec.key,
            "checkpoint": spec.checkpoint,
            "revision": spec.revision,
            "width": spec.width,
        },
        "input": {
            "fasta_path": Path(fasta_path).resolve().relative_to(root).as_posix(),
            "fasta_sha256": metadata["fasta_sha256"],
            "sequence_digest": metadata["sequence_digest"],
            "reference_h5": Path(reference_h5).resolve().relative_to(root).as_posix(),
            "record_count": summary.record_count,
        },
        "generation": _generation_manifest(metadata),
        "artifact": {
            "path": spec.final_h5.as_posix(),
            "sha256": sha256_file(h5_path),
            "record_count": summary.record_count,
            "vector_width": summary.vector_width,
            "dtype": summary.dtype,
        },
        "validation": {
            "validated_at_utc": datetime.now(timezone.utc).isoformat(),
            "duplicate_sequence_groups": summary.duplicate_sequence_groups,
            "duplicate_sequence_records": summary.duplicate_sequence_records,
            "max_duplicate_absolute_difference": summary.max_duplicate_absolute_difference,
            "atol": DUPLICATE_ATOL,
            "rtol": DUPLICATE_RTOL,
            "python": platform.python_version(),
            "h5py": h5py.__version__,
            "numpy": np.__version__,
        },
    }


def _validated_file(
    *,
    spec: ModelSpec,
    h5_path: Path,
    fasta_path: Path,
    reference_h5: Path,
    route: str,
    expected_count: int,
    expected_width: int,
) -> tuple[ValidationSummary, dict[str, Any]]:
    sequences, summary = validate_vectors(
        h5_path, fasta_path, reference_h5, expected_count, expected_width
    )
    metadata = validate_h5_metadata(
        h5_path, spec, fasta_path=fasta_path, sequences=sequences, route=route
    )
    return summary, metadata


def validate_and_promote(
    *,
    model_key: str,
    fasta_path: Path,
    reference_h5: Path,
    temporary_h5: Path,
    final_h5: Path,
    output_root: Path,
    route: str,
    expected_count: int,
    expected_width: int,
    colab_sha256: str,
    repository_root: Path = REPO,
) -> dict[str, Any]:
    spec = validate_request(
        model_key=model_key,
        final_h5=final_h5,
        output_root=output_root,
        route=route,
        expected_count=expected_count,
        expected_width=expected_width,
        repository_root=repository_root,
    )
    temporary_h5 = Path(temporary_h5)
    _validate_temporary_path(temporary_h5, final_h5, repository_root)
    digest = sha256_file(temporary_h5)
    reported_digest = colab_sha256.strip().lower()
    if len(reported_digest) != 64 or any(
        character not in "0123456789abcdef" for character in reported_digest
    ):
        raise ValueError("Colab SHA256 must be 64 hexadecimal characters")
    if digest != reported_digest:
        raise ValueError("HDF5 SHA256 differs from the digest printed in Colab")
    summary, metadata = _validated_file(
        spec=spec,
        h5_path=temporary_h5,
        fasta_path=fasta_path,
        reference_h5=reference_h5,
        route=route,
        expected_count=expected_count,
        expected_width=expected_width,
    )

    final_h5 = Path(final_h5)
    output_root = Path(output_root)
    manifest_path = output_root / "embedding_manifest.json"
    if final_h5.exists() or manifest_path.exists():
        raise FileExistsError("refusing to replace an existing final HDF5 or manifest")
    final_h5.parent.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)
    suffix = uuid.uuid4().hex
    staged_h5 = final_h5.with_name(f".{final_h5.name}.{suffix}.tmp")
    staged_manifest = manifest_path.with_name(f".{manifest_path.name}.{suffix}.tmp")
    published_h5 = False
    published_manifest = False
    try:
        shutil.copy2(temporary_h5, staged_h5)
        if sha256_file(staged_h5) != digest:
            raise ValueError("staged HDF5 digest changed during the local copy")
        payload = _manifest_payload(
            spec=spec,
            h5_path=staged_h5,
            fasta_path=fasta_path,
            reference_h5=reference_h5,
            metadata=metadata,
            summary=summary,
            repository_root=repository_root,
        )
        staged_manifest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        os.replace(staged_h5, final_h5)
        published_h5 = True
        os.replace(staged_manifest, manifest_path)
        published_manifest = True
        validate_persisted(
            model_key=model_key,
            fasta_path=fasta_path,
            reference_h5=reference_h5,
            final_h5=final_h5,
            output_root=output_root,
            route=route,
            expected_count=expected_count,
            expected_width=expected_width,
            repository_root=repository_root,
        )
    except BaseException:
        if published_manifest:
            manifest_path.unlink(missing_ok=True)
        if published_h5:
            final_h5.unlink(missing_ok=True)
        raise
    finally:
        staged_h5.unlink(missing_ok=True)
        staged_manifest.unlink(missing_ok=True)
    temporary_h5.unlink()
    return payload


def validate_persisted(
    *,
    model_key: str,
    fasta_path: Path,
    reference_h5: Path,
    final_h5: Path,
    output_root: Path,
    route: str,
    expected_count: int,
    expected_width: int,
    repository_root: Path = REPO,
) -> dict[str, Any]:
    spec = validate_request(
        model_key=model_key,
        final_h5=final_h5,
        output_root=output_root,
        route=route,
        expected_count=expected_count,
        expected_width=expected_width,
        repository_root=repository_root,
    )
    payload = json.loads(
        (Path(output_root) / "embedding_manifest.json").read_text(encoding="utf-8")
    )
    if payload.get("schema") != MANIFEST_SCHEMA:
        raise ValueError("embedding manifest schema mismatch")
    summary, metadata = _validated_file(
        spec=spec,
        h5_path=final_h5,
        fasta_path=fasta_path,
        reference_h5=reference_h5,
        route=route,
        expected_count=expected_count,
        expected_width=expected_width,
    )
    expected = _manifest_payload(
        spec=spec,
        h5_path=final_h5,
        fasta_path=fasta_path,
        reference_h5=reference_h5,
        metadata=metadata,
        summary=summary,
        repository_root=repository_root,
    )
    for section in ("model", "input", "generation", "artifact"):
        if payload.get(section) != expected[section]:
            raise ValueError(f"embedding manifest {section} does not match the HDF5")
    for name in (
        "duplicate_sequence_groups",
        "duplicate_sequence_records",
        "max_duplicate_absolute_difference",
        "atol",
        "rtol",
    ):
        if payload.get("validation", {}).get(name) != expected["validation"][name]:
            raise ValueError("embedding manifest validation does not match the HDF5")
    return payload


def _add_contract_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", choices=tuple(MODEL_SPECS), required=True)
    parser.add_argument("--fasta", type=Path, required=True)
    parser.add_argument("--reference-h5", type=Path, required=True)
    parser.add_argument("--final-h5", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--route", choices=(GENERATION_ROUTE,), required=True)
    parser.add_argument("--expected-count", type=int, required=True)
    parser.add_argument("--expected-width", type=int, required=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    validate_parser = commands.add_parser("validate", help="validate and publish a download")
    verify_parser = commands.add_parser("verify", help="revalidate a published file")
    for command_parser in (validate_parser, verify_parser):
        _add_contract_arguments(command_parser)
    validate_parser.add_argument("--temporary-h5", type=Path, required=True)
    validate_parser.add_argument("--colab-sha256", required=True)
    arguments = parser.parse_args(argv)
    contract = {
        "model_key": arguments.model,
        "fasta_path": arguments.fasta,
        "reference_h5": arguments.reference_h5,
        "final_h5": arguments.final_h5,
        "output_root": arguments.output_root,
        "route": arguments.route,
        "expected_count": arguments.expected_count,
        "expected_width": arguments.expected_width,
    }
    if arguments.command == "validate":
        payload = validate_and_promote(
            **contract,
            temporary_h5=arguments.temporary_h5,
            colab_sha256=arguments.colab_sha256,
        )
        label = "VALIDATION"
    else:
        payload = validate_persisted(**contract)
        label = "VERIFICATION"
    print(
        f"{label} PASS: {payload['artifact']['record_count']} vectors; "
        f"SHA256 {payload['artifact']['sha256']}"
    )


if __name__ == "__main__":
    main()