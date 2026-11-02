from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import DATASETS, read_fasta

FASTA = DATASETS["phosphatase"][2]


def sequence_lengths(path: Path = FASTA) -> pd.Series:
    lengths = pd.Series({accession: len(seq) for accession, seq in read_fasta(path).items()},
                        name="length", dtype=int)
    if (lengths <= 0).any():
        raise ValueError("a sequence in the FASTA is empty")
    return lengths


def length_partition(protein_ids, k: int, lengths: pd.Series | None = None) -> np.ndarray:
    ids = np.asarray(protein_ids, dtype=str)
    if ids.ndim != 1 or len(set(ids.tolist())) != len(ids):
        raise ValueError("need unique protein IDs")
    if not 2 <= int(k) <= len(ids):
        raise ValueError(f"K must be between 2 and {len(ids)}, got {k}")
    lengths = sequence_lengths() if lengths is None else lengths
    missing = sorted(set(ids.tolist()) - set(lengths.index))
    if missing:
        raise ValueError(f"{len(missing)} protein IDs have no sequence length, e.g. {missing[:3]}")
    order = np.lexsort((ids, lengths.loc[ids].to_numpy()))
    rank = np.empty(len(ids), dtype=np.int64)
    rank[order] = np.arange(len(ids))
    return rank * int(k) // len(ids)
