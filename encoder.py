"""
encoder.py — Genomic sequence encoding schemes.

one_hot_encode(seq)         → (L, 4)  binary matrix
voss_encode(seq)            → (L, 4)  same representation, treated as 4 signals
integer_encode(seq)         → (L, 1)  integer per nucleotide
kmer_encode(seq, k)         → (4^k,)  normalised k-mer frequency vector

encode_sequence(seq, method, k) → dispatches to one of the above
encode_dataset(seqs_dict, method, k) → (X, y) arrays ready for training
"""

from itertools import product

import numpy as np

# ── LOOKUP TABLES ─────────────────────────────────────────────────────────────
_ONE_HOT = {
    "A": [1, 0, 0, 0],
    "C": [0, 1, 0, 0],
    "G": [0, 0, 1, 0],
    "T": [0, 0, 0, 1],
    "N": [0, 0, 0, 0],
}
_INTEGER = {"A": 1, "C": 2, "G": 3, "T": 4, "N": 0}
_KMER_CACHE: dict[int, list[str]] = {}


def _kmer_list(k: int) -> list[str]:
    if k not in _KMER_CACHE:
        _KMER_CACHE[k] = ["".join(p) for p in product("ACGT", repeat=k)]
    return _KMER_CACHE[k]


# ── ENCODING FUNCTIONS ────────────────────────────────────────────────────────

def one_hot_encode(seq: str) -> np.ndarray:
    """Returns shape (L, 4) — A/C/G/T binary indicators."""
    return np.array([_ONE_HOT.get(c, [0, 0, 0, 0]) for c in seq], dtype=np.float32)


def voss_encode(seq: str) -> np.ndarray:
    """
    Voss representation: same (L, 4) layout as one-hot but each column
    is treated as an independent binary signal in the frequency domain.
    Identical values to one_hot for ACGT; kept separate for conceptual clarity.
    """
    return one_hot_encode(seq)


def integer_encode(seq: str) -> np.ndarray:
    """Returns shape (L, 1) — A=1, C=2, G=3, T=4, N=0."""
    arr = np.array([_INTEGER.get(c, 0) for c in seq], dtype=np.float32)
    return arr.reshape(-1, 1)


def kmer_encode(seq: str, k: int = 3) -> np.ndarray:
    """
    Returns shape (4^k,) — normalised k-mer frequency vector.
    Ignores k-mers containing 'N'.
    """
    kmers = _kmer_list(k)
    idx   = {km: i for i, km in enumerate(kmers)}
    counts = np.zeros(len(kmers), dtype=np.float32)
    for i in range(len(seq) - k + 1):
        km = seq[i : i + k]
        if km in idx:
            counts[idx[km]] += 1
    total = counts.sum()
    if total > 0:
        counts /= total
    return counts


def encode_sequence(seq: str, method: str, k: int = 3) -> np.ndarray:
    """Dispatch to the correct encoding function."""
    if method in ("One-Hot", "Voss"):
        # Both use the same matrix; conceptually distinct
        return one_hot_encode(seq) if method == "One-Hot" else voss_encode(seq)
    if method == "Integer":
        return integer_encode(seq)
    if method == "K-mer":
        return kmer_encode(seq, k)
    raise ValueError(f"Unknown encoding method: {method!r}")


def encode_dataset(
    sequences_dict: dict[str, list[str]],
    method: str,
    k: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Encode all sequences and build (X, y) arrays.

    y labels: 0 = Coding, 1 = lncRNA, 2 = Pseudogene
    X shape:
        One-Hot / Voss  → (N, L, 4)
        Integer         → (N, L, 1)
        K-mer           → (N, 4^k)
    """
    label_map = {"Coding": 0, "lncRNA": 1, "Pseudogene": 2}
    X_list, y_list = [], []
    for class_name, seqs in sequences_dict.items():
        label = label_map[class_name]
        for seq in seqs:
            X_list.append(encode_sequence(seq, method, k))
            y_list.append(label)
    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.int32)
