"""
api.py — Load genomic sequences from local FASTA files.

Data files (in data/ folder):
    cds.fasta   → 123,410 human coding sequences
    lnc.fasta   → 17,012  human lncRNA sequences
    psd.fasta   → 7,868   human pseudogene sequences

fetch_coding_sequences()     → list[str]
fetch_lncrna_sequences()     → list[str]
fetch_pseudogene_sequences() → list[str]
fetch_all()                  → dict[str, list[str]]
"""

import random
from pathlib import Path

import streamlit as st
from Bio import SeqIO

from preprocess import preprocess_pool

# ── FILE PATHS ────────────────────────────────────────────────────────────────
_DATA_DIR = Path(__file__).parent / "data"
_FILES = {
    "Coding":     _DATA_DIR / "cds.fasta",
    "lncRNA":     _DATA_DIR / "lnc.fasta",
    "Pseudogene": _DATA_DIR / "psd.fasta",
}


@st.cache_data(ttl=None, show_spinner=False)
def _load_fasta(class_name: str, seq_length: int) -> tuple[list[str], dict]:
    """
    Read FASTA file → run full preprocessing pipeline → return clean sequences.
    Cached permanently since data files don't change between runs.
    """
    path = _FILES[class_name]
    if not path.exists():
        st.error(f"File not found: {path}")
        return [], {}

    raw = [str(r.seq) for r in SeqIO.parse(str(path), "fasta")]
    sequences, report = preprocess_pool(raw, seq_length)
    return sequences, report


# ── PUBLIC FETCH FUNCTIONS ────────────────────────────────────────────────────

def fetch_coding_sequences(n: int = 500, seq_length: int = 1000) -> tuple[list[str], dict]:
    pool, report = _load_fasta("Coding", seq_length)
    return random.sample(pool, min(n, len(pool))), report


def fetch_lncrna_sequences(n: int = 500, seq_length: int = 1000) -> tuple[list[str], dict]:
    pool, report = _load_fasta("lncRNA", seq_length)
    return random.sample(pool, min(n, len(pool))), report


def fetch_pseudogene_sequences(n: int = 500, seq_length: int = 1000) -> tuple[list[str], dict]:
    pool, report = _load_fasta("Pseudogene", seq_length)
    return random.sample(pool, min(n, len(pool))), report


def fetch_all(n_per_class: int, seq_length: int) -> tuple[dict[str, list[str]], dict]:
    """Return sequences + preprocessing reports for all three classes."""
    seqs, reports = {}, {}
    for cls, fn in [
        ("Coding",     fetch_coding_sequences),
        ("lncRNA",     fetch_lncrna_sequences),
        ("Pseudogene", fetch_pseudogene_sequences),
    ]:
        seqs[cls], reports[cls] = fn(n_per_class, seq_length)
    return seqs, reports
