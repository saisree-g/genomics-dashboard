"""
api.py — Load genomic sequences from local FASTA files or sample.csv fallback.

Local development  → reads cds.fasta / lnc.fasta / psd.fasta (full datasets)
Streamlit Cloud    → reads data/sample.csv (500 pre-cleaned sequences/class,
                     committed to the repo)

fetch_all()  → dict[str, list[str]], dict[str, dict]
"""

import random
from pathlib import Path

import pandas as pd
import streamlit as st
from Bio import SeqIO

from preprocess import preprocess_pool

_DATA_DIR = Path(__file__).parent / "data"
_FASTA_FILES = {
    "Coding":     _DATA_DIR / "cds.fasta",
    "lncRNA":     _DATA_DIR / "lnc.fasta",
    "Pseudogene": _DATA_DIR / "psd.fasta",
}
_SAMPLE_CSV = _DATA_DIR / "sample.csv"


# ─────────────────────────────────────────────────────────────────────────────
# LOADERS
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=None, show_spinner=False)
def _load_from_fasta(class_name: str, seq_length: int) -> tuple[list[str], dict]:
    """Full FASTA load + preprocessing pipeline. Used when local files exist."""
    path = _FASTA_FILES[class_name]
    raw  = [str(r.seq) for r in SeqIO.parse(str(path), "fasta")]
    return preprocess_pool(raw, seq_length)


@st.cache_data(ttl=None, show_spinner=False)
def _load_from_csv(seq_length: int) -> dict[str, list[str]]:
    """
    Load pre-cleaned sample.csv. Used on Streamlit Cloud where FASTA
    files are not present. Truncates/pads sequences to seq_length.
    """
    if not _SAMPLE_CSV.exists():
        st.error(
            "Neither FASTA files nor sample.csv found. "
            "Run `python create_sample.py` locally and commit data/sample.csv."
        )
        return {"Coding": [], "lncRNA": [], "Pseudogene": []}

    df = pd.read_csv(_SAMPLE_CSV)
    result = {}
    for cls in ["Coding", "lncRNA", "Pseudogene"]:
        seqs = df[df["class"] == cls]["sequence"].tolist()
        # Truncate to requested seq_length
        result[cls] = [s[:seq_length] for s in seqs if len(s) >= seq_length]
    return result


def _fasta_available() -> bool:
    return all(p.exists() for p in _FASTA_FILES.values())


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────

def fetch_all(
    n_per_class: int, seq_length: int
) -> tuple[dict[str, list[str]], dict[str, dict]]:
    """
    Return (sequences_dict, preprocessing_reports).
    Automatically uses FASTA files locally or sample.csv on Streamlit Cloud.
    """
    if _fasta_available():
        seqs, reports = {}, {}
        for cls in ["Coding", "lncRNA", "Pseudogene"]:
            pool, report = _load_from_fasta(cls, seq_length)
            seqs[cls]    = random.sample(pool, min(n_per_class, len(pool)))
            reports[cls] = report
        return seqs, reports

    # Fallback: sample.csv
    st.info("Local FASTA files not found — loading from pre-built sample dataset.")
    pool    = _load_from_csv(seq_length)
    seqs    = {cls: random.sample(s, min(n_per_class, len(s))) for cls, s in pool.items()}
    reports = {
        cls: {"raw": len(pool[cls]), "final": len(seqs[cls]),
              "dropped_low_quality": 0, "dropped_duplicates": 0}
        for cls in seqs
    }
    return seqs, reports
