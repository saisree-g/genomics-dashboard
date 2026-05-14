"""
create_sample.py — Run this ONCE locally to extract a clean sample dataset.

Reads local FASTA files → preprocesses → saves 500 sequences per class
to data/sample.csv (committed to git, used by Streamlit Cloud).

Usage:
    python create_sample.py
"""

import random
import pandas as pd
from pathlib import Path
from Bio import SeqIO
from preprocess import preprocess_pool

DATA_DIR    = Path(__file__).parent / "data"
OUTPUT_FILE = DATA_DIR / "sample.csv"
N_PER_CLASS = 500
SEQ_LENGTH  = 1000
RANDOM_SEED = 42

FILES = {
    "Coding":     DATA_DIR / "cds.fasta",
    "lncRNA":     DATA_DIR / "lnc.fasta",
    "Pseudogene": DATA_DIR / "psd.fasta",
}

random.seed(RANDOM_SEED)
rows = []

for class_name, path in FILES.items():
    print(f"Processing {class_name} from {path.name}...")
    raw = [str(r.seq) for r in SeqIO.parse(str(path), "fasta")]
    clean, report = preprocess_pool(raw, SEQ_LENGTH)
    sample = random.sample(clean, min(N_PER_CLASS, len(clean)))
    for seq in sample:
        rows.append({"class": class_name, "sequence": seq})
    print(f"  {report['raw']} raw → {report['final']} clean → {len(sample)} sampled")

df = pd.DataFrame(rows)
df.to_csv(OUTPUT_FILE, index=False)
print(f"\nSaved {len(df)} sequences to {OUTPUT_FILE} ({OUTPUT_FILE.stat().st_size / 1024:.1f} KB)")
