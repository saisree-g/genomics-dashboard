"""
config.py — All constants for the Genomics Sequence Classifier dashboard.
"""

# ── API ──────────────────────────────────────────────────────────────────────
ENTREZ_EMAIL        = "genomics.dashboard@example.com"
RNACENTRAL_API      = "https://rnacentral.org/api/v1/rna/"

# NCBI Entrez search queries
NCBI_CODING_QUERY      = (
    '"Homo sapiens"[Organism] AND mRNA[Filter] AND CDS[Feature Key] '
    'AND 150:600[SLEN] NOT predicted[Title]'
)
NCBI_PSEUDOGENE_QUERY  = (
    '"Homo sapiens"[Organism] AND pseudogene[Title] '
    'AND 150:600[SLEN] NOT predicted[Title]'
)

# RNAcentral query params (taxon 9606 = Homo sapiens)
RNACENTRAL_PARAMS = {
    "organism":   9606,
    "rna_type":   "lncRNA",
    "format":     "json",
    "page_size":  100,
    "min_length": 150,
    "max_length": 600,
}

# ── CLASSES ──────────────────────────────────────────────────────────────────
CLASS_LABELS = ["Coding", "lncRNA", "Pseudogene"]
CLASS_MAP    = {"Coding": 0, "lncRNA": 1, "Pseudogene": 2}
CLASS_COLORS = {
    "Coding":     "#1A4D3A",
    "lncRNA":     "#0077b6",
    "Pseudogene": "#e63946",
}

# ── ENCODING ─────────────────────────────────────────────────────────────────
NUCLEOTIDES   = list("ACGT")
ENCODING_OPTIONS = ["One-Hot", "Voss", "Integer", "K-mer"]

# ── PREPROCESSING ────────────────────────────────────────────────────────────
MIN_ACGT_RATIO     = 0.95   # drop sequences with >5% non-ACGT bases
MIN_SEQ_LEN        = 100    # drop sequences shorter than this after cleaning

# ── MODEL DEFAULTS ───────────────────────────────────────────────────────────
DEFAULT_SEQ_LEN     = 1000
DEFAULT_N_PER_CLASS = 500
DEFAULT_EPOCHS      = 30
DEFAULT_BATCH_SIZE  = 32
DEFAULT_FILTERS     = 64
DEFAULT_KERNEL_SIZE = 5
DEFAULT_GRU_UNITS   = 64
DEFAULT_DROPOUT     = 0.3
DEFAULT_K           = 3
