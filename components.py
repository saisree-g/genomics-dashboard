"""
components.py — Shared UI components and sidebar.

CSS         : injected once at startup
kpi_row()   : horizontal KPI card row
render_sidebar() : all sidebar controls, returns cfg dict
"""

import streamlit as st

from config import (
    DEFAULT_BATCH_SIZE, DEFAULT_DROPOUT, DEFAULT_EPOCHS,
    DEFAULT_FILTERS, DEFAULT_GRU_UNITS, DEFAULT_K,
    DEFAULT_KERNEL_SIZE, DEFAULT_N_PER_CLASS, DEFAULT_SEQ_LEN,
    ENCODING_OPTIONS,
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────
CSS = """
<style>
    [data-testid="stAppViewContainer"] { background: #f5f7fa; }

    /* Sidebar */
    [data-testid="stSidebar"] { background: #0d1b2a !important; }
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] div,
    [data-testid="stSidebar"] a,
    [data-testid="stSidebar"] .stMarkdown { color: #e0e6ed !important; }
    [data-testid="stSidebar"] .stButton > button {
        background: #1A4D3A; color: white; border: none;
        border-radius: 6px; font-weight: 600; width: 100%;
    }

    /* Main content */
    section.main * { color: #0d1b2a; }
    section.main h1, section.main h2, section.main h3 { color: #0d1b2a !important; }

    /* KPI cards */
    .kpi-card {
        background: #ffffff; border-radius: 10px; padding: 14px 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.07); text-align: center;
        border-top: 3px solid #1A4D3A; min-width: 0;
    }
    .kpi-value {
        font-size: 1.8rem; font-weight: 700; color: #0d1b2a !important;
        line-height: 1.1; white-space: nowrap;
    }
    .kpi-label {
        font-size: 0.66rem; color: #6c757d !important; text-transform: uppercase;
        letter-spacing: 0.04em; margin-top: 5px;
        white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
    }

    /* Section headers */
    .section-header {
        font-size: 1.05rem; font-weight: 600; color: #0d1b2a !important;
        border-bottom: 2px solid #1A4D3A; padding-bottom: 6px;
        margin-bottom: 14px;
    }

    /* Info / insight boxes */
    .insight-box {
        background: #e8f4fd; border-left: 4px solid #1A4D3A;
        padding: 12px 16px; border-radius: 0 6px 6px 0; margin: 10px 0;
        font-size: 0.88rem; color: #1a3a50 !important;
    }

    /* Dashboard title */
    .dash-title {
        font-size: 1.9rem; font-weight: 700; color: #0d1b2a;
        margin-bottom: 4px; line-height: 1.2;
    }
    .dash-subtitle { font-size: 0.88rem; color: #4a5568; margin-bottom: 0; }

    div[data-testid="stMetric"] { display: none; }
</style>
"""


# ─────────────────────────────────────────────────────────────────────────────
# KPI ROW
# ─────────────────────────────────────────────────────────────────────────────

def kpi_row(metrics: list[tuple[str, str]]) -> None:
    """Render a horizontal strip of KPI cards. Each item is (label, value)."""
    cols = st.columns(len(metrics))
    for col, (label, value) in zip(cols, metrics):
        col.markdown(
            f'<div class="kpi-card">'
            f'<div class="kpi-value">{value}</div>'
            f'<div class="kpi-label">{label}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

def render_sidebar() -> dict:
    """Render all sidebar controls and return a config dict."""
    st.sidebar.markdown("## 🧬 Genomics Classifier")
    st.sidebar.markdown("---")

    # ── Data ──────────────────────────────────────────────────────────────────
    st.sidebar.markdown("### Data")
    n_per_class = st.sidebar.slider(
        "Sequences per class", 50, 400, DEFAULT_N_PER_CLASS, 50,
        help="Fetched from NCBI (Coding / Pseudogene) and RNAcentral (lncRNA).",
    )
    seq_length = st.sidebar.slider(
        "Sequence length (bp)", 100, 500, DEFAULT_SEQ_LEN, 50,
        help="Sequences are truncated or N-padded to this length.",
    )

    # ── Encoding ──────────────────────────────────────────────────────────────
    st.sidebar.markdown("### Encoding")
    encoding = st.sidebar.selectbox(
        "Encoding method",
        ENCODING_OPTIONS,
        help=(
            "One-Hot / Voss → (L × 4) matrix · "
            "Integer → (L × 1) · "
            "K-mer → frequency vector (4^k)"
        ),
    )
    k = DEFAULT_K
    if encoding == "K-mer":
        k = st.sidebar.slider("K (k-mer length)", 2, 5, DEFAULT_K)

    # ── Model ─────────────────────────────────────────────────────────────────
    st.sidebar.markdown("### Model")
    architecture = st.sidebar.selectbox(
        "Architecture",
        ["CNN", "CNN + BiGRU"],
        help="CNN + BiGRU replicates the IEEE paper hybrid model.",
    )
    epochs     = st.sidebar.slider("Epochs", 5, 50, DEFAULT_EPOCHS)
    batch_size = st.sidebar.select_slider(
        "Batch size", [16, 32, 64, 128], DEFAULT_BATCH_SIZE
    )
    filters     = st.sidebar.slider("Conv filters", 16, 128, DEFAULT_FILTERS, 16)
    kernel_size = st.sidebar.slider("Kernel size", 3, 9, DEFAULT_KERNEL_SIZE, 2)
    dropout     = st.sidebar.slider("Dropout", 0.1, 0.6, DEFAULT_DROPOUT, 0.05)
    gru_units   = DEFAULT_GRU_UNITS
    if "BiGRU" in architecture:
        gru_units = st.sidebar.slider("GRU units", 16, 128, DEFAULT_GRU_UNITS, 16)

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "**Data:** NCBI Entrez · RNAcentral\n\n"
        "**Models:** CNN · CNN+BiGRU\n\n"
        "**Paper:** IEEE Xplore 10563808\n\n"
        "**Cache TTL:** 60 min"
    )

    return {
        "n_per_class":  n_per_class,
        "seq_length":   seq_length,
        "encoding":     encoding,
        "k":            k,
        "architecture": architecture,
        "epochs":       epochs,
        "batch_size":   batch_size,
        "filters":      filters,
        "kernel_size":  kernel_size,
        "dropout":      dropout,
        "gru_units":    gru_units,
    }
