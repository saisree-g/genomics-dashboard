"""
app.py — Entry point for the Genomics Sequence Classifier dashboard.

Run with:
    streamlit run app.py

Module responsibilities:
    app.py        → page config, CSS injection, main() orchestration
    config.py     → all constants (API URLs, class labels, defaults)
    api.py        → NCBI Entrez + RNAcentral fetching with Streamlit cache
    encoder.py    → One-Hot / Voss / Integer / K-mer encoding
    model.py      → CNN and CNN+BiGRU builders, training loop, evaluation
    components.py → kpi_row(), render_sidebar(), CSS string
    tabs.py       → one function per dashboard tab
"""

import warnings
warnings.filterwarnings("ignore")

import streamlit as st

from components import CSS, render_sidebar
from tabs import tab_data, tab_encoding, tab_train, tab_results, tab_predictor

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG  (must be the first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Genomics Sequence Classifier",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(CSS, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    cfg = render_sidebar()

    st.markdown(
        '<div class="dash-title">Genomics Sequence Classifier</div>'
        '<div class="dash-subtitle">'
        "Live classification of Coding &nbsp;·&nbsp; lncRNA &nbsp;·&nbsp; Pseudogene regions"
        " &nbsp;·&nbsp; NCBI + RNAcentral APIs"
        " &nbsp;·&nbsp; CNN &amp; CNN+BiGRU"
        " &nbsp;·&nbsp; One-Hot · Voss · Integer · K-mer encodings"
        "</div>"
        '<hr style="border:none;border-top:1px solid #dee2e6;margin:16px 0 8px">',
        unsafe_allow_html=True,
    )

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📡  Data Pipeline",
        "🔬  Encoding Explorer",
        "🧠  Train Model",
        "📊  Evaluation",
        "🔍  Predictor",
    ])

    with tab1: tab_data(cfg)
    with tab2: tab_encoding(cfg)
    with tab3: tab_train(cfg)
    with tab4: tab_results(cfg)
    with tab5: tab_predictor(cfg)


if __name__ == "__main__":
    main()
