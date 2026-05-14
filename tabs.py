"""
tabs.py — One function per dashboard tab.

tab_data()      → 📡 Data Pipeline   (fetch, explore sequences)
tab_encoding()  → 🔬 Encoding Explorer (visualise all 4 encodings side by side)
tab_train()     → 🧠 Train Model     (build, train, show learning curves)
tab_results()   → 📊 Evaluation      (confusion matrix, ROC, metrics)
tab_predictor() → 🔍 Predictor       (paste sequence → classify)
"""

from itertools import product

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from api import fetch_all
from components import kpi_row
from encoder import (
    encode_dataset,
    encode_sequence,
    integer_encode,
    kmer_encode,
    one_hot_encode,
    voss_encode,
)
from model import (
    build_cnn,
    build_cnn_bigru,
    build_mlp,
    evaluate_model,
    train_model,
)
from config import CLASS_COLORS, CLASS_LABELS


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _gc(seq: str) -> float:
    return sum(c in "GC" for c in seq) / len(seq) if seq else 0.0


def _section(title: str) -> None:
    st.markdown(f'<div class="section-header">{title}</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — DATA PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def tab_data(cfg: dict) -> None:
    _section("Data Pipeline")

    col_btn, col_desc = st.columns([1, 4])
    with col_btn:
        fetch_btn = st.button("🔄 Fetch Sequences", use_container_width=True)
    with col_desc:
        st.caption(
            f"Fetching {cfg['n_per_class']} sequences × 3 classes "
            f"at {cfg['seq_length']} bp — NCBI (Coding, Pseudogene) + RNAcentral (lncRNA)"
        )

    if fetch_btn:
        with st.spinner("Loading and preprocessing sequences from local FASTA files…"):
            seqs, prep_reports = fetch_all(cfg["n_per_class"], cfg["seq_length"])
        st.session_state.sequences    = seqs
        st.session_state.prep_reports = prep_reports
        for key in ("model", "history", "X_test", "y_test", "eval_results", "encoder_cfg"):
            st.session_state.pop(key, None)

    if "sequences" not in st.session_state:
        st.info("Click **Fetch Sequences** to load and preprocess sequences from local FASTA files.")
        return

    seqs         = st.session_state.sequences
    prep_reports = st.session_state.get("prep_reports", {})
    total = sum(len(v) for v in seqs.values())

    kpi_row([
        ("Coding",     str(len(seqs.get("Coding",     [])))),
        ("lncRNA",     str(len(seqs.get("lncRNA",     [])))),
        ("Pseudogene", str(len(seqs.get("Pseudogene", [])))),
        ("Total",      str(total)),
    ])

    # Preprocessing report
    if prep_reports:
        with st.expander("🧹 Preprocessing report — what was cleaned"):
            rows = []
            for cls, r in prep_reports.items():
                rows.append({
                    "Class":             cls,
                    "Raw sequences":     r["raw"],
                    "Dropped (quality)": r["dropped_low_quality"],
                    "Dropped (dupes)":   r["dropped_duplicates"],
                    "Final":             r["final"],
                    "Retention %":       f"{r['final']/r['raw']*100:.1f}%" if r["raw"] else "—",
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)

    # Class bar chart
    with c1:
        st.markdown("**Class Distribution**")
        df_cnt = pd.DataFrame([{"Class": k, "Count": len(v)} for k, v in seqs.items()])
        fig = px.bar(
            df_cnt, x="Class", y="Count", color="Class",
            color_discrete_map=CLASS_COLORS, template="plotly_white",
        )
        fig.update_layout(showlegend=False, height=300, margin=dict(t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)

    # GC content violin
    with c2:
        st.markdown("**GC Content by Class**")
        gc_rows = [
            {"Class": cls, "GC %": _gc(s) * 100}
            for cls, slist in seqs.items()
            for s in slist
        ]
        fig = px.violin(
            pd.DataFrame(gc_rows), x="Class", y="GC %", color="Class",
            color_discrete_map=CLASS_COLORS, box=True, template="plotly_white",
        )
        fig.update_layout(showlegend=False, height=300, margin=dict(t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)

    # Length distribution
    st.markdown("**Sequence Length Distribution**")
    len_rows = [
        {"Class": cls, "Length (bp)": len(s)}
        for cls, slist in seqs.items()
        for s in slist
    ]
    fig = px.histogram(
        pd.DataFrame(len_rows), x="Length (bp)", color="Class",
        barmode="overlay", opacity=0.7,
        color_discrete_map=CLASS_COLORS, template="plotly_white",
    )
    fig.update_layout(height=280, margin=dict(t=20, b=20))
    st.plotly_chart(fig, use_container_width=True)

    with st.expander(f"📋 Sample sequences ({min(5, min(len(v) for v in seqs.values()))} per class)"):
        rows = [
            {
                "Class":              cls,
                "Length (bp)":        len(s),
                "GC %":               f"{_gc(s)*100:.1f}",
                "Sequence (first 60 bp)": s[:60] + "…",
            }
            for cls, slist in seqs.items()
            for s in slist[:5]
        ]
        st.dataframe(pd.DataFrame(rows), use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — ENCODING EXPLORER
# ─────────────────────────────────────────────────────────────────────────────

def tab_encoding(cfg: dict) -> None:
    _section("Encoding Explorer")

    if "sequences" not in st.session_state:
        st.info("Fetch sequences first in the **Data Pipeline** tab.")
        return

    seqs = st.session_state.sequences

    c1, c2 = st.columns([3, 1])
    with c1:
        options = [
            f"{cls} #{i + 1}"
            for cls, slist in seqs.items()
            for i in range(min(5, len(slist)))
        ]
        selected = st.selectbox("Select a sample sequence", options)
    with c2:
        k_vis = st.slider("K-mer size (visualisation)", 2, 5, 3, key="enc_k")

    cls_name = selected.split(" #")[0]
    idx      = int(selected.split("#")[1]) - 1
    seq      = seqs[cls_name][idx]

    st.markdown(
        f"**Class:** `{cls_name}` · "
        f"**Length:** `{len(seq)} bp` · "
        f"**GC:** `{_gc(seq)*100:.1f}%` · "
        f"**Preview:** `{seq[:80]}{'…' if len(seq) > 80 else ''}`"
    )
    st.markdown("---")

    display_seq = seq[:100]  # keep heatmaps manageable

    # ── One-Hot heatmap ───────────────────────────────────────────────────────
    st.markdown("#### One-Hot Encoding  *(L × 4 binary matrix)*")
    oh = one_hot_encode(display_seq)
    fig = px.imshow(
        oh.T,
        labels={"x": "Position (bp)", "y": "Nucleotide", "color": "Value"},
        y=["A", "C", "G", "T"],
        color_continuous_scale=[[0, "#f0faf5"], [1, "#1A4D3A"]],
        aspect="auto", template="plotly_white",
    )
    fig.update_layout(height=160, margin=dict(t=10, b=10), coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)

    # ── Voss signal plot ──────────────────────────────────────────────────────
    st.markdown("#### Voss Representation  *(4 independent binary signals)*")
    voss  = voss_encode(display_seq)
    vsub  = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.03,
                          row_titles=["A", "C", "G", "T"])
    vcols = ["#1A4D3A", "#0077b6", "#e63946", "#f4a261"]
    for i, col in enumerate(vcols):
        vsub.add_trace(
            go.Scatter(y=voss[:, i].tolist(), mode="lines",
                       line=dict(color=col, width=1), name=["A","C","G","T"][i]),
            row=i + 1, col=1,
        )
    vsub.update_layout(height=260, showlegend=False,
                       margin=dict(t=10, b=10), template="plotly_white")
    st.plotly_chart(vsub, use_container_width=True)

    c1, c2 = st.columns(2)

    # ── Integer encoding bar chart ────────────────────────────────────────────
    with c1:
        st.markdown("#### Integer Encoding  *(A=1 C=2 G=3 T=4)*")
        int_enc = integer_encode(display_seq[:80]).flatten()
        fig = px.bar(
            x=list(range(len(int_enc))), y=int_enc.tolist(),
            labels={"x": "Position", "y": "Integer value"},
            color=int_enc.tolist(),
            color_continuous_scale=[[0,"#eee"],[0.33,"#1A4D3A"],
                                     [0.66,"#0077b6"],[1,"#e63946"]],
            template="plotly_white",
        )
        fig.update_layout(height=240, margin=dict(t=10, b=10),
                          coloraxis_showscale=False, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    # ── K-mer frequency bar chart ─────────────────────────────────────────────
    with c2:
        st.markdown(f"#### K-mer Frequencies  *(top-20, k={k_vis})*")
        km      = kmer_encode(seq, k_vis)
        kmers   = ["".join(p) for p in product("ACGT", repeat=k_vis)]
        top_idx = np.argsort(km)[::-1][:20]
        fig = px.bar(
            x=[kmers[i] for i in top_idx],
            y=[float(km[i]) for i in top_idx],
            labels={"x": f"{k_vis}-mer", "y": "Relative frequency"},
            color=[float(km[i]) for i in top_idx],
            color_continuous_scale=[[0, "#c8e6c9"], [1, "#1A4D3A"]],
            template="plotly_white",
        )
        fig.update_layout(height=240, margin=dict(t=10, b=10),
                          coloraxis_showscale=False, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("📋 Encoding comparison table (first 10 positions)"):
        oh10  = one_hot_encode(seq[:10])
        int10 = integer_encode(seq[:10]).flatten()
        rows  = [
            {
                "Pos":              i,
                "Base":             seq[i],
                "Integer":          int(int10[i]),
                "One-Hot (A,C,G,T)": str(oh10[i].astype(int).tolist()),
            }
            for i in range(10)
        ]
        st.dataframe(pd.DataFrame(rows), use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — TRAIN MODEL
# ─────────────────────────────────────────────────────────────────────────────

def tab_train(cfg: dict) -> None:
    _section("Train Model")

    if "sequences" not in st.session_state:
        st.info("Fetch sequences first in the **Data Pipeline** tab.")
        return

    seqs  = st.session_state.sequences
    total = sum(len(v) for v in seqs.values())

    # Config summary
    kpi_row([
        ("Encoding",     cfg["encoding"]),
        ("Architecture", cfg["architecture"]),
        ("Epochs",       str(cfg["epochs"])),
        ("Sequences",    str(total)),
    ])
    st.markdown("<br>", unsafe_allow_html=True)

    # K-mer note
    if cfg["encoding"] == "K-mer":
        st.markdown(
            '<div class="insight-box">'
            "K-mer encoding produces a flat frequency vector (no positional info). "
            "An MLP is used instead of CNN/BiGRU for this encoding."
            "</div>",
            unsafe_allow_html=True,
        )

    if st.button("🚀 Build & Train Model", use_container_width=False):
        with st.spinner("Encoding sequences…"):
            X, y = encode_dataset(seqs, cfg["encoding"], cfg["k"])

        input_shape = X.shape[1:]
        st.success(f"Encoded dataset: {X.shape} — input shape per sample: {input_shape}")

        # Build model
        if cfg["encoding"] == "K-mer":
            model = build_mlp(input_shape, dropout=cfg["dropout"])
        elif cfg["architecture"] == "CNN":
            model = build_cnn(
                input_shape, filters=cfg["filters"],
                kernel_size=cfg["kernel_size"], dropout=cfg["dropout"],
            )
        else:
            model = build_cnn_bigru(
                input_shape, filters=cfg["filters"],
                gru_units=cfg["gru_units"], dropout=cfg["dropout"],
            )

        with st.expander("🏗 Model architecture"):
            st.code(str(model), language=None)

        st.markdown("**Training progress**")
        progress_bar = st.progress(0.0)
        status_text  = st.empty()

        model, history, X_test, y_test = train_model(
            model, X, y,
            cfg["epochs"], cfg["batch_size"],
            progress_bar, status_text,
        )

        eval_results = evaluate_model(model, X_test, y_test)

        st.session_state.model       = model
        st.session_state.history     = history
        st.session_state.X_test      = X_test
        st.session_state.y_test      = y_test
        st.session_state.encoder_cfg = cfg
        st.session_state.eval_results = eval_results

        st.success(
            f"Done! Test accuracy: **{eval_results['report']['accuracy']:.1%}** — "
            "see the **Evaluation** tab for full results."
        )

    # ── Learning curves (shown if model already trained) ──────────────────────
    if "history" in st.session_state:
        h = st.session_state.history
        ep = list(range(1, len(h["loss"]) + 1))
        st.markdown("---")
        st.markdown("**Learning curves**")
        c1, c2 = st.columns(2)

        with c1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=ep, y=h["loss"],     name="Train",
                                     line=dict(color="#1A4D3A")))
            fig.add_trace(go.Scatter(x=ep, y=h["val_loss"], name="Validation",
                                     line=dict(color="#e63946", dash="dash")))
            fig.update_layout(title="Loss", xaxis_title="Epoch",
                              template="plotly_white", height=300,
                              margin=dict(t=40, b=20))
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=ep, y=h["accuracy"],     name="Train",
                                     line=dict(color="#1A4D3A")))
            fig.add_trace(go.Scatter(x=ep, y=h["val_accuracy"], name="Validation",
                                     line=dict(color="#e63946", dash="dash")))
            fig.update_layout(title="Accuracy", xaxis_title="Epoch",
                              template="plotly_white", height=300,
                              margin=dict(t=40, b=20))
            st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

def tab_results(cfg: dict) -> None:
    _section("Evaluation Results")

    if "eval_results" not in st.session_state:
        st.info("Train a model first in the **Train Model** tab.")
        return

    res    = st.session_state.eval_results
    report = res["report"]
    cm     = res["confusion_matrix"]
    roc    = res["roc"]
    proba  = res["y_pred_proba"]
    y_test = res["y_test"]

    kpi_row([
        ("Overall Accuracy", f"{report['accuracy']:.1%}"),
        ("Coding F1",        f"{report['Coding']['f1-score']:.3f}"),
        ("lncRNA F1",        f"{report['lncRNA']['f1-score']:.3f}"),
        ("Pseudogene F1",    f"{report['Pseudogene']['f1-score']:.3f}"),
    ])

    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)

    # Confusion matrix
    with c1:
        st.markdown("**Confusion Matrix**")
        fig = px.imshow(
            cm,
            labels=dict(x="Predicted", y="Actual", color="Count"),
            x=CLASS_LABELS, y=CLASS_LABELS,
            text_auto=True,
            color_continuous_scale=[[0, "#f0faf5"], [1, "#1A4D3A"]],
            template="plotly_white",
        )
        fig.update_layout(height=360, margin=dict(t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)

    # Per-class precision / recall / F1
    with c2:
        st.markdown("**Per-class Metrics**")
        rows = [
            {"Class": cls, "Metric": m, "Score": report[cls][m]}
            for cls in CLASS_LABELS
            for m in ("precision", "recall", "f1-score")
        ]
        fig = px.bar(
            pd.DataFrame(rows), x="Class", y="Score", color="Metric",
            barmode="group", template="plotly_white",
            color_discrete_sequence=["#1A4D3A", "#0077b6", "#e63946"],
        )
        fig.update_layout(height=360, margin=dict(t=20, b=20), yaxis_range=[0, 1])
        st.plotly_chart(fig, use_container_width=True)

    # ROC curves
    st.markdown("**ROC Curves (one-vs-rest)**")
    fig = go.Figure()
    for cls, (fpr, tpr, auc_score) in roc.items():
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr,
            name=f"{cls}  (AUC = {auc_score:.3f})",
            line=dict(color=CLASS_COLORS[cls], width=2),
        ))
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        line=dict(dash="dash", color="gray", width=1),
        showlegend=False,
    ))
    fig.update_layout(
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        template="plotly_white", height=360,
        margin=dict(t=20, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("📋 Full classification report"):
        st.dataframe(
            pd.DataFrame(report).T.round(4),
            use_container_width=True,
        )

    with st.expander(f"📋 Test-set predictions ({len(y_test)} samples)"):
        pred_rows = [
            {
                "Sample":       i,
                "True":         CLASS_LABELS[int(y_test[i])],
                "Predicted":    CLASS_LABELS[int(res["y_pred"][i])],
                "Conf Coding":  f"{proba[i][0]:.3f}",
                "Conf lncRNA":  f"{proba[i][1]:.3f}",
                "Conf Pseudo":  f"{proba[i][2]:.3f}",
                "Correct":      y_test[i] == res["y_pred"][i],
            }
            for i in range(len(y_test))
        ]
        st.dataframe(pd.DataFrame(pred_rows), use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 5 — SEQUENCE PREDICTOR
# ─────────────────────────────────────────────────────────────────────────────

def tab_predictor(cfg: dict) -> None:
    _section("Sequence Predictor")

    if "model" not in st.session_state:
        st.info("Train a model first in the **Train Model** tab.")
        return

    model   = st.session_state.model
    enc_cfg = st.session_state.encoder_cfg

    st.markdown(
        f'<div class="insight-box">'
        f"Model: <b>{enc_cfg['architecture']}</b> · "
        f"Encoding: <b>{enc_cfg['encoding']}</b> · "
        f"Trained on sequences of <b>{enc_cfg['seq_length']} bp</b>"
        f"</div>",
        unsafe_allow_html=True,
    )

    seq_input = st.text_area(
        "Paste a raw genomic sequence (ACGT)",
        height=130,
        placeholder="ATCGATCGATCGATCG…",
        help="Non-ACGT characters are replaced with N. "
             "Sequence is truncated or padded to the trained length.",
    )

    if st.button("🔍 Classify Sequence") and seq_input.strip():
        raw = seq_input.strip().upper()
        seq = "".join(c if c in "ACGT" else "N" for c in raw)

        target = enc_cfg["seq_length"]
        seq    = seq[:target] if len(seq) >= target else seq + "N" * (target - len(seq))

        encoded = encode_sequence(seq, enc_cfg["encoding"], enc_cfg["k"])
        X       = encoded.reshape(1, *encoded.shape)
        proba   = model.predict(X, verbose=0)[0]

        pred_cls    = CLASS_LABELS[int(np.argmax(proba))]
        confidence  = float(np.max(proba))
        color       = CLASS_COLORS[pred_cls]

        # Result box
        st.markdown(
            f'<div style="padding:20px;border-radius:8px;border:2px solid {color};'
            f'text-align:center;margin:16px 0">'
            f'<div style="font-size:2.2rem;font-weight:700;color:{color}">{pred_cls}</div>'
            f'<div style="font-size:0.95rem;color:#4a5568;margin-top:6px">'
            f'Confidence: <b>{confidence:.1%}</b></div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # Probability bars
        fig = px.bar(
            x=CLASS_LABELS,
            y=[float(p) for p in proba],
            color=CLASS_LABELS,
            color_discrete_map=CLASS_COLORS,
            labels={"x": "Class", "y": "Probability"},
            template="plotly_white",
        )
        fig.update_layout(
            showlegend=False, height=280,
            margin=dict(t=20, b=20), yaxis_range=[0, 1],
        )
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("📋 Input sequence details"):
            st.markdown(f"**Used length:** {len(seq)} bp")
            st.markdown(f"**GC content:** {_gc(seq)*100:.1f}%")
            st.markdown(f"**N content:** {seq.count('N')/len(seq)*100:.1f}%")
            st.code(seq[:200] + ("…" if len(seq) > 200 else ""), language=None)
