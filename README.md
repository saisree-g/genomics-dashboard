# Genomics Sequence Classifier

An interactive deep learning dashboard for classifying human genomic sequences into three classes — **Coding**, **lncRNA**, and **Pseudogene** — using CNN and CNN+BiGRU models.

Built as an extension of published IEEE research on pseudogene classification using deep learning.

---

## Live Demo

> Deployed on Streamlit Community Cloud

---

## What It Does

| Tab | Description |
|---|---|
| 📡 Data Pipeline | Load sequences, inspect class balance, GC content, and length distribution |
| 🔬 Encoding Explorer | Visualise One-Hot, Voss, Integer, and K-mer encodings side by side |
| 🧠 Train Model | Configure and train CNN or CNN+BiGRU with live progress |
| 📊 Evaluation | Confusion matrix, per-class F1, ROC curves, full prediction table |
| 🔍 Predictor | Paste any raw sequence and get a classification with confidence scores |

---

## Models

- **CNN** — 1D Convolutional Neural Network with batch normalisation and global average pooling
- **CNN + BiGRU** — CNN feature extractor followed by Bidirectional GRU layers (replicates the IEEE paper architecture)

## Encoding Schemes

| Encoding | Shape | Description |
|---|---|---|
| One-Hot | (L, 4) | Binary indicator per nucleotide |
| Voss | (L, 4) | Four independent binary signals |
| Integer | (L, 1) | A=1, C=2, G=3, T=4 |
| K-mer | (4^k,) | Normalised k-mer frequency vector |

---

## Data Sources

| Class | Source |
|---|---|
| Coding regions | [NCBI Nucleotide](https://www.ncbi.nlm.nih.gov/nucleotide/) |
| lncRNA | [RNAcentral](https://rnacentral.org/) |
| Pseudogenes | [pseudogene.org](http://www.pseudogene.org/Human/) |

A pre-cleaned sample of 500 sequences per class is included in `data/sample.csv` for deployment. Full FASTA files are excluded from the repo due to size (327MB total).

---

## Run Locally

**Requirements:** Python 3.12, full FASTA files in `data/`

```bash
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python -m streamlit run app.py
```

**To regenerate the sample dataset from full FASTA files:**
```bash
python create_sample.py
```

---

## Project Structure

```
genomics-dashboard/
├── app.py           # Entry point — page config and tab wiring
├── config.py        # Constants, defaults, class labels
├── api.py           # Data loading (FASTA locally, sample.csv on cloud)
├── preprocess.py    # Clean → validate → deduplicate → truncate pipeline
├── encoder.py       # One-Hot, Voss, Integer, K-mer encoding
├── model.py         # CNN, CNN+BiGRU, MLP builders + training loop
├── components.py    # CSS, KPI cards, sidebar
├── tabs.py          # One function per dashboard tab
├── create_sample.py # Script to extract sample.csv from full FASTA files
├── requirements.txt
├── runtime.txt      # Python 3.12 for Streamlit Cloud
└── data/
    └── sample.csv   # 1500 pre-cleaned sequences (500 per class)
```

---

## Reference

Based on:
> Gali Sai Sree. *Deep Learning for the Classification of Pseudogenes in the Genome.* IEEE Xplore, 2024. [DOI: 10.1109/10563808](https://ieeexplore.ieee.org/document/10563808/)
