# RansomwareWatch

AI-based real-time ransomware detection system with a Streamlit UI.

This project implements an end-to-end pipeline:
1. Load and preprocess labeled network traffic data (benign vs ransomware).
   In the original project setup/presentation, the CSVs were derived from the **CICIDS 2017** dataset.
2. Train a **WCGAN-GP** module (PyTorch) to learn the ransomware feature distribution.
3. Train an **XGBoost** classifier and generate evaluation artifacts (confusion matrix, ROC curve, PDF training report).
4. Run detection on uploaded CSVs, with optional **SHAP** explanations.

Presentation: [`docs/RansomwareWatch_Presentation.pdf`](docs/RansomwareWatch_Presentation.pdf)

---

## Repository layout

- `streamlit_app.py` — Streamlit app (training + detection)
- `ransomwarewatch/`
  - `data_preprocessing.py` — data loading/cleaning/scaling + train/test split
  - `wcgan.py` — WCGAN-GP training loop (PyTorch)
  - `xgboost_model.py` — XGBoost model + metrics + PDF report generation
- `data/README.md` — where to place datasets locally (datasets are gitignored)
- `docs/` — project documentation/presentation

---

## Setup

### Requirements
- Python 3.9–3.11 recommended
- Dependencies in `requirements.txt`

Create environment + install:

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
# source .venv/bin/activate

pip install -r requirements.txt
```

---

## Dataset (not committed to Git)

This repository intentionally does **not** track datasets.

Place these files locally:

- `data/benign.csv`
- `data/ransom.csv`

See: `data/README.md`

---

## Run the app

```bash
streamlit run streamlit_app.py
```

---

## Notes / limitations (current)

- The WCGAN-GP module is trained during the pipeline, but synthetic samples are **not yet fed into XGBoost training** (augmentation can be added as a future improvement).
- Training generates `training_report_*.pdf` locally (gitignored by design).
- SHAP rendering in Streamlit can vary by environment/backend and may need small adjustments.

---

## License

Add a license before using this project commercially.
