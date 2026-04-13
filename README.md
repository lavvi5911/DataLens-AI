# 🔬 DataLens AI — Dataset Debugger

A production-grade, self-diagnosing data intelligence system built with Streamlit.

---

## 🚀 Deploy on Streamlit Cloud (Free)

### Step 1 — Push to GitHub
```bash
git init
git add .
git commit -m "Initial commit — DataLens AI"
git remote add origin https://github.com/<your-username>/datalens-ai.git
git push -u origin main
```

### Step 2 — Connect to Streamlit Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click **New app**
3. Connect your GitHub repo
4. Set **Main file path** → `app.py`
5. Click **Deploy**

Your app will be live in ~60 seconds.

---

## 🖥️ Run Locally

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate        # macOS/Linux
# venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt

# Launch
streamlit run app.py
```

---

## 📁 Project Structure

```
datalens-ai/
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
├── assets/
│   └── styles.css          # Premium dark UI styles
└── utils/
    ├── __init__.py
    ├── data_loader.py       # CSV loading & validation
    ├── analyzer.py          # Statistical profiling
    ├── detector.py          # Issue detection engine
    ├── recommender.py       # Preprocessing & model recommendations
    └── explainer.py         # Plain-English insight generation
```

---

## ✅ Features

| Feature | Details |
|---|---|
| **Missing values** | Per-column analysis with severity thresholds |
| **Outliers** | IQR-based detection with visual boxplots |
| **Skewness** | Coefficient + transform recommendations |
| **Multicollinearity** | Pairwise correlation > 0.85 |
| **Class imbalance** | Minority class detection + SMOTE advice |
| **Data leakage** | High correlation with assumed target |
| **Duplicate rows** | Count + removal guidance |
| **High cardinality** | Encoding strategy per column |
| **Health Score** | 0–100 dataset quality score |
| **Download report** | Full .txt diagnostic report |

---

## 📦 Dependencies

```
streamlit==1.35.0
pandas==2.2.2
numpy==1.26.4
scikit-learn==1.4.2
plotly==5.22.0
scipy==1.13.0
```

No GPU required. Runs on free-tier cloud instances.
