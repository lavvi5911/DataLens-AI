# 🔬 DataLens AI — AI Dataset Debugger

> **Self-Diagnosing Data Intelligence System**
> Upload any CSV dataset and get instant, automated diagnostics, plain-English AI insights, and actionable preprocessing recommendations — no code required.

Live Demo: [datalens-ai.streamlit.app](https://datalens-ai-j7ewbhqp8ks4m8eqzrvzo4.streamlit.app)

---

## 📋 Table of Contents

1. [What It Does](#what-it-does)
2. [Features](#features)
3. [Project Structure](#project-structure)
4. [How Each File Works](#how-each-file-works)
5. [Deploy on Streamlit Cloud](#deploy-on-streamlit-cloud)
6. [Run Locally](#run-locally)
7. [Dependencies](#dependencies)
8. [Known Issues & Fixes Applied](#known-issues--fixes-applied)
9. [Testing the App](#testing-the-app)
10. [Changelog](#changelog)

---

## What It Does

DataLens AI accepts any tabular CSV dataset and automatically runs a full diagnostic pipeline:

1. **Loads** the file (handles encoding, large files, type coercion)
2. **Profiles** the data (shape, dtypes, statistics, correlation)
3. **Detects** 9 categories of data quality and ML-readiness issues
4. **Scores** the dataset from 0–100 (Dataset Health Score)
5. **Explains** every finding in plain English — no jargon
6. **Recommends** preprocessing steps, encoding strategies, and model choices with copy-paste code snippets
7. **Exports** a full diagnostic report as a downloadable `.txt` file

---

## Features

### 🧠 Issue Detection Engine
Automatically detects and severity-rates these issues:

| Issue | Detection Method | Severity Logic |
|---|---|---|
| Missing values | `df.isnull().mean()` per column | >30% = Critical, >10% = High, >1% = Medium |
| Duplicate rows | `df.duplicated().sum()` | >5% of rows = High |
| Outliers | IQR method per numeric column | >15% outlier points = High |
| Skewness | `df.skew()` per numeric column | \|skew\| > 2 = High, > 1 = Medium |
| Low variance / constants | `std < 1e-6` or `nunique == 1` | Constant = High |
| Multicollinearity | Pairwise correlation > 0.85 | > 0.95 = High |
| Class imbalance | Minority class % of last column | < 5% = Critical, < 10% = High |
| Data leakage | Correlation with target > 0.95 | Always = Critical |
| High cardinality | Unique ratio > 80% and > 50 unique values | Medium |

### 📊 Dataset Health Score (0–100)
Each detected issue deducts points based on severity:
- Critical issues: −15 to −20 points each
- High issues: −5 to −8 points each
- Medium issues: −2 to −4 points each
- Low issues: −1 point each

Score interpretation:
- **75–100** → Excellent — ready for modeling
- **50–74** → Needs Work — address high-severity issues first
- **0–49** → Critical — significant cleaning required

### 💡 AI Insight Engine
Rule-based natural language generation (no external API) produces business-friendly explanations like:

> *"Feature 'income' has a skewness of 3.2, indicating a severe right-skewed distribution. Most values are clustered near zero with a long tail of high earners. Log or power transforms typically resolve this for linear models."*

### 💻 Recommendation Engine
Generates specific, copy-paste ready code recommendations for:
- **Preprocessing**: imputation, duplicate removal, outlier capping, scaling
- **Feature engineering**: PCA, VIF selection, variance thresholding
- **Encoding**: OHE for low cardinality, target/frequency encoding for high cardinality
- **Modeling**: task detection (classification vs regression), model suggestions, evaluation metrics, SMOTE for imbalance

### 📈 Interactive Visualizations (Plotly)
- Correlation heatmap
- Distribution histogram + boxplot side by side
- Missing values bar chart (color-coded by severity)
- Outlier boxplots (up to 8 features)
- Feature skewness bar chart with threshold lines
- Categorical value count bar chart

### 🗂️ Large Dataset Support
- Files up to **500 MB** accepted
- Datasets up to **50M cells** loaded in full
- Larger datasets automatically **chunked and sampled** to 200K representative rows
- **dtype downcasting** reduces memory by 40–60%: `float64→float32`, `int64→smallest int`, `object→category`

---

## Project Structure

```
datalens-ai/
│
├── app.py                      # Main Streamlit application — all UI and navigation
│
├── requirements.txt            # Python package dependencies (loosely pinned for compatibility)
│
├── .streamlit/
│   └── config.toml             # Streamlit server config — upload limit, theme colours
│
├── assets/
│   └── styles.css              # Full custom dark UI — ~550 lines of premium CSS
│
└── utils/
    ├── __init__.py
    ├── data_loader.py          # CSV loading, encoding detection, sampling, dtype optimisation
    ├── analyzer.py             # Statistical profiling (skew, kurtosis, correlation, cardinality)
    ├── detector.py             # 9-category issue detection engine + health score
    ├── recommender.py          # Preprocessing, feature engineering, encoding, model suggestions
    └── explainer.py            # Plain-English insight generation from detected issues
```

---

## How Each File Works

### `app.py`
The main Streamlit app. Handles:
- Page config and CSS injection
- A custom JS floating sidebar toggle button (fixes Streamlit 1.56 visibility bug)
- Session state management for all computed results
- 5-section sidebar navigation: Upload → Overview → Issues → Insights → Recommendations
- Plotly chart rendering
- Downloadable report generation

### `utils/data_loader.py`
- Tries UTF-8, Latin-1, CP1252 encodings automatically
- Estimates dataset size before loading to decide full-load vs chunked-sample
- `_clean_column_names()`: strips whitespace, replaces special chars, deduplicates names
- `_coerce_numeric()`: converts object columns that are ≥90% numeric to float
- `_downcast_dtypes()`: shrinks memory footprint significantly

### `utils/analyzer.py`
- `DataAnalyzer.run()` returns a dict with: shape, dtypes, missing stats, descriptive stats, skewness, kurtosis, correlation matrix, cardinality ratios, duplicate count, memory usage
- Used for the Data Overview section charts and statistics tab

### `utils/detector.py`
- `IssueDetector.detect_all()` runs all 9 checkers and returns a list of issue dicts
- Each issue has: `severity`, `title`, `description`, `fix`, `feature`, `category`, `value`
- `_detected` boolean flag prevents `detect_all()` from being called multiple times (which would reset `_health_deductions` to 0 — the original health score = 0 bug)
- `health_score()` checks `_detected` flag, not `if not self._issues` (empty list is falsy — would wrongly re-run on clean datasets)

### `utils/recommender.py`
- Reads `detector._issues` (already populated — no re-run)
- `_detect_task()`: heuristic — last column with ≤20 unique values → classification, else regression
- Generates 4 recommendation sections with title, detail, and code snippet per item

### `utils/explainer.py`
- Template-driven NLP — no external model or API needed
- Each `_*_insights()` method reads specific issue categories and generates `{type, title, body, action}` dicts
- Types map to colours/icons in the UI: `info`=indigo, `warning`=amber, `success`=green, `error`=red

### `assets/styles.css`
- Full dark theme with CSS variables
- Uses `display: none` (not `visibility: hidden`) on `header`/`footer`/`#MainMenu` to avoid cascading invisibility to child elements like the sidebar toggle
- Glassmorphism cards, gradient hero, animated health bar, JetBrains Mono for code blocks

### `.streamlit/config.toml`
```toml
[server]
maxUploadSize = 500        # allows up to 500MB CSV uploads

[theme]
base = "dark"
primaryColor = "#6366f1"
```

---

## Deploy on Streamlit Cloud

### Step 1 — Push to GitHub
```bash
git init
git add .
git commit -m "DataLens AI — initial commit"
git remote add origin https://github.com/<your-username>/datalens-ai.git
git push -u origin main
```

### Step 2 — Connect to Streamlit Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with GitHub
3. Click **New app**
4. Select your repository and branch (`main`)
5. Set **Main file path** → `app.py`
6. Click **Deploy**

The app will be live in ~60 seconds. The upload box should show **"500MB per file · CSV"** confirming the config was picked up.

### Step 3 — Verify deployment
After the app loads:
- Upload the `titanic.csv` test file (see Testing section)
- Check the sidebar Health Score widget appears
- Navigate all 5 sections
- Confirm the sidebar toggle button (☰) is visible top-left even when sidebar is closed

---

## Run Locally

```bash
# 1. Clone
git clone https://github.com/<your-username>/datalens-ai.git
cd datalens-ai

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Dependencies

```
streamlit>=1.35.0       # Web framework and UI components
pandas>=2.2.0           # DataFrame operations, CSV loading
numpy>=1.26.0           # Numeric computation
scikit-learn>=1.4.0     # Statistical utilities
plotly>=5.22.0          # Interactive charts
scipy>=1.13.0           # Statistical tests (skewness, kurtosis)
```

Loosely pinned with `>=` to allow pip/uv to resolve pre-built wheels for the
current Python version (3.14 on Streamlit Cloud) without source compilation.

---

## Known Issues & Fixes Applied

### 1. `scikit-learn==1.4.2` build failure on Python 3.14
**Symptom:** Streamlit Cloud logs show `numpy==2.0.0rc1` not found error.
**Cause:** Exact-pinned versions had no pre-built wheels for Python 3.14, forcing source builds that needed a non-existent numpy release candidate.
**Fix:** Switched all pins from `==` to `>=` so pip picks the latest compatible wheel.

### 2. Dataset Health Score showing 0
**Symptom:** Health score widget always showed 0 regardless of dataset issues.
**Cause (Bug A):** `detect_all()` was called 3 times — once inside `RecommendationEngine.__init__()`, once inside `InsightExplainer.__init__()`, and once explicitly in `app.py`. Each call reset `_health_deductions = 0.0`.
**Cause (Bug B):** `health_score()` used `if not self._issues` as the "already ran" guard. An empty list `[]` is falsy in Python, so a perfectly clean dataset triggered a re-run that wiped deductions.
**Fix:** Added `_detected: bool = False` flag to `IssueDetector`. Set to `True` after first `detect_all()` run. `health_score()` checks this flag. In `app.py`, `detect_all()` and `health_score()` are called before `RecommendationEngine` and `InsightExplainer` are constructed.

### 3. Empty label warnings flooding logs
**Symptom:** Streamlit logs full of `` `label` got an empty value `` stacktraces.
**Cause:** `st.radio("", ...)` and `st.file_uploader("", ...)` with empty string labels — Streamlit 1.56 upgraded this from a silent warning to a logged stacktrace.
**Fix:** Changed both to non-empty labels with `label_visibility="hidden"`.

### 4. `use_container_width` deprecation warnings
**Symptom:** Logs show `Please replace use_container_width with width` repeated for every chart.
**Cause:** Streamlit 1.56 deprecated `use_container_width=True`.
**Fix:** Replaced all 8 occurrences with `width='stretch'`.

### 5. Sidebar toggle button invisible after closing
**Symptom:** Once the sidebar is collapsed, there is no button visible to reopen it.
**Cause:** Our CSS used `header { visibility: hidden !important }` to hide Streamlit's default toolbar. `visibility: hidden` is **inherited by all child elements**. In Streamlit 1.56, the sidebar toggle button is rendered as a child of `<header>`, so it became invisible too. Previous CSS-only fixes using `[data-testid="collapsedControl"]` and `[data-testid="stSidebarToggle"]` failed because the button's `data-testid` and DOM position change between Streamlit versions.
**Fix (Two-part):**
- Changed `header { visibility: hidden }` to `header { display: none }` in CSS — `display: none` removes the element from layout entirely and does not cascade to children the same way
- Injected a custom JavaScript floating button (`☰`) that is always visible at `position: fixed; top: 12px; left: 12px; z-index: 99999`. This button finds and clicks the native Streamlit toggle using multiple selector fallbacks, making it version-proof

### 6. Large file upload blocked
**Symptom:** Files over 5M cells (e.g. creditcard.csv at 143.8MB) showed `Dataset is very large` error and were rejected.
**Cause:** `validate_dataset()` had a hard 5M cell limit and returned `False`, blocking loading entirely.
**Fix:** Removed the hard block. Added smart chunked sampling for files over 50M cells. Added `.streamlit/config.toml` with `maxUploadSize = 500` to allow up to 500MB uploads.

---

## Testing the App

### Quick test with a synthetic dirty dataset
Run this Python script locally to generate a CSV that triggers every detector:

```python
import pandas as pd
import numpy as np

np.random.seed(42)
n = 500
df = pd.DataFrame({
    "age":      np.random.normal(35, 12, n),
    "income":   np.random.exponential(50000, n),        # right-skewed
    "income2":  np.random.exponential(50000, n) * 1.01, # multicollinear with income
    "score":    np.random.normal(70, 10, n),
    "category": np.random.choice(["A","B","C"], n, p=[0.9, 0.07, 0.03]),  # imbalanced
    "city":     [f"city_{i}" for i in range(n)],         # high cardinality
    "target":   np.random.randint(0, 2, n),
})
# Inject missing values
df.loc[np.random.choice(n, 80), "age"]     = np.nan   # 16% missing
df.loc[np.random.choice(n, 150), "income"] = np.nan   # 30% missing
# Inject outliers
df.loc[0, "score"] = 9999
# Inject duplicates
df = pd.concat([df, df.head(25)], ignore_index=True)

df.to_csv("test_dirty_dataset.csv", index=False)
print("Generated test_dirty_dataset.csv")
```

Expected results after upload:
- Health Score: ~35–50
- Issues: missing values (Critical + High), outliers, skewness, multicollinearity, class imbalance, high cardinality, duplicates
- All 5 sections should populate with content

### Real-world test datasets
| Dataset | Source | What it tests |
|---|---|---|
| Titanic | [Kaggle](https://www.kaggle.com/datasets/yasserh/titanic-dataset) | Missing values, categorical encoding |
| House Prices | [Kaggle](https://www.kaggle.com/datasets/lespin/house-prices-dataset) | Skewness, multicollinearity |
| Credit Card Fraud | [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) | Class imbalance, large file |

---

## Changelog

| Version | Date | Changes |
|---|---|---|
| v1.0 | 2026-04-14 | Initial release — 9 detectors, 5 UI sections, health score |
| v1.1 | 2026-04-14 | Fixed requirements.txt for Python 3.14 (== → >=) |
| v1.2 | 2026-04-14 | Large file support — 500MB limit, chunked sampling, dtype downcasting |
| v1.3 | 2026-04-14 | Fixed health score = 0 bug (detect_all reset + falsy _issues guard) |
| v1.4 | 2026-04-14 | Fixed empty label warnings, use_container_width deprecation |
| v1.5 | 2026-04-14 | Fixed sidebar toggle — JS floating button + display:none CSS fix |
