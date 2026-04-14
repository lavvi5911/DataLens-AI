import pandas as pd
import numpy as np
from scipy import stats
from typing import List, Dict, Any


class IssueDetector:
    """
    Detects data quality and ML-readiness issues in a DataFrame.
    """

    MISSING_HIGH   = 0.30   # >30% missing → critical
    MISSING_MED    = 0.10   # >10% missing → high
    MISSING_LOW    = 0.01   # >1%  missing → medium
    CORR_THRESHOLD = 0.85   # high multicollinearity
    LEAKAGE_THRESH = 0.95   # suspicious correlation with target
    SKEW_HIGH      = 2.0
    SKEW_MED       = 1.0
    IMBALANCE_RATIO = 0.10  # minority class < 10% of total
    VARIANCE_THRESH = 1e-6  # near-constant columns
    Z_SCORE_THRESH  = 3.5

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.num_df = df.select_dtypes(include=np.number)
        self.cat_df = df.select_dtypes(include="object")
        self._issues: List[Dict[str, Any]] = []
        self._health_deductions: float = 0.0
        self._detected: bool = False

    # ── Public API ─────────────────────────────────────────────────────────────

    def detect_all(self) -> List[Dict[str, Any]]:
        self._issues = []
        self._health_deductions = 0.0
        self._check_missing()
        self._check_duplicates()
        self._check_outliers()
        self._check_skewness()
        self._check_low_variance()
        self._check_multicollinearity()
        self._check_class_imbalance()
        self._check_data_leakage()
        self._check_high_cardinality()
        self._detected = True
        return self._issues

    def health_score(self) -> int:
        if not self._detected:
            self.detect_all()
        score = max(0, min(100, 100 - int(self._health_deductions)))
        return score

    # ── Checkers ───────────────────────────────────────────────────────────────

    def _check_missing(self):
        miss = self.df.isnull().mean()
        for col, pct in miss.items():
            if pct == 0:
                continue
            if pct > self.MISSING_HIGH:
                sev, ded = "critical", 15
            elif pct > self.MISSING_MED:
                sev, ded = "high", 8
            elif pct > self.MISSING_LOW:
                sev, ded = "medium", 3
            else:
                sev, ded = "low", 1
            self._add(
                severity=sev,
                title=f"Missing Values: {col}",
                description=(f"Column '{col}' has {pct*100:.1f}% missing values "
                             f"({int(pct*len(self.df)):,} of {len(self.df):,} rows)."),
                fix=("Drop this column" if pct > 0.5
                     else "Impute with median/mode or use a missing-indicator flag."),
                feature=col,
                category="missing_values",
                value=round(pct, 4),
                deduction=ded,
            )

    def _check_duplicates(self):
        n_dup = self.df.duplicated().sum()
        if n_dup == 0:
            return
        pct = n_dup / len(self.df)
        sev = "high" if pct > 0.05 else "medium"
        self._add(
            severity=sev,
            title="Duplicate Rows Detected",
            description=f"{n_dup:,} duplicate rows found ({pct*100:.1f}% of dataset).",
            fix="Remove duplicates using df.drop_duplicates() before training.",
            feature=None,
            category="duplicates",
            value=int(n_dup),
            deduction=8 if pct > 0.05 else 4,
        )

    def _check_outliers(self):
        for col in self.num_df.columns:
            col_data = self.num_df[col].dropna()
            if len(col_data) < 10:
                continue
            # IQR method
            q1, q3 = col_data.quantile(0.25), col_data.quantile(0.75)
            iqr = q3 - q1
            if iqr == 0:
                continue
            n_iqr = int(((col_data < (q1 - 1.5 * iqr)) | (col_data > (q3 + 1.5 * iqr))).sum())
            pct_out = n_iqr / len(col_data)
            if pct_out > 0.05:
                sev = "high" if pct_out > 0.15 else "medium"
                self._add(
                    severity=sev,
                    title=f"Outliers Detected: {col}",
                    description=(f"'{col}' has {n_iqr} outliers ({pct_out*100:.1f}% of values) "
                                 f"based on the IQR method."),
                    fix="Cap outliers (Winsorizing), log-transform, or use a robust scaler.",
                    feature=col,
                    category="outliers",
                    value=n_iqr,
                    deduction=5 if pct_out > 0.15 else 2,
                )

    def _check_skewness(self):
        if self.num_df.empty:
            return
        skew = self.num_df.skew()
        for col, val in skew.items():
            abs_skew = abs(val)
            if abs_skew > self.SKEW_HIGH:
                sev, ded = "high", 5
            elif abs_skew > self.SKEW_MED:
                sev, ded = "medium", 2
            else:
                continue
            direction = "right (positive)" if val > 0 else "left (negative)"
            self._add(
                severity=sev,
                title=f"High Skewness: {col}",
                description=(f"'{col}' has a skewness of {val:.2f}, indicating a {direction}-skewed "
                             f"distribution. This may hurt linear model performance."),
                fix=("Apply log1p(), Box-Cox, or Yeo-Johnson transformation to normalize distribution."
                     if abs_skew > self.SKEW_HIGH
                     else "Consider a mild normalization or power transform."),
                feature=col,
                category="skewness",
                value=round(val, 4),
                deduction=ded,
            )

    def _check_low_variance(self):
        for col in self.num_df.columns:
            col_data = self.num_df[col].dropna()
            if len(col_data) == 0:
                continue
            std = col_data.std()
            n_unique = col_data.nunique()
            if n_unique == 1:
                self._add(
                    severity="high",
                    title=f"Constant Column: {col}",
                    description=f"'{col}' has only one unique value and provides zero predictive value.",
                    fix="Drop this column — it carries no information.",
                    feature=col,
                    category="low_variance",
                    value=0,
                    deduction=6,
                )
            elif n_unique <= 2 or std < self.VARIANCE_THRESH:
                self._add(
                    severity="medium",
                    title=f"Near-Constant Column: {col}",
                    description=f"'{col}' has very low variance (std={std:.6f}), which rarely benefits models.",
                    fix="Consider dropping or binning this feature.",
                    feature=col,
                    category="low_variance",
                    value=round(float(std), 6),
                    deduction=3,
                )

    def _check_multicollinearity(self):
        if self.num_df.shape[1] < 2:
            return
        corr = self.num_df.corr().abs()
        seen = set()
        for i, col_a in enumerate(corr.columns):
            for j, col_b in enumerate(corr.columns):
                if i >= j:
                    continue
                pair = frozenset((col_a, col_b))
                if pair in seen:
                    continue
                val = corr.loc[col_a, col_b]
                if val >= self.CORR_THRESHOLD:
                    seen.add(pair)
                    sev = "high" if val >= 0.95 else "medium"
                    self._add(
                        severity=sev,
                        title=f"High Multicollinearity: {col_a} & {col_b}",
                        description=(f"'{col_a}' and '{col_b}' are {val*100:.1f}% correlated. "
                                     f"Keeping both may cause redundancy and inflate model complexity."),
                        fix=("Drop one of the features or apply PCA / VIF-based selection."),
                        feature=f"{col_a},{col_b}",
                        category="multicollinearity",
                        value=round(float(val), 4),
                        deduction=5 if val >= 0.95 else 3,
                    )

    def _check_class_imbalance(self):
        """Check the last object column or binary numeric column as a proxy target."""
        # Try categorical columns first
        cat_cols = self.df.select_dtypes(include="object").columns.tolist()
        # Also include numeric columns with ≤20 unique values as potential targets
        num_targets = [c for c in self.num_df.columns if 2 <= self.num_df[c].nunique() <= 20]

        candidates = cat_cols[-1:] + num_targets[-1:]  # use last column as likely target
        checked = set()
        for col in candidates:
            if col in checked:
                continue
            checked.add(col)
            vc = self.df[col].value_counts(normalize=True)
            if len(vc) < 2 or len(vc) > 50:
                continue
            minority_pct = float(vc.min())
            if minority_pct < self.IMBALANCE_RATIO:
                sev = "high" if minority_pct < 0.05 else "medium"
                self._add(
                    severity=sev,
                    title=f"Class Imbalance: {col}",
                    description=(f"'{col}' (potential target) has a minority class of only "
                                 f"{minority_pct*100:.1f}%. This can bias classification models."),
                    fix=("Use SMOTE oversampling, class_weight='balanced', or stratified k-fold CV."),
                    feature=col,
                    category="class_imbalance",
                    value=round(minority_pct, 4),
                    deduction=8 if minority_pct < 0.05 else 5,
                )
            break  # only check most likely target

    def _check_data_leakage(self):
        if self.num_df.shape[1] < 2:
            return
        corr = self.num_df.corr().abs()
        # Heuristic: check last numeric column as target proxy
        cols = corr.columns.tolist()
        if len(cols) < 2:
            return
        target = cols[-1]
        for col in cols[:-1]:
            val = corr.loc[col, target]
            if val >= self.LEAKAGE_THRESH:
                self._add(
                    severity="critical",
                    title=f"Potential Data Leakage: {col}",
                    description=(f"'{col}' is {val*100:.1f}% correlated with '{target}' (assumed target). "
                                 f"This could indicate the feature leaks target information."),
                    fix=(f"Investigate whether '{col}' is derived from the target or is unavailable at inference time."),
                    feature=col,
                    category="data_leakage",
                    value=round(float(val), 4),
                    deduction=20,
                )

    def _check_high_cardinality(self):
        for col in self.cat_df.columns:
            n_unique = self.df[col].nunique()
            n_rows   = len(self.df)
            ratio    = n_unique / n_rows
            if ratio > 0.8 and n_unique > 50:
                self._add(
                    severity="medium",
                    title=f"High Cardinality: {col}",
                    description=(f"'{col}' has {n_unique:,} unique values ({ratio*100:.0f}% of rows). "
                                 f"One-hot encoding will create an explosion of features."),
                    fix="Use target encoding, frequency encoding, or drop this column.",
                    feature=col,
                    category="high_cardinality",
                    value=n_unique,
                    deduction=4,
                )

    # ── Internal ───────────────────────────────────────────────────────────────

    def _add(self, severity, title, description, fix, feature, category, value, deduction):
        self._issues.append({
            "severity":    severity,
            "title":       title,
            "description": description,
            "fix":         fix,
            "feature":     feature,
            "category":    category,
            "value":       value,
        })
        self._health_deductions += deduction
