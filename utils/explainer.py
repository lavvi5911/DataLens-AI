import pandas as pd
import numpy as np
from typing import List, Dict, Any


class InsightExplainer:
    """
    Converts raw statistical findings into plain-English, business-friendly insights.
    No external NLP model required — uses template-driven generation with dynamic data.
    """

    def __init__(self, df: pd.DataFrame, detector, analyzer):
        self.df       = df
        self.detector = detector
        self.analyzer = analyzer
        self.issues   = detector._issues if detector._issues else detector.detect_all()
        self.analysis = analyzer.run()
        self.num_df   = df.select_dtypes(include=np.number)
        self.cat_df   = df.select_dtypes(include="object")

    def generate(self) -> List[Dict[str, Any]]:
        insights = []
        insights += self._dataset_summary()
        insights += self._missing_insights()
        insights += self._outlier_insights()
        insights += self._skewness_insights()
        insights += self._multicollinearity_insights()
        insights += self._class_balance_insights()
        insights += self._leakage_insights()
        insights += self._overall_quality()
        return insights

    # ── Insight generators ─────────────────────────────────────────────────────

    def _dataset_summary(self) -> List[Dict]:
        n_rows, n_cols = self.df.shape
        n_num = len(self.num_df.columns)
        n_cat = len(self.cat_df.columns)
        miss_pct = self.analysis["missing"]["total_pct"]
        dup = self.analysis["duplicate_count"]
        mem = self.analysis["memory_mb"]
        return [{
            "type": "info",
            "title": "Dataset at a Glance",
            "body": (
                f"Your dataset contains <strong>{n_rows:,} rows</strong> and "
                f"<strong>{n_cols} columns</strong> "
                f"({n_num} numeric, {n_cat} categorical). "
                f"It occupies <strong>{mem} MB</strong> in memory. "
                + (f"Overall <strong>{miss_pct}% of cells are missing</strong>, "
                   f"which requires attention before training. "
                   if miss_pct > 1 else "Missing data is minimal — great. ")
                + (f"<strong>{dup:,} duplicate rows</strong> were found and should be removed."
                   if dup > 0 else "No duplicate rows detected.")
            ),
            "action": None,
        }]

    def _missing_insights(self) -> List[Dict]:
        miss_issues = [i for i in self.issues if i["category"] == "missing_values"]
        if not miss_issues:
            return [{
                "type": "success",
                "title": "No Missing Values",
                "body": "Every cell in your dataset is filled. This is ideal and means you can skip imputation.",
                "action": None,
            }]
        worst = max(miss_issues, key=lambda x: x["value"])
        pct = worst["value"] * 100
        body = (
            f"<strong>{len(miss_issues)} features</strong> have missing values. "
            f"The worst offender is <strong>'{worst['feature']}'</strong> with {pct:.1f}% missing. "
        )
        if pct > 50:
            body += (
                f"Columns with >50% missing data are generally not worth keeping — "
                f"they introduce more noise than signal and can mislead your model."
            )
        elif pct > 20:
            body += (
                f"A missingness rate above 20% warrants careful attention. "
                f"Consider whether the data is missing at random (MAR) or not at random (MNAR), "
                f"as this affects the best imputation strategy."
            )
        else:
            body += (
                f"Moderate missingness can be handled safely with median/mode imputation "
                f"or by training a model that handles NaN natively (e.g. XGBoost)."
            )
        return [{
            "type": "warning",
            "title": f"Missing Data Found in {len(miss_issues)} Column(s)",
            "body": body,
            "action": "Use SimpleImputer or KNNImputer from scikit-learn to fill gaps.",
        }]

    def _outlier_insights(self) -> List[Dict]:
        out_issues = [i for i in self.issues if i["category"] == "outliers"]
        if not out_issues:
            return []
        worst = max(out_issues, key=lambda x: x["value"])
        return [{
            "type": "warning",
            "title": f"Outliers in {len(out_issues)} Feature(s)",
            "body": (
                f"<strong>{len(out_issues)} numeric features</strong> contain statistical outliers. "
                f"The most affected is <strong>'{worst['feature']}'</strong> with {worst['value']} outlier points. "
                f"Outliers can severely distort linear models, SVMs, and k-means clustering, "
                f"while tree-based models are more robust to them. "
                f"Before deciding, visualize the distribution — some 'outliers' may be legitimate edge cases."
            ),
            "action": "Try Winsorization (clip at 1st/99th percentile) or robust scaling.",
        }]

    def _skewness_insights(self) -> List[Dict]:
        skew_issues = [i for i in self.issues if i["category"] == "skewness"]
        if not skew_issues:
            return []
        high = [i for i in skew_issues if abs(i["value"]) > 2]
        worst = max(skew_issues, key=lambda x: abs(x["value"]))
        return [{
            "type": "warning" if high else "info",
            "title": f"Skewed Distributions in {len(skew_issues)} Feature(s)",
            "body": (
                f"<strong>{len(skew_issues)} features</strong> show skewed distributions. "
                f"<strong>'{worst['feature']}'</strong> has the highest skewness of {worst['value']:.2f}. "
                + (f"<strong>{len(high)} feature(s) have severe skewness (>2)</strong>, "
                   f"which can break assumptions in linear and logistic regression. "
                   if high else "")
                + f"A skewed feature means most values cluster at one end, "
                  f"with a long tail pulling the mean away from the median. "
                  f"Log or power transforms typically resolve this."
            ),
            "action": "Apply np.log1p() for right-skewed or PowerTransformer for general skewness.",
        }]

    def _multicollinearity_insights(self) -> List[Dict]:
        mc_issues = [i for i in self.issues if i["category"] == "multicollinearity"]
        if not mc_issues:
            return []
        worst = max(mc_issues, key=lambda x: x["value"])
        pair  = worst["feature"].split(",")
        return [{
            "type": "warning",
            "title": f"Redundant Features Detected ({len(mc_issues)} Pairs)",
            "body": (
                f"<strong>{len(mc_issues)} pairs</strong> of numeric features are highly correlated (>85%). "
                f"The strongest pair is <strong>'{pair[0]}' ↔ '{pair[1]}'</strong> "
                f"at {worst['value']*100:.1f}% correlation. "
                f"Redundant features don't add information but do add complexity, slow training, "
                f"and can destabilize coefficient estimates in linear models. "
                f"In tree models they dilute feature importance scores."
            ),
            "action": "Drop one from each correlated pair, or use PCA to compress them.",
        }]

    def _class_balance_insights(self) -> List[Dict]:
        ci_issues = [i for i in self.issues if i["category"] == "class_imbalance"]
        if not ci_issues:
            return []
        iss = ci_issues[0]
        pct = iss["value"] * 100
        return [{
            "type": "error" if pct < 5 else "warning",
            "title": f"Class Imbalance in '{iss['feature']}'",
            "body": (
                f"The target column <strong>'{iss['feature']}'</strong> has a minority class "
                f"of only <strong>{pct:.1f}%</strong>. "
                + (f"At this extreme level, a naive model can achieve high accuracy "
                   f"by simply predicting the majority class every time — making accuracy a useless metric. "
                   if pct < 5
                   else f"This moderate imbalance can lead to poor recall on the minority class. ")
                + f"Evaluate using F1-score, AUC-ROC, or precision-recall curves instead of plain accuracy."
            ),
            "action": "Use SMOTE for oversampling, or set class_weight='balanced' in your estimator.",
        }]

    def _leakage_insights(self) -> List[Dict]:
        lk_issues = [i for i in self.issues if i["category"] == "data_leakage"]
        if not lk_issues:
            return []
        return [{
            "type": "error",
            "title": f"⚡ Potential Data Leakage Detected!",
            "body": (
                f"<strong>{len(lk_issues)} feature(s)</strong> are suspiciously correlated (>95%) "
                f"with what appears to be the target column. "
                f"Data leakage is one of the most dangerous silent bugs in ML — "
                f"it makes your model look great in training but fail catastrophically in production. "
                f"Common causes: derived features, post-event data, or ID columns accidentally correlated with the target."
            ),
            "action": "Audit each flagged feature — verify it's available at prediction time.",
        }]

    def _overall_quality(self) -> List[Dict]:
        score  = self.detector.health_score()
        n_crit = sum(1 for i in self.issues if i["severity"] == "critical")
        n_high = sum(1 for i in self.issues if i["severity"] == "high")

        if score >= 80:
            return [{
                "type": "success",
                "title": "Overall: Good Data Quality",
                "body": (
                    f"Your dataset scores <strong>{score}/100</strong> for ML readiness. "
                    f"The data is relatively clean with few quality issues. "
                    f"Follow the minor recommendations and you'll be ready to train models confidently."
                ),
                "action": None,
            }]
        elif score >= 55:
            return [{
                "type": "warning",
                "title": "Overall: Moderate Issues — Attention Needed",
                "body": (
                    f"Your dataset scores <strong>{score}/100</strong>. "
                    f"There are {n_high} high-severity issues that need addressing before training. "
                    f"Skipping these steps could lead to significantly degraded model performance or misleading results."
                ),
                "action": "Work through the Recommendations tab systematically.",
            }]
        else:
            return [{
                "type": "error",
                "title": "Overall: Critical Data Quality Problems",
                "body": (
                    f"Your dataset scores only <strong>{score}/100</strong>. "
                    f"There are <strong>{n_crit} critical</strong> and {n_high} high-severity issues. "
                    f"Training a model on this data as-is will likely produce unreliable or biased results. "
                    f"Significant data cleaning is required."
                ),
                "action": "Address all critical issues before proceeding to modeling.",
            }]
