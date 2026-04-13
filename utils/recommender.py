import pandas as pd
import numpy as np
from typing import Dict, List, Any


class RecommendationEngine:
    """
    Generates actionable preprocessing, feature engineering, and modeling
    recommendations based on the dataset and detected issues.
    """

    def __init__(self, df: pd.DataFrame, detector):
        self.df       = df
        self.detector = detector
        self.issues   = detector._issues if detector._issues else detector.detect_all()
        self.num_df   = df.select_dtypes(include=np.number)
        self.cat_df   = df.select_dtypes(include="object")

    def generate(self) -> Dict[str, List[Dict[str, Any]]]:
        return {
            "preprocessing":       self._preprocessing(),
            "feature_engineering": self._feature_engineering(),
            "encoding":            self._encoding(),
            "modeling":            self._modeling(),
        }

    # ── Preprocessing ──────────────────────────────────────────────────────────

    def _preprocessing(self) -> List[Dict]:
        recs = []
        issue_cats = {i["category"] for i in self.issues}

        # Duplicates
        if "duplicates" in issue_cats:
            n_dup = self.df.duplicated().sum()
            recs.append({
                "title": "Remove Duplicate Rows",
                "detail": f"Drop {n_dup:,} duplicate rows to prevent data leakage and biased evaluation.",
                "code": "df = df.drop_duplicates()"
            })

        # Missing values
        missing_cols = [i["feature"] for i in self.issues if i["category"] == "missing_values"]
        if missing_cols:
            drop_cols   = [i["feature"] for i in self.issues
                           if i["category"] == "missing_values" and i["value"] > 0.5]
            impute_cols = [i["feature"] for i in self.issues
                           if i["category"] == "missing_values" and i["value"] <= 0.5]
            if drop_cols:
                recs.append({
                    "title": "Drop High-Missingness Columns",
                    "detail": f"Columns {drop_cols} have >50% missing — drop them.",
                    "code": f"df = df.drop(columns={drop_cols})"
                })
            if impute_cols:
                recs.append({
                    "title": "Impute Missing Values",
                    "detail": "Use median for numeric columns and mode for categorical columns.",
                    "code": ("from sklearn.impute import SimpleImputer\n"
                             "imp = SimpleImputer(strategy='median')  # or 'most_frequent'\n"
                             "df[num_cols] = imp.fit_transform(df[num_cols])")
                })

        # Outliers
        if "outliers" in issue_cats:
            recs.append({
                "title": "Handle Outliers (Winsorization)",
                "detail": "Cap extreme values at the 1st and 99th percentiles to reduce outlier impact.",
                "code": ("from scipy.stats import mstats\n"
                         "for col in num_cols:\n"
                         "    df[col] = mstats.winsorize(df[col], limits=[0.01, 0.01])")
            })

        # Scaling
        if len(self.num_df.columns) > 0:
            recs.append({
                "title": "Scale Numeric Features",
                "detail": "Standardize or normalize numeric columns for distance-based and gradient models.",
                "code": ("from sklearn.preprocessing import StandardScaler\n"
                         "scaler = StandardScaler()\n"
                         "df[num_cols] = scaler.fit_transform(df[num_cols])")
            })

        # Skewness
        if "skewness" in issue_cats:
            recs.append({
                "title": "Transform Skewed Features",
                "detail": "Apply log1p or Yeo-Johnson power transform to reduce skew.",
                "code": ("import numpy as np\n"
                         "from sklearn.preprocessing import PowerTransformer\n"
                         "pt = PowerTransformer(method='yeo-johnson')\n"
                         "df[skewed_cols] = pt.fit_transform(df[skewed_cols])")
            })

        return recs

    # ── Feature Engineering ────────────────────────────────────────────────────

    def _feature_engineering(self) -> List[Dict]:
        recs = []
        issue_cats = {i["category"] for i in self.issues}

        if "multicollinearity" in issue_cats:
            recs.append({
                "title": "Reduce Multicollinearity with PCA",
                "detail": "Highly correlated features can be collapsed using PCA without losing information.",
                "code": ("from sklearn.decomposition import PCA\n"
                         "pca = PCA(n_components=0.95)  # retain 95% variance\n"
                         "X_pca = pca.fit_transform(X_scaled)")
            })
            recs.append({
                "title": "Variance Inflation Factor (VIF) Selection",
                "detail": "Iteratively drop features with VIF > 10 to eliminate collinear predictors.",
                "code": ("from statsmodels.stats.outliers_influence import variance_inflation_factor\n"
                         "vif = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]\n"
                         "keep = [c for c, v in zip(X.columns, vif) if v < 10]")
            })

        if "low_variance" in issue_cats:
            recs.append({
                "title": "Remove Low-Variance Features",
                "detail": "Drop constant or near-constant columns using VarianceThreshold.",
                "code": ("from sklearn.feature_selection import VarianceThreshold\n"
                         "sel = VarianceThreshold(threshold=0.01)\n"
                         "X_sel = sel.fit_transform(X)")
            })

        recs.append({
            "title": "Feature Importance Screening",
            "detail": "Use a quick RandomForest to rank features by importance before modeling.",
            "code": ("from sklearn.ensemble import RandomForestClassifier  # or Regressor\n"
                     "rf = RandomForestClassifier(n_estimators=100, random_state=42)\n"
                     "rf.fit(X_train, y_train)\n"
                     "importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)")
        })

        return recs

    # ── Encoding ───────────────────────────────────────────────────────────────

    def _encoding(self) -> List[Dict]:
        recs = []
        if self.cat_df.empty:
            return recs

        low_card_cols  = [c for c in self.cat_df.columns if self.df[c].nunique() <= 10]
        high_card_cols = [c for c in self.cat_df.columns if self.df[c].nunique()  > 10]

        if low_card_cols:
            recs.append({
                "title": "One-Hot Encode Low-Cardinality Columns",
                "detail": f"Columns {low_card_cols} have ≤10 unique values — safe for one-hot encoding.",
                "code": (f"df = pd.get_dummies(df, columns={low_card_cols}, drop_first=True)")
            })

        if high_card_cols:
            recs.append({
                "title": "Target / Frequency Encode High-Cardinality Columns",
                "detail": f"Columns {high_card_cols} have many unique values — avoid OHE explosion.",
                "code": ("# Frequency encoding example\n"
                         "for col in high_card_cols:\n"
                         "    freq = df[col].value_counts(normalize=True)\n"
                         "    df[col + '_freq'] = df[col].map(freq)")
            })

        recs.append({
            "title": "Label Encode Ordinal Features",
            "detail": "For ordinal categories (e.g., low/medium/high), use OrdinalEncoder.",
            "code": ("from sklearn.preprocessing import OrdinalEncoder\n"
                     "enc = OrdinalEncoder()\n"
                     "df[ordinal_cols] = enc.fit_transform(df[ordinal_cols])")
        })

        return recs

    # ── Modeling ───────────────────────────────────────────────────────────────

    def _modeling(self) -> List[Dict]:
        recs     = []
        task     = self._detect_task()
        issue_cats = {i["category"] for i in self.issues}

        if task == "classification":
            if "class_imbalance" in issue_cats:
                recs.append({
                    "title": "Handle Imbalance: SMOTE + Class Weights",
                    "detail": "Use SMOTE to oversample minority class and set class_weight='balanced'.",
                    "code": ("from imblearn.over_sampling import SMOTE\n"
                             "sm = SMOTE(random_state=42)\n"
                             "X_res, y_res = sm.fit_resample(X_train, y_train)")
                })
            recs.append({
                "title": "Start with Gradient Boosting (XGBoost / LightGBM)",
                "detail": "Tree-based boosting handles mixed data types, outliers, and class imbalance well.",
                "code": ("from sklearn.ensemble import GradientBoostingClassifier\n"
                         "model = GradientBoostingClassifier(n_estimators=200, learning_rate=0.05)\n"
                         "model.fit(X_train, y_train)")
            })
            recs.append({
                "title": "Cross-Validate with Stratified K-Fold",
                "detail": "Preserve class distribution across folds for reliable evaluation.",
                "code": ("from sklearn.model_selection import StratifiedKFold, cross_val_score\n"
                         "cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)\n"
                         "scores = cross_val_score(model, X, y, cv=cv, scoring='f1_weighted')")
            })

        else:  # regression
            recs.append({
                "title": "Start with Regularized Regression (Ridge / Lasso)",
                "detail": "Regularization handles multicollinearity and prevents overfitting.",
                "code": ("from sklearn.linear_model import Ridge\n"
                         "model = Ridge(alpha=1.0)\n"
                         "model.fit(X_train, y_train)")
            })
            recs.append({
                "title": "Try Gradient Boosting Regressor",
                "detail": "Handles non-linearities and feature interactions automatically.",
                "code": ("from sklearn.ensemble import GradientBoostingRegressor\n"
                         "model = GradientBoostingRegressor(n_estimators=200, max_depth=4)\n"
                         "model.fit(X_train, y_train)")
            })
            recs.append({
                "title": "Evaluate with RMSE + R² + MAE",
                "detail": "Use multiple metrics to get a complete picture of regression quality.",
                "code": ("from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error\n"
                         "rmse = mean_squared_error(y_test, y_pred, squared=False)\n"
                         "r2   = r2_score(y_test, y_pred)")
            })

        return recs

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _detect_task(self) -> str:
        """
        Heuristic: infer classification vs regression from the last column.
        """
        last_col = self.df.iloc[:, -1]
        if last_col.dtype == "object":
            return "classification"
        n_unique = last_col.nunique()
        if n_unique <= 20:
            return "classification"
        return "regression"
