import pandas as pd
import numpy as np
from scipy import stats


class DataAnalyzer:
    """
    Computes comprehensive statistical profiling for a dataframe.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.num_df = df.select_dtypes(include=np.number)
        self.cat_df = df.select_dtypes(include="object")

    def run(self) -> dict:
        return {
            "shape": self.df.shape,
            "dtypes": self.df.dtypes.astype(str).to_dict(),
            "missing": self._missing_stats(),
            "descriptive": self._descriptive_stats(),
            "skewness": self._skewness(),
            "kurtosis": self._kurtosis(),
            "correlation": self._correlation(),
            "cardinality": self._cardinality(),
            "duplicate_count": int(self.df.duplicated().sum()),
            "memory_mb": round(self.df.memory_usage(deep=True).sum() / 1e6, 2),
        }

    def _missing_stats(self) -> dict:
        missing = self.df.isnull().sum()
        pct = (missing / len(self.df) * 100).round(2)
        return {
            "by_column": missing.to_dict(),
            "pct_by_column": pct.to_dict(),
            "total_cells": int(self.df.size),
            "total_missing": int(missing.sum()),
            "total_pct": round(missing.sum() / self.df.size * 100, 2),
        }

    def _descriptive_stats(self) -> dict:
        if self.num_df.empty:
            return {}
        desc = self.num_df.describe().T
        return desc.to_dict()

    def _skewness(self) -> dict:
        if self.num_df.empty:
            return {}
        return self.num_df.skew().round(4).to_dict()

    def _kurtosis(self) -> dict:
        if self.num_df.empty:
            return {}
        return self.num_df.kurtosis().round(4).to_dict()

    def _correlation(self) -> dict:
        if self.num_df.shape[1] < 2:
            return {}
        corr = self.num_df.corr()
        return corr.round(4).to_dict()

    def _cardinality(self) -> dict:
        result = {}
        for col in self.df.columns:
            n_unique = int(self.df[col].nunique())
            n_total  = int(self.df[col].count())
            result[col] = {
                "unique": n_unique,
                "ratio": round(n_unique / n_total, 4) if n_total > 0 else 0,
            }
        return result
