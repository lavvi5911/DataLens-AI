import pandas as pd
import numpy as np
import io


def load_dataset(file_obj) -> tuple[pd.DataFrame | None, str | None]:
    """
    Load a CSV file uploaded via Streamlit's file_uploader.
    Returns (dataframe, None) on success or (None, error_message) on failure.
    """
    try:
        content = file_obj.read()
        file_obj.seek(0)
        # Try common encodings
        for enc in ("utf-8", "latin-1", "cp1252"):
            try:
                df = pd.read_csv(io.BytesIO(content), encoding=enc, low_memory=False)
                df = _clean_column_names(df)
                df = _coerce_numeric(df)
                return df, None
            except UnicodeDecodeError:
                continue
        return None, "Could not decode the file. Ensure it is a valid CSV with UTF-8 or Latin-1 encoding."
    except pd.errors.EmptyDataError:
        return None, "The uploaded file is empty."
    except pd.errors.ParserError as e:
        return None, f"CSV parsing error: {str(e)[:200]}"
    except Exception as e:
        return None, f"Unexpected error while loading file: {str(e)[:200]}"


def validate_dataset(df: pd.DataFrame) -> tuple[bool, str]:
    """
    Basic validation checks on a loaded dataframe.
    Returns (True, "OK") or (False, reason).
    """
    if df is None or df.empty:
        return False, "Dataset is empty."
    if df.shape[0] < 2:
        return False, "Dataset must have at least 2 rows."
    if df.shape[1] < 1:
        return False, "Dataset must have at least 1 column."
    if df.shape[0] * df.shape[1] > 5_000_000:
        return False, ("Dataset is very large (> 5M cells). "
                       "Consider uploading a sampled subset for best performance.")
    return True, "OK"


def _clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = (
        df.columns
        .str.strip()
        .str.replace(r"\s+", "_", regex=True)
        .str.replace(r"[^\w]", "_", regex=True)
    )
    # Deduplicate column names
    seen = {}
    new_cols = []
    for c in df.columns:
        if c in seen:
            seen[c] += 1
            new_cols.append(f"{c}_{seen[c]}")
        else:
            seen[c] = 0
            new_cols.append(c)
    df.columns = new_cols
    return df


def _coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """
    Try to convert object columns that look numeric into float.
    """
    for col in df.select_dtypes(include="object").columns:
        stripped = df[col].astype(str).str.replace(",", "").str.strip()
        converted = pd.to_numeric(stripped, errors="coerce")
        # Only coerce if ≥90% of non-null values parsed successfully
        non_null = df[col].notna().sum()
        if non_null > 0 and (converted.notna().sum() / non_null) >= 0.9:
            df[col] = converted
    return df
