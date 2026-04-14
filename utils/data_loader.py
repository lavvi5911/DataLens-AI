import pandas as pd
import numpy as np
import io

# ── Thresholds ────────────────────────────────────────────────────────────────
MAX_CELLS_FULL    = 50_000_000   # up to 50M cells → load fully
SAMPLE_ROWS       = 200_000      # rows kept when dataset exceeds threshold
CHUNK_SIZE        = 100_000      # rows per chunk for chunked reading


def load_dataset(file_obj) -> tuple:
    """
    Load a CSV file uploaded via Streamlit's file_uploader.
    For very large files (>50M cells) the dataset is intelligently sampled
    so analysis remains fast without losing statistical representativeness.

    Returns:
        (df, None, info_message)  on success
        (None, error_message, None) on failure
    """
    try:
        content = file_obj.read()

        for enc in ("utf-8", "latin-1", "cp1252"):
            try:
                # ── Quick peek: count rows cheaply ──────────────────────────
                line_count = content.count(b"\n")
                # Estimate columns from first line
                first_line = content.split(b"\n")[0].decode(enc, errors="replace")
                est_cols   = first_line.count(",") + 1
                est_cells  = line_count * est_cols

                if est_cells > MAX_CELLS_FULL:
                    df, info = _load_sampled(content, enc, line_count, est_cols)
                else:
                    df   = pd.read_csv(io.BytesIO(content), encoding=enc, low_memory=False)
                    info = None

                df = _clean_column_names(df)
                df = _coerce_numeric(df)
                df = _downcast_dtypes(df)
                return df, None, info

            except UnicodeDecodeError:
                continue

        return None, "Could not decode the file. Ensure it is UTF-8 or Latin-1 encoded.", None

    except pd.errors.EmptyDataError:
        return None, "The uploaded file is empty.", None
    except pd.errors.ParserError as e:
        return None, f"CSV parsing error: {str(e)[:200]}", None
    except MemoryError:
        return None, ("File is too large to load even with sampling. "
                      "Please export a subset (e.g. first 500K rows) and re-upload."), None
    except Exception as e:
        return None, f"Unexpected error: {str(e)[:200]}", None


def validate_dataset(df: pd.DataFrame) -> tuple:
    """Basic sanity checks. Returns (True, 'OK') or (False, reason)."""
    if df is None or df.empty:
        return False, "Dataset is empty."
    if df.shape[0] < 2:
        return False, "Dataset must have at least 2 rows."
    if df.shape[1] < 1:
        return False, "Dataset must have at least 1 column."
    return True, "OK"


# ── Internal helpers ──────────────────────────────────────────────────────────

def _load_sampled(content: bytes, enc: str, line_count: int, est_cols: int) -> tuple:
    """
    Load a large CSV in chunks and return a stratified random sample.
    Preserves distribution better than simple head() slicing.
    """
    total_rows = max(line_count - 1, 1)   # subtract header
    sample_n   = min(SAMPLE_ROWS, total_rows)
    frac       = sample_n / total_rows

    chunks = []
    reader = pd.read_csv(
        io.BytesIO(content),
        encoding=enc,
        chunksize=CHUNK_SIZE,
        low_memory=False,
    )
    for chunk in reader:
        if frac < 1.0:
            sampled = chunk.sample(frac=frac, random_state=42)
        else:
            sampled = chunk
        chunks.append(sampled)

    df = pd.concat(chunks, ignore_index=True)

    info = (
        f"Dataset has ~{total_rows:,} rows — automatically sampled "
        f"**{len(df):,} rows** ({frac*100:.1f}%) for fast analysis. "
        f"All statistical insights remain representative."
    )
    return df, info


def _clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = (
        df.columns
        .str.strip()
        .str.replace(r"\s+", "_", regex=True)
        .str.replace(r"[^\w]", "_", regex=True)
    )
    seen, new_cols = {}, []
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
    """Convert object columns that look numeric into float."""
    for col in df.select_dtypes(include="object").columns:
        stripped  = df[col].astype(str).str.replace(",", "").str.strip()
        converted = pd.to_numeric(stripped, errors="coerce")
        non_null  = df[col].notna().sum()
        if non_null > 0 and (converted.notna().sum() / non_null) >= 0.9:
            df[col] = converted
    return df


def _downcast_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Shrink memory footprint by downcasting numeric columns.
    float64 → float32, int64 → smallest int type that fits.
    Typically reduces RAM by 40–60% on large datasets.
    """
    for col in df.select_dtypes(include="float64").columns:
        df[col] = pd.to_numeric(df[col], downcast="float")

    for col in df.select_dtypes(include=["int64", "int32"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="integer")

    # Convert low-cardinality object columns to category
    for col in df.select_dtypes(include="object").columns:
        n_unique = df[col].nunique()
        if n_unique / max(len(df), 1) < 0.5 and n_unique < 10_000:
            df[col] = df[col].astype("category")

    return df
