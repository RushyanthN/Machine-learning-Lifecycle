import pandas as pd
import numpy as np
import io
import uuid
import math
from typing import Dict

_dataframes: Dict[str, pd.DataFrame] = {}


def store_dataframe(df: pd.DataFrame) -> str:
    session_id = str(uuid.uuid4())
    _dataframes[session_id] = df.copy()
    return session_id


def get_dataframe(session_id: str) -> pd.DataFrame:
    if session_id not in _dataframes:
        raise KeyError(f"Session '{session_id}' not found. Please upload data first.")
    return _dataframes[session_id].copy()


def update_dataframe(session_id: str, df: pd.DataFrame) -> None:
    _dataframes[session_id] = df.copy()


def parse_csv(file_bytes: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(file_bytes))


def _sanitize(obj):
    """Replace NaN/Inf with None for JSON serialization."""
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        f = float(obj)
        return None if math.isnan(f) or math.isinf(f) else f
    if isinstance(obj, np.ndarray):
        return _sanitize(obj.tolist())
    return obj


def dataframe_summary(df: pd.DataFrame) -> dict:
    return _sanitize({
        "shape": list(df.shape),
        "columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "missing": df.isnull().sum().to_dict(),
        "duplicates": int(df.duplicated().sum()),
        "head": df.head(5).fillna("").to_dict(orient="records"),
    })
