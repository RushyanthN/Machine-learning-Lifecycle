import pandas as pd
import numpy as np
import json
from backend.services.cleaner import detect_column_types


def get_data_overview(df: pd.DataFrame) -> dict:
    col_types = detect_column_types(df)
    return {
        "shape": list(df.shape),
        "columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "column_types": col_types,
        "head": df.head(5).fillna("").to_dict(orient="records"),
        "tail": df.tail(5).fillna("").to_dict(orient="records"),
    }


def get_info(df: pd.DataFrame) -> list[dict]:
    rows = []
    for col in df.columns:
        rows.append({
            "column": col,
            "dtype": str(df[col].dtype),
            "non_null": int(df[col].notna().sum()),
            "null": int(df[col].isna().sum()),
            "null_pct": round(float(df[col].isna().mean() * 100), 2),
            "unique": int(df[col].nunique()),
        })
    return rows


def get_describe(df: pd.DataFrame) -> dict:
    numeric_desc = df.describe().round(4)
    result = {"numeric": numeric_desc.to_dict()}

    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    if len(cat_cols) > 0:
        cat_desc = df[cat_cols].describe()
        result["categorical"] = cat_desc.to_dict()

    return result


def get_null_analysis(df: pd.DataFrame) -> dict:
    null_counts = df.isnull().sum()
    null_pct = (df.isnull().mean() * 100).round(2)
    return {
        "total_nulls": int(null_counts.sum()),
        "total_cells": int(df.shape[0] * df.shape[1]),
        "null_pct_overall": round(float(null_counts.sum() / (df.shape[0] * df.shape[1]) * 100), 2),
        "per_column": {
            col: {"count": int(null_counts[col]), "pct": float(null_pct[col])}
            for col in df.columns if null_counts[col] > 0
        },
    }


def get_value_counts(df: pd.DataFrame, column: str, top_n: int = 20) -> dict:
    vc = df[column].value_counts().head(top_n)
    return {
        "column": column,
        "counts": {str(k): int(v) for k, v in vc.items()},
        "unique_total": int(df[column].nunique()),
    }
