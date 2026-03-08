import pandas as pd
import numpy as np
from typing import Optional


def detect_column_types(df: pd.DataFrame) -> dict:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    datetime_cols = df.select_dtypes(include=["datetime64"]).columns.tolist()

    for col in categorical_cols[:]:
        try:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                parsed = pd.to_datetime(df[col], errors="coerce")
            if parsed.notna().sum() > len(df) * 0.5:
                datetime_cols.append(col)
                categorical_cols.remove(col)
        except (ValueError, TypeError):
            pass

    return {
        "numeric": numeric_cols,
        "categorical": categorical_cols,
        "datetime": datetime_cols,
    }


def clean_data(
    df: pd.DataFrame,
    missing_strategy: str = "drop",
    remove_duplicates: bool = True,
    fill_value: Optional[dict] = None,
) -> tuple[pd.DataFrame, dict]:
    report = {
        "original_shape": list(df.shape),
        "missing_before": df.isnull().sum().to_dict(),
        "duplicates_before": int(df.duplicated().sum()),
        "actions": [],
    }

    if remove_duplicates:
        before = len(df)
        df = df.drop_duplicates().reset_index(drop=True)
        removed = before - len(df)
        if removed > 0:
            report["actions"].append(f"Removed {removed} duplicate rows")

    col_types = detect_column_types(df)

    if missing_strategy == "drop":
        before = len(df)
        df = df.dropna().reset_index(drop=True)
        removed = before - len(df)
        report["actions"].append(f"Dropped {removed} rows with missing values")

    elif missing_strategy == "mean":
        for col in col_types["numeric"]:
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].mean())
                report["actions"].append(f"Filled '{col}' missing values with mean")
        for col in col_types["categorical"]:
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].mode().iloc[0] if not df[col].mode().empty else "Unknown")
                report["actions"].append(f"Filled '{col}' missing values with mode")

    elif missing_strategy == "median":
        for col in col_types["numeric"]:
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].median())
                report["actions"].append(f"Filled '{col}' missing values with median")
        for col in col_types["categorical"]:
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].mode().iloc[0] if not df[col].mode().empty else "Unknown")
                report["actions"].append(f"Filled '{col}' missing values with mode")

    elif missing_strategy == "mode":
        for col in df.columns:
            if df[col].isnull().any():
                mode_val = df[col].mode()
                df[col] = df[col].fillna(mode_val.iloc[0] if not mode_val.empty else "Unknown")
                report["actions"].append(f"Filled '{col}' missing values with mode")

    elif missing_strategy == "ffill":
        df = df.ffill()
        report["actions"].append("Applied forward fill for missing values")

    elif missing_strategy == "bfill":
        df = df.bfill()
        report["actions"].append("Applied backward fill for missing values")

    report["final_shape"] = list(df.shape)
    report["missing_after"] = df.isnull().sum().to_dict()
    report["column_types"] = col_types

    return df, report
