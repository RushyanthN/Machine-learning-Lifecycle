import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder
from typing import Optional


def apply_scaling(df: pd.DataFrame, columns: list[str], method: str = "standard") -> tuple[pd.DataFrame, dict]:
    df = df.copy()
    info = {"method": method, "columns": columns}

    if method == "standard":
        scaler = StandardScaler()
    elif method == "minmax":
        scaler = MinMaxScaler()
    elif method == "robust":
        scaler = RobustScaler()
    else:
        raise ValueError(f"Unknown scaling method: {method}")

    df[columns] = scaler.fit_transform(df[columns])
    info["stats"] = {col: {"mean": float(df[col].mean()), "std": float(df[col].std())} for col in columns}
    return df, info


def apply_encoding(df: pd.DataFrame, columns: list[str], method: str = "onehot") -> tuple[pd.DataFrame, dict]:
    df = df.copy()
    info = {"method": method, "columns": columns, "new_columns": []}

    if method == "onehot":
        df = pd.get_dummies(df, columns=columns, drop_first=False, dtype=int)
        info["new_columns"] = [c for c in df.columns if any(c.startswith(f"{col}_") for col in columns)]
    elif method == "label":
        for col in columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            info["new_columns"].append(col)
    else:
        raise ValueError(f"Unknown encoding method: {method}")

    return df, info


def apply_log_transform(df: pd.DataFrame, columns: list[str]) -> tuple[pd.DataFrame, dict]:
    df = df.copy()
    info = {"columns": columns, "warnings": []}

    for col in columns:
        min_val = df[col].min()
        if min_val <= 0:
            shift = abs(min_val) + 1
            df[col] = np.log1p(df[col] + shift)
            info["warnings"].append(f"'{col}' had values <= 0; shifted by {shift} before log transform")
        else:
            df[col] = np.log1p(df[col])

    return df, info


def apply_polynomial_features(df: pd.DataFrame, columns: list[str], degree: int = 2) -> tuple[pd.DataFrame, dict]:
    df = df.copy()
    new_cols = []

    for col in columns:
        for d in range(2, degree + 1):
            new_col_name = f"{col}_pow{d}"
            df[new_col_name] = df[col] ** d
            new_cols.append(new_col_name)

    if len(columns) >= 2:
        for i in range(len(columns)):
            for j in range(i + 1, len(columns)):
                new_col_name = f"{columns[i]}_x_{columns[j]}"
                df[new_col_name] = df[columns[i]] * df[columns[j]]
                new_cols.append(new_col_name)

    return df, {"new_columns": new_cols, "degree": degree}


def get_transform_preview(df: pd.DataFrame) -> dict:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    return {
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "shape": list(df.shape),
        "head": df.head(5).to_dict(orient="records"),
    }
