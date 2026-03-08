import pandas as pd
import numpy as np
import json
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from backend.services.cleaner import detect_column_types


def get_correlation_matrix(df: pd.DataFrame) -> dict:
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.shape[1] < 2:
        return {"error": "Need at least 2 numeric columns"}

    corr = numeric_df.corr().round(4)
    fig = go.Figure(data=go.Heatmap(
        z=corr.values, x=corr.columns.tolist(), y=corr.index.tolist(),
        colorscale="RdBu_r", zmin=-1, zmax=1,
        text=np.round(corr.values, 2), texttemplate="%{text}",
    ))
    fig.update_layout(title="Correlation Matrix", template="plotly_white",
                      height=500, margin=dict(l=40, r=40, t=50, b=40))

    return {
        "matrix": corr.to_dict(),
        "plot": json.loads(fig.to_json()),
    }


def get_covariance_matrix(df: pd.DataFrame) -> dict:
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.shape[1] < 2:
        return {"error": "Need at least 2 numeric columns"}

    cov = numeric_df.cov().round(4)
    fig = go.Figure(data=go.Heatmap(
        z=cov.values, x=cov.columns.tolist(), y=cov.index.tolist(),
        colorscale="Viridis",
        text=np.round(cov.values, 2), texttemplate="%{text}",
    ))
    fig.update_layout(title="Covariance Matrix", template="plotly_white",
                      height=500, margin=dict(l=40, r=40, t=50, b=40))

    return {
        "matrix": cov.to_dict(),
        "plot": json.loads(fig.to_json()),
    }


def get_vif(df: pd.DataFrame) -> dict:
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    numeric_df = df.select_dtypes(include=[np.number]).dropna()
    if numeric_df.shape[1] < 2:
        return {"error": "Need at least 2 numeric columns"}

    vif_data = []
    cols = numeric_df.columns.tolist()
    X = numeric_df.values

    for i, col in enumerate(cols):
        try:
            vif_val = variance_inflation_factor(X, i)
            vif_data.append({
                "feature": col,
                "vif": round(float(vif_val), 4),
                "status": "High" if vif_val > 10 else "Moderate" if vif_val > 5 else "Low",
            })
        except Exception:
            vif_data.append({"feature": col, "vif": None, "status": "Error"})

    return {"vif": vif_data}


def get_feature_importance(df: pd.DataFrame, target_col: str, task: str) -> dict:
    numeric_df = df.select_dtypes(include=[np.number]).dropna()
    feature_cols = [c for c in numeric_df.columns if c != target_col]

    if not feature_cols or target_col not in numeric_df.columns:
        return {"error": "Invalid target or insufficient features"}

    X = numeric_df[feature_cols].values
    y = numeric_df[target_col].values

    if task == "classification":
        if y.dtype == object or not np.issubdtype(y.dtype, np.number):
            le = LabelEncoder()
            y = le.fit_transform(y)
        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    else:
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)

    model.fit(X, y)
    importances = model.feature_importances_

    imp_df = pd.DataFrame({
        "feature": feature_cols,
        "importance": importances,
    }).sort_values("importance", ascending=True)

    fig = px.bar(imp_df, x="importance", y="feature", orientation="h",
                 title="Feature Importance (Random Forest)", template="plotly_white")
    fig.update_layout(height=max(300, len(feature_cols) * 30),
                      margin=dict(l=40, r=40, t=50, b=40))

    return {
        "importances": imp_df.sort_values("importance", ascending=False).to_dict(orient="records"),
        "plot": json.loads(fig.to_json()),
    }


def create_column(df: pd.DataFrame, name: str, expression: str) -> pd.DataFrame:
    """Create a new column using a pandas eval expression."""
    df = df.copy()
    df[name] = df.eval(expression)
    return df


def delete_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    df = df.copy()
    return df.drop(columns=columns, errors="ignore")
