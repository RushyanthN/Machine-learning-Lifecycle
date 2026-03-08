import pandas as pd
import numpy as np
import json
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve,
    precision_recall_curve, classification_report,
)
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.svm import SVR, SVC
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBRegressor, XGBClassifier


REGRESSION_MODELS = {
    "Linear Regression": LinearRegression,
    "Ridge Regression": Ridge,
    "Lasso Regression": Lasso,
    "Decision Tree": DecisionTreeRegressor,
    "Random Forest": RandomForestRegressor,
    "KNN": KNeighborsRegressor,
    "SVR": SVR,
    "XGBoost": XGBRegressor,
}

CLASSIFICATION_MODELS = {
    "Logistic Regression": LogisticRegression,
    "Decision Tree": DecisionTreeClassifier,
    "Random Forest": RandomForestClassifier,
    "KNN": KNeighborsClassifier,
    "SVC": SVC,
    "XGBoost": XGBClassifier,
    "Naive Bayes": GaussianNB,
}

DEFAULT_HYPERPARAMS = {
    "Linear Regression": {},
    "Ridge Regression": {"alpha": 1.0},
    "Lasso Regression": {"alpha": 1.0},
    "Decision Tree": {"max_depth": None, "min_samples_split": 2, "min_samples_leaf": 1},
    "Random Forest": {"n_estimators": 100, "max_depth": None, "min_samples_split": 2, "min_samples_leaf": 1},
    "KNN": {"n_neighbors": 5, "weights": "uniform"},
    "SVR": {"C": 1.0, "kernel": "rbf"},
    "SVC": {"C": 1.0, "kernel": "rbf"},
    "XGBoost": {"n_estimators": 100, "max_depth": 6, "learning_rate": 0.1},
    "Logistic Regression": {"C": 1.0, "max_iter": 1000},
    "Naive Bayes": {},
}

HYPERPARAM_OPTIONS = {
    "Ridge Regression": {
        "alpha": {"type": "float", "min": 0.01, "max": 100.0, "default": 1.0},
    },
    "Lasso Regression": {
        "alpha": {"type": "float", "min": 0.01, "max": 100.0, "default": 1.0},
    },
    "Decision Tree": {
        "max_depth": {"type": "int_or_none", "min": 1, "max": 50, "default": None},
        "min_samples_split": {"type": "int", "min": 2, "max": 50, "default": 2},
        "min_samples_leaf": {"type": "int", "min": 1, "max": 50, "default": 1},
    },
    "Random Forest": {
        "n_estimators": {"type": "int", "min": 10, "max": 500, "default": 100},
        "max_depth": {"type": "int_or_none", "min": 1, "max": 50, "default": None},
        "min_samples_split": {"type": "int", "min": 2, "max": 50, "default": 2},
        "min_samples_leaf": {"type": "int", "min": 1, "max": 50, "default": 1},
    },
    "KNN": {
        "n_neighbors": {"type": "int", "min": 1, "max": 50, "default": 5},
        "weights": {"type": "select", "options": ["uniform", "distance"], "default": "uniform"},
    },
    "SVR": {
        "C": {"type": "float", "min": 0.01, "max": 100.0, "default": 1.0},
        "kernel": {"type": "select", "options": ["rbf", "linear", "poly", "sigmoid"], "default": "rbf"},
    },
    "SVC": {
        "C": {"type": "float", "min": 0.01, "max": 100.0, "default": 1.0},
        "kernel": {"type": "select", "options": ["rbf", "linear", "poly", "sigmoid"], "default": "rbf"},
    },
    "XGBoost": {
        "n_estimators": {"type": "int", "min": 10, "max": 500, "default": 100},
        "max_depth": {"type": "int", "min": 1, "max": 20, "default": 6},
        "learning_rate": {"type": "float", "min": 0.001, "max": 1.0, "default": 0.1},
    },
    "Logistic Regression": {
        "C": {"type": "float", "min": 0.01, "max": 100.0, "default": 1.0},
        "max_iter": {"type": "int", "min": 100, "max": 5000, "default": 1000},
    },
    "Linear Regression": {},
    "Naive Bayes": {},
}


def _build_model(name: str, task: str, params: dict):
    registry = REGRESSION_MODELS if task == "regression" else CLASSIFICATION_MODELS
    cls = registry[name]
    p = dict(params)

    if "random_state" not in p and name not in ("KNN", "Naive Bayes"):
        p["random_state"] = 42
    if name == "SVC":
        p["probability"] = True
    if name == "XGBoost" and task == "classification":
        p.setdefault("eval_metric", "logloss")
    if name == "XGBoost":
        p["verbosity"] = 0

    clean = {}
    for k, v in p.items():
        if v == "None":
            clean[k] = None
        else:
            clean[k] = v

    try:
        return cls(**clean)
    except TypeError:
        allowed = set(cls.__init__.__code__.co_varnames)
        return cls(**{k: v for k, v in clean.items() if k in allowed})


def evaluate_regression(y_true, y_pred) -> dict:
    return {
        "MAE": round(float(mean_absolute_error(y_true, y_pred)), 4),
        "MSE": round(float(mean_squared_error(y_true, y_pred)), 4),
        "RMSE": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 4),
        "R2": round(float(r2_score(y_true, y_pred)), 4),
    }


def evaluate_classification(y_true, y_pred, y_proba=None) -> dict:
    n_classes = len(np.unique(y_true))
    average = "binary" if n_classes == 2 else "weighted"

    metrics = {
        "Accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "Precision": round(float(precision_score(y_true, y_pred, average=average, zero_division=0)), 4),
        "Recall": round(float(recall_score(y_true, y_pred, average=average, zero_division=0)), 4),
        "F1": round(float(f1_score(y_true, y_pred, average=average, zero_division=0)), 4),
    }

    cm = confusion_matrix(y_true, y_pred)
    metrics["Confusion Matrix"] = cm.tolist()

    if y_proba is not None:
        try:
            if n_classes == 2:
                auc_score = roc_auc_score(y_true, y_proba[:, 1] if y_proba.ndim == 2 else y_proba)
            else:
                auc_score = roc_auc_score(y_true, y_proba, multi_class="ovr", average="weighted")
            metrics["ROC-AUC"] = round(float(auc_score), 4)
        except Exception:
            metrics["ROC-AUC"] = None

    return metrics


def _diagnostic_regression(model, X_test, y_test, y_pred) -> dict:
    residuals = y_test - y_pred

    fig_resid = px.scatter(
        x=y_pred, y=residuals,
        title="Residuals vs Predicted", template="plotly_white",
        labels={"x": "Predicted", "y": "Residuals"},
    )
    fig_resid.add_hline(y=0, line_dash="dash", line_color="red")
    fig_resid.update_layout(height=350, margin=dict(l=40, r=40, t=50, b=40))

    fig_actual = px.scatter(
        x=y_test, y=y_pred,
        title="Actual vs Predicted", template="plotly_white",
        labels={"x": "Actual", "y": "Predicted"},
    )
    mn, mx = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
    fig_actual.add_trace(go.Scatter(x=[mn, mx], y=[mn, mx], mode="lines",
                                     line=dict(dash="dash", color="red"), name="Perfect"))
    fig_actual.update_layout(height=350, margin=dict(l=40, r=40, t=50, b=40))

    fig_hist = px.histogram(
        x=residuals, title="Residual Distribution", template="plotly_white",
        labels={"x": "Residual", "y": "Count"},
    )
    fig_hist.update_layout(height=350, margin=dict(l=40, r=40, t=50, b=40))

    return {
        "residuals_vs_predicted": json.loads(fig_resid.to_json()),
        "actual_vs_predicted": json.loads(fig_actual.to_json()),
        "residual_distribution": json.loads(fig_hist.to_json()),
    }


def _diagnostic_classification(model, X_test, y_test, y_pred, y_proba) -> dict:
    plots = {}

    cm = confusion_matrix(y_test, y_pred)
    fig_cm = px.imshow(cm, text_auto=True, title="Confusion Matrix",
                       color_continuous_scale="Blues",
                       labels={"x": "Predicted", "y": "Actual"})
    fig_cm.update_layout(height=350, margin=dict(l=40, r=40, t=50, b=40))
    plots["confusion_matrix"] = json.loads(fig_cm.to_json())

    if y_proba is not None:
        n_classes = len(np.unique(y_test))
        if n_classes == 2:
            prob = y_proba[:, 1] if y_proba.ndim == 2 else y_proba
            fpr, tpr, _ = roc_curve(y_test, prob)
            fig_roc = go.Figure()
            fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name="ROC"))
            fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                                          line=dict(dash="dash", color="gray"), name="Random"))
            fig_roc.update_layout(title="ROC Curve", xaxis_title="FPR", yaxis_title="TPR",
                                   template="plotly_white", height=350,
                                   margin=dict(l=40, r=40, t=50, b=40))
            plots["roc_curve"] = json.loads(fig_roc.to_json())

            prec, rec, _ = precision_recall_curve(y_test, prob)
            fig_pr = go.Figure()
            fig_pr.add_trace(go.Scatter(x=rec, y=prec, mode="lines", name="PR"))
            fig_pr.update_layout(title="Precision-Recall Curve", xaxis_title="Recall",
                                  yaxis_title="Precision", template="plotly_white", height=350,
                                  margin=dict(l=40, r=40, t=50, b=40))
            plots["precision_recall"] = json.loads(fig_pr.to_json())

    return plots


def train_and_evaluate(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: list[str],
    task: str,
    model_configs: list[dict],
    test_size: float = 0.2,
) -> dict:
    X = df[feature_cols].values
    y = df[target_col].values

    if task == "classification":
        from sklearn.preprocessing import LabelEncoder
        if y.dtype == object or not np.issubdtype(y.dtype, np.number):
            le = LabelEncoder()
            y = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    results = {}
    for cfg in model_configs:
        name = cfg["name"]
        params = cfg.get("params", {})
        try:
            model = _build_model(name, task, params)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            if task == "regression":
                metrics = evaluate_regression(y_test, y_pred)
                diagnostics = _diagnostic_regression(model, X_test, y_test, y_pred)
            else:
                y_proba = None
                if hasattr(model, "predict_proba"):
                    try:
                        y_proba = model.predict_proba(X_test)
                    except Exception:
                        pass
                metrics = evaluate_classification(y_test, y_pred, y_proba)
                diagnostics = _diagnostic_classification(model, X_test, y_test, y_pred, y_proba)

            results[name] = {
                "status": "success",
                "metrics": metrics,
                "diagnostics": diagnostics,
                "params_used": params,
            }
        except Exception as e:
            results[name] = {"status": "error", "error": str(e)}

    return {
        "task": task,
        "target": target_col,
        "features": feature_cols,
        "test_size": test_size,
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "results": results,
    }


def get_available_models(task: str) -> list[str]:
    if task == "regression":
        return list(REGRESSION_MODELS.keys())
    return list(CLASSIFICATION_MODELS.keys())


def get_hyperparam_options(model_name: str) -> dict:
    return HYPERPARAM_OPTIONS.get(model_name, {})


def get_default_hyperparams(model_name: str) -> dict:
    return DEFAULT_HYPERPARAMS.get(model_name, {})
