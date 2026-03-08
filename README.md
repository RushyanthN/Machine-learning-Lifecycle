# ML Prediction Platform

Upload CSV data and walk through a guided machine learning pipeline: understand, explore, engineer features, train models with custom hyperparameters, and compare results.

## Pipeline

1. **Upload** — CSV upload with auto type detection
2. **Understand** — head/tail, .info(), .describe(), dtypes
3. **EDA** — missing values, descriptive stats, value counts, interactive plots
4. **Feature Engineering** — correlation, covariance, VIF, feature importance, scaling, encoding, log/polynomial transforms, create/delete columns
5. **Modeling** — select multiple models, tune hyperparameters, train, compare metrics, diagnostic plots

## Quick Start

```bash
pip install -r requirements.txt
```

Terminal 1 — Backend:
```bash
uvicorn backend.main:app --reload --port 8001
```

Terminal 2 — Frontend:
```bash
streamlit run frontend/app.py
```

Open http://localhost:8501

## Models

**Regression:** Linear, Ridge, Lasso, Decision Tree, Random Forest, KNN, SVR, XGBoost

**Classification:** Logistic Regression, Decision Tree, Random Forest, KNN, SVC, XGBoost, Naive Bayes

Each model exposes tunable hyperparameters with sensible defaults.

## Diagnostic Plots

- **Regression:** Residuals vs Predicted, Actual vs Predicted, Residual Distribution
- **Classification:** Confusion Matrix, ROC Curve, Precision-Recall Curve
