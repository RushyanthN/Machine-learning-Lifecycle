# ML Prediction Platform

A guided machine learning pipeline: upload CSV data, understand it, explore with visualizations, engineer features, train multiple models with custom hyperparameters, and compare results side-by-side.

**Live Demo:** [ml-frontend-ky8d.onrender.com](https://ml-frontend-ky8d.onrender.com/)

> First load may take 30-60 seconds — Render free tier spins down after inactivity.

## Pipeline

1. **Upload** — CSV upload with auto type detection
2. **Understand** — head/tail, .info(), .describe(), dtypes
3. **EDA** — missing values, descriptive stats, value counts, interactive plots (histogram, boxplot, scatter, heatmap, etc.)
4. **Feature Engineering** — correlation, covariance, VIF (multicollinearity), feature importance, scaling, encoding, log/polynomial transforms, create/delete columns
5. **Modeling** — select multiple models, tune hyperparameters per model, train, compare metrics, diagnostic plots, best model highlight

## Models

**Regression:** Linear, Ridge, Lasso, Decision Tree, Random Forest, KNN, SVR, XGBoost

**Classification:** Logistic Regression, Decision Tree, Random Forest, KNN, SVC, XGBoost, Naive Bayes

Each model exposes tunable hyperparameters with sensible defaults.

## Diagnostic Plots

- **Regression:** Residuals vs Predicted, Actual vs Predicted, Residual Distribution
- **Classification:** Confusion Matrix, ROC Curve, Precision-Recall Curve

## Tech Stack

- **Backend:** FastAPI, scikit-learn, XGBoost, pandas, Plotly, statsmodels
- **Frontend:** Streamlit
- **Deployment:** Render (2 web services)

## Deployment

| Service | URL |
|---------|-----|
| Frontend | [ml-frontend-ky8d.onrender.com](https://ml-frontend-ky8d.onrender.com/) |
| Backend API | [ml-backend-xjt5.onrender.com](https://ml-backend-xjt5.onrender.com/) |

Deployed via `render.yaml` Blueprint — connects to GitHub and auto-deploys on push.

## Local Development

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
