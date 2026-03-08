import json
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from backend.services.cleaner import detect_column_types


def recommend_plots(df: pd.DataFrame) -> list[dict]:
    col_types = detect_column_types(df)
    recommendations = []

    for col in col_types["numeric"]:
        recommendations.append({
            "type": "histogram",
            "columns": [col],
            "title": f"Distribution of {col}",
        })
        recommendations.append({
            "type": "boxplot",
            "columns": [col],
            "title": f"Boxplot of {col}",
        })

    numeric_cols = col_types["numeric"]
    if len(numeric_cols) >= 2:
        recommendations.append({
            "type": "heatmap",
            "columns": numeric_cols,
            "title": "Correlation Heatmap",
        })
        for i in range(min(len(numeric_cols), 3)):
            for j in range(i + 1, min(len(numeric_cols), 4)):
                recommendations.append({
                    "type": "scatter",
                    "columns": [numeric_cols[i], numeric_cols[j]],
                    "title": f"{numeric_cols[i]} vs {numeric_cols[j]}",
                })

    for cat_col in col_types["categorical"][:3]:
        for num_col in numeric_cols[:3]:
            recommendations.append({
                "type": "bar",
                "columns": [cat_col, num_col],
                "title": f"Mean {num_col} by {cat_col}",
            })
            recommendations.append({
                "type": "boxplot_grouped",
                "columns": [cat_col, num_col],
                "title": f"{num_col} by {cat_col}",
            })

    for dt_col in col_types["datetime"]:
        for num_col in numeric_cols[:3]:
            recommendations.append({
                "type": "line",
                "columns": [dt_col, num_col],
                "title": f"{num_col} over {dt_col}",
            })

    return recommendations


def generate_plot(df: pd.DataFrame, plot_type: str, columns: list[str], title: str = "") -> dict:
    fig = None

    if plot_type == "histogram":
        fig = px.histogram(df, x=columns[0], title=title, template="plotly_white")

    elif plot_type == "boxplot":
        fig = px.box(df, y=columns[0], title=title, template="plotly_white")

    elif plot_type == "boxplot_grouped":
        fig = px.box(df, x=columns[0], y=columns[1], title=title, template="plotly_white",
                     color=columns[0])

    elif plot_type == "scatter":
        fig = px.scatter(df, x=columns[0], y=columns[1], title=title, template="plotly_white",
                         opacity=0.6)

    elif plot_type == "line":
        try:
            temp_df = df.copy()
            temp_df[columns[0]] = pd.to_datetime(temp_df[columns[0]])
            temp_df = temp_df.sort_values(columns[0])
            fig = px.line(temp_df, x=columns[0], y=columns[1], title=title, template="plotly_white")
        except Exception:
            fig = px.line(df, x=columns[0], y=columns[1], title=title, template="plotly_white")

    elif plot_type == "bar":
        agg_df = df.groupby(columns[0])[columns[1]].mean().reset_index()
        agg_df = agg_df.sort_values(columns[1], ascending=False).head(20)
        fig = px.bar(agg_df, x=columns[0], y=columns[1], title=title, template="plotly_white")

    elif plot_type == "heatmap":
        corr = df[columns].corr()
        fig = go.Figure(data=go.Heatmap(
            z=corr.values,
            x=corr.columns.tolist(),
            y=corr.index.tolist(),
            colorscale="RdBu_r",
            zmin=-1, zmax=1,
            text=np.round(corr.values, 2),
            texttemplate="%{text}",
        ))
        fig.update_layout(title=title, template="plotly_white")

    if fig is None:
        return {}

    fig.update_layout(margin=dict(l=40, r=40, t=50, b=40))
    return json.loads(fig.to_json())
