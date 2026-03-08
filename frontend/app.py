import os
import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

API = os.environ.get("API_URL", "http://localhost:8001/api")

st.set_page_config(page_title="ML Prediction Platform", page_icon="📊", layout="wide")

# ── State ─────────────────────────────────────────────────────────────────────

STEPS = ["Upload", "Understand", "EDA", "Feature Engineering", "Modeling"]

for key, default in [("step", 0), ("sid", None)]:
    if key not in st.session_state:
        st.session_state[key] = default

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("📊 ML Platform")
    st.caption("Guided machine learning pipeline")
    st.markdown("---")

    for i, name in enumerate(STEPS):
        disabled = i > 0 and st.session_state.sid is None
        marker = "✓" if i < st.session_state.step else "→" if i == st.session_state.step else " "
        if st.button(f"{marker}  {i+1}. {name}", key=f"nav{i}", disabled=disabled, use_container_width=True):
            st.session_state.step = i

    st.markdown("---")
    if st.session_state.sid:
        st.success(f"Session `{st.session_state.sid[:8]}…`")
    if st.button("Reset Session", use_container_width=True):
        st.session_state.clear()
        st.rerun()

# ── Helpers ───────────────────────────────────────────────────────────────────

def api(method, path, **kw):
    fn = requests.post if method == "post" else requests.get
    r = fn(f"{API}{path}", **kw)
    r.raise_for_status()
    return r.json()

def nav_buttons(back=None, forward=None, fwd_label="Continue →"):
    cols = st.columns([1, 4, 1])
    if back is not None:
        with cols[0]:
            if st.button("← Back"):
                st.session_state.step = back
                st.rerun()
    if forward is not None:
        with cols[2]:
            if st.button(fwd_label):
                st.session_state.step = forward
                st.rerun()

# ═════════════════════════════════════════════════════════════════════════════
# STEP 1: UPLOAD
# ═════════════════════════════════════════════════════════════════════════════

if st.session_state.step == 0:
    st.header("Step 1 — Upload Data")
    st.write("Upload a CSV file to start your analysis.")

    file = st.file_uploader("Choose CSV", type=["csv"])
    if file and st.button("Upload", type="primary"):
        with st.spinner("Uploading…"):
            r = api("post", "/upload", files={"file": (file.name, file.getvalue(), "text/csv")})
        st.session_state.sid = r["session_id"]
        st.session_state.filename = r["filename"]
        st.success(f"Uploaded **{r['filename']}** — {r['summary']['shape'][0]} rows × {r['summary']['shape'][1]} columns")
        st.session_state.step = 1
        st.rerun()

# ═════════════════════════════════════════════════════════════════════════════
# STEP 2: UNDERSTAND THE DATA
# ═════════════════════════════════════════════════════════════════════════════

elif st.session_state.step == 1:
    st.header("Step 2 — Understand Your Data")
    sid = st.session_state.sid

    overview = api("post", "/eda/overview", json={"session_id": sid})
    info_data = api("post", "/eda/info", json={"session_id": sid})
    desc_data = api("post", "/eda/describe", json={"session_id": sid})

    # .shape
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", f"{overview['shape'][0]:,}")
    c2.metric("Columns", overview['shape'][1])
    c3.metric("File", st.session_state.get("filename", "—"))

    tab_head, tab_info, tab_desc, tab_dtypes = st.tabs(["Head / Tail", "Info", "Describe", "Dtypes"])

    with tab_head:
        st.subheader("First 5 Rows")
        st.dataframe(pd.DataFrame(overview["head"]), use_container_width=True, hide_index=True)
        st.subheader("Last 5 Rows")
        st.dataframe(pd.DataFrame(overview["tail"]), use_container_width=True, hide_index=True)

    with tab_info:
        st.subheader("Column Information")
        info_df = pd.DataFrame(info_data["info"])
        st.dataframe(info_df, use_container_width=True, hide_index=True)

    with tab_desc:
        st.subheader("Descriptive Statistics (Numeric)")
        if "numeric" in desc_data:
            st.dataframe(pd.DataFrame(desc_data["numeric"]).round(4), use_container_width=True)
        if "categorical" in desc_data:
            st.subheader("Descriptive Statistics (Categorical)")
            st.dataframe(pd.DataFrame(desc_data["categorical"]), use_container_width=True)

    with tab_dtypes:
        st.subheader("Data Types")
        dtype_df = pd.DataFrame([
            {"Column": col, "Type": dtype}
            for col, dtype in overview["dtypes"].items()
        ])
        st.dataframe(dtype_df, use_container_width=True, hide_index=True)

    nav_buttons(back=0, forward=2)

# ═════════════════════════════════════════════════════════════════════════════
# STEP 3: EXPLORATORY DATA ANALYSIS
# ═════════════════════════════════════════════════════════════════════════════

elif st.session_state.step == 2:
    st.header("Step 3 — Exploratory Data Analysis")
    sid = st.session_state.sid

    tab_nulls, tab_stats, tab_viz = st.tabs(["Missing Values", "Statistics", "Visualizations"])

    # ── Missing Values ─────────────────────────────────────────────────────
    with tab_nulls:
        nulls = api("post", "/eda/nulls", json={"session_id": sid})

        c1, c2, c3 = st.columns(3)
        c1.metric("Total Missing", nulls["total_nulls"])
        c2.metric("Total Cells", f"{nulls['total_cells']:,}")
        c3.metric("Missing %", f"{nulls['null_pct_overall']}%")

        if nulls["per_column"]:
            null_df = pd.DataFrame([
                {"Column": col, "Missing": d["count"], "Percent": f"{d['pct']}%"}
                for col, d in nulls["per_column"].items()
            ]).sort_values("Missing", ascending=False)
            st.dataframe(null_df, use_container_width=True, hide_index=True)

            fig = px.bar(null_df, x="Column", y="Missing", title="Missing Values by Column",
                         template="plotly_white", color="Missing", color_continuous_scale="Reds")
            fig.update_layout(height=350, margin=dict(t=40, b=20))
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Handle Missing Values")
            strategy = st.selectbox("Strategy", ["drop", "mean", "median", "mode", "ffill", "bfill"])
            remove_dup = st.checkbox("Also remove duplicates", value=True)
            if st.button("Apply Cleaning", type="primary"):
                with st.spinner("Cleaning…"):
                    r = api("post", "/clean", data={"session_id": sid, "missing_strategy": strategy, "remove_duplicates": remove_dup})
                st.success("Cleaned!")
                for action in r["report"]["actions"]:
                    st.write(f"• {action}")
        else:
            st.success("No missing values found!")

    # ── Descriptive Statistics ─────────────────────────────────────────────
    with tab_stats:
        desc = api("post", "/eda/describe", json={"session_id": sid})
        if "numeric" in desc:
            st.subheader("Numeric Summary")
            st.dataframe(pd.DataFrame(desc["numeric"]).round(4), use_container_width=True)
        if "categorical" in desc:
            st.subheader("Categorical Summary")
            st.dataframe(pd.DataFrame(desc["categorical"]), use_container_width=True)

        overview = api("post", "/eda/overview", json={"session_id": sid})
        all_cols = overview["columns"]
        vc_col = st.selectbox("Inspect value counts for", all_cols)
        if vc_col:
            vc = api("post", "/eda/value_counts", json={"session_id": sid, "column": vc_col})
            vc_df = pd.DataFrame([{"Value": k, "Count": v} for k, v in vc["counts"].items()])
            c1, c2 = st.columns([1, 2])
            with c1:
                st.dataframe(vc_df, use_container_width=True, hide_index=True)
            with c2:
                fig = px.bar(vc_df.head(15), x="Value", y="Count", template="plotly_white",
                             title=f"Top Values — {vc_col}")
                fig.update_layout(height=350, margin=dict(t=40, b=20))
                st.plotly_chart(fig, use_container_width=True)

    # ── Visualizations ─────────────────────────────────────────────────────
    with tab_viz:
        overview = api("post", "/eda/overview", json={"session_id": sid})
        num_cols = [c for c, t in overview["dtypes"].items() if t.startswith(("int", "float"))]
        all_cols = overview["columns"]

        plot_type = st.selectbox("Chart Type", ["histogram", "boxplot", "scatter", "line", "bar", "heatmap", "boxplot_grouped"])

        if plot_type in ("histogram", "boxplot"):
            col = st.selectbox("Column", num_cols if num_cols else all_cols)
            cols = [col]
        elif plot_type == "heatmap":
            cols = st.multiselect("Numeric Columns", num_cols, default=num_cols[:6])
        else:
            c1, c2 = st.columns(2)
            with c1:
                xc = st.selectbox("X", all_cols, key="viz_x")
            with c2:
                yc = st.selectbox("Y", all_cols, index=min(1, len(all_cols)-1), key="viz_y")
            cols = [xc, yc]

        if st.button("Generate", type="primary"):
            with st.spinner("Rendering…"):
                r = api("post", "/visualize/plot", json={
                    "session_id": sid, "plot_type": plot_type, "columns": cols, "title": f"{plot_type.title()} Plot"
                })
            if r.get("plot"):
                st.plotly_chart(go.Figure(r["plot"]), use_container_width=True)

    nav_buttons(back=1, forward=3)

# ═════════════════════════════════════════════════════════════════════════════
# STEP 4: FEATURE ENGINEERING
# ═════════════════════════════════════════════════════════════════════════════

elif st.session_state.step == 3:
    st.header("Step 4 — Feature Engineering")
    sid = st.session_state.sid

    preview = api("post", "/transform/preview", json={"session_id": sid})
    num_cols = preview["numeric_columns"]
    cat_cols = preview["categorical_columns"]

    st.write(f"**{len(num_cols)}** numeric · **{len(cat_cols)}** categorical columns  ·  Shape: {preview['shape']}")

    tab_corr, tab_cov, tab_vif, tab_imp, tab_xform, tab_manage = st.tabs([
        "Correlation", "Covariance", "Multicollinearity", "Importance",
        "Transformations", "Manage Columns"
    ])

    # ── Correlation ────────────────────────────────────────────────────────
    with tab_corr:
        if st.button("Compute Correlation", key="btn_corr"):
            with st.spinner("Computing…"):
                r = api("post", "/features/correlation", json={"session_id": sid})
            if "plot" in r:
                st.plotly_chart(go.Figure(r["plot"]), use_container_width=True)
            if "error" in r:
                st.warning(r["error"])

    # ── Covariance ─────────────────────────────────────────────────────────
    with tab_cov:
        if st.button("Compute Covariance", key="btn_cov"):
            with st.spinner("Computing…"):
                r = api("post", "/features/covariance", json={"session_id": sid})
            if "plot" in r:
                st.plotly_chart(go.Figure(r["plot"]), use_container_width=True)
            if "error" in r:
                st.warning(r["error"])

    # ── VIF ────────────────────────────────────────────────────────────────
    with tab_vif:
        st.write("Variance Inflation Factor measures multicollinearity. VIF > 10 is concerning.")
        if st.button("Compute VIF", key="btn_vif"):
            with st.spinner("Computing…"):
                r = api("post", "/features/vif", json={"session_id": sid})
            if "vif" in r:
                vif_df = pd.DataFrame(r["vif"])
                st.dataframe(vif_df, use_container_width=True, hide_index=True)

                high = vif_df[vif_df["status"] == "High"]
                if not high.empty:
                    st.warning(f"{len(high)} feature(s) have high multicollinearity (VIF > 10). Consider removing them.")
            if "error" in r:
                st.warning(r["error"])

    # ── Feature Importance ─────────────────────────────────────────────────
    with tab_imp:
        target = st.selectbox("Target column", num_cols + cat_cols, key="imp_target")
        task_type = st.radio("Task type", ["regression", "classification"], horizontal=True, key="imp_task")
        if st.button("Compute Importance", key="btn_imp"):
            with st.spinner("Training Random Forest…"):
                r = api("post", "/features/importance", json={"session_id": sid, "target_col": target, "task": task_type})
            if "plot" in r:
                st.plotly_chart(go.Figure(r["plot"]), use_container_width=True)
            if "importances" in r:
                st.dataframe(pd.DataFrame(r["importances"]), use_container_width=True, hide_index=True)
            if "error" in r:
                st.warning(r["error"])

    # ── Transformations ────────────────────────────────────────────────────
    with tab_xform:
        xform_type = st.selectbox("Transform", ["Scaling", "Encoding", "Log Transform", "Polynomial Features"])

        if xform_type == "Scaling":
            sel = st.multiselect("Columns", num_cols, key="xf_scale_cols")
            method = st.selectbox("Method", ["standard", "minmax", "robust"], key="xf_scale_m")
            if st.button("Apply", key="xf_scale_btn") and sel:
                r = api("post", "/transform/scale", json={"session_id": sid, "columns": sel, "method": method})
                st.success("Scaling applied.")

        elif xform_type == "Encoding":
            sel = st.multiselect("Columns", cat_cols, key="xf_enc_cols")
            method = st.selectbox("Method", ["onehot", "label"], key="xf_enc_m")
            if st.button("Apply", key="xf_enc_btn") and sel:
                r = api("post", "/transform/encode", json={"session_id": sid, "columns": sel, "method": method})
                st.success("Encoding applied.")

        elif xform_type == "Log Transform":
            sel = st.multiselect("Columns", num_cols, key="xf_log_cols")
            if st.button("Apply", key="xf_log_btn") and sel:
                r = api("post", "/transform/log", json={"session_id": sid, "columns": sel})
                st.success("Log transform applied.")

        elif xform_type == "Polynomial Features":
            sel = st.multiselect("Columns", num_cols, key="xf_poly_cols")
            deg = st.slider("Degree", 2, 4, 2, key="xf_poly_deg")
            if st.button("Apply", key="xf_poly_btn") and sel:
                r = api("post", "/transform/polynomial", json={"session_id": sid, "columns": sel, "degree": deg})
                st.success("Polynomial features created.")

    # ── Manage Columns ─────────────────────────────────────────────────────
    with tab_manage:
        st.subheader("Create New Column")
        new_name = st.text_input("Column name", key="new_col_name")
        new_expr = st.text_input("Expression (e.g. `col_a + col_b * 2`)", key="new_col_expr")
        if st.button("Create", key="create_col_btn") and new_name and new_expr:
            try:
                r = api("post", "/features/create", json={"session_id": sid, "name": new_name, "expression": new_expr})
                st.success(f"Column '{new_name}' created.")
            except Exception as e:
                st.error(f"Invalid expression: {e}")

        st.markdown("---")
        st.subheader("Delete Columns")
        all_c = num_cols + cat_cols
        del_cols = st.multiselect("Select columns to remove", all_c, key="del_cols")
        if st.button("Delete Selected", key="del_col_btn") and del_cols:
            r = api("post", "/features/delete", json={"session_id": sid, "columns": del_cols})
            st.success(f"Removed {len(del_cols)} column(s).")
            st.rerun()

    nav_buttons(back=2, forward=4, fwd_label="Proceed to Modeling →")

# ═════════════════════════════════════════════════════════════════════════════
# STEP 5: MODELING
# ═════════════════════════════════════════════════════════════════════════════

elif st.session_state.step == 4:
    st.header("Step 5 — Model Training & Comparison")
    sid = st.session_state.sid

    preview = api("post", "/transform/preview", json={"session_id": sid})
    all_cols = preview["numeric_columns"] + preview["categorical_columns"]

    # ── Config ─────────────────────────────────────────────────────────────
    st.subheader("Configuration")
    c1, c2 = st.columns(2)
    with c1:
        task = st.radio("Task", ["regression", "classification"], horizontal=True)
    with c2:
        test_pct = st.slider("Test split %", 10, 40, 20, 5)

    target = st.selectbox("Target Variable", all_cols)
    features = st.multiselect(
        "Feature Variables",
        [c for c in all_cols if c != target],
        default=[c for c in preview["numeric_columns"] if c != target][:8],
    )

    models_list = api("get", "/predict/models", params={"task": task})["models"]
    selected = st.multiselect("Select Models", models_list, default=models_list)

    # ── Per-model hyperparameters ──────────────────────────────────────────
    st.subheader("Hyperparameters")
    st.caption("Defaults are pre-filled. Expand a model to customize.")

    model_configs = []
    for model_name in selected:
        hp = api("get", "/predict/hyperparams", params={"model_name": model_name})
        options = hp.get("options", {})
        defaults = hp.get("defaults", {})
        params = {}

        if options:
            with st.expander(f"⚙ {model_name}", expanded=False):
                for pname, spec in options.items():
                    ptype = spec["type"]
                    default_val = defaults.get(pname, spec.get("default"))

                    if ptype == "int":
                        params[pname] = st.slider(
                            pname, spec["min"], spec["max"],
                            value=int(default_val) if default_val is not None else spec["min"],
                            key=f"hp_{model_name}_{pname}"
                        )
                    elif ptype == "float":
                        params[pname] = st.number_input(
                            pname, min_value=float(spec["min"]), max_value=float(spec["max"]),
                            value=float(default_val), step=0.01,
                            key=f"hp_{model_name}_{pname}"
                        )
                    elif ptype == "select":
                        idx = spec["options"].index(default_val) if default_val in spec["options"] else 0
                        params[pname] = st.selectbox(
                            pname, spec["options"], index=idx,
                            key=f"hp_{model_name}_{pname}"
                        )
                    elif ptype == "int_or_none":
                        use_none = st.checkbox(f"{pname} = None (unlimited)", value=default_val is None,
                                               key=f"hp_{model_name}_{pname}_none")
                        if use_none:
                            params[pname] = None
                        else:
                            params[pname] = st.slider(
                                pname, spec["min"], spec["max"],
                                value=int(default_val) if default_val is not None else spec["min"],
                                key=f"hp_{model_name}_{pname}"
                            )
        else:
            st.caption(f"**{model_name}** — no tunable hyperparameters")

        model_configs.append({"name": model_name, "params": params})

    # ── Train ──────────────────────────────────────────────────────────────
    st.markdown("---")
    if st.button("Train All Models", type="primary", use_container_width=True) and features and selected:
        with st.spinner("Training… this may take a moment."):
            r = api("post", "/predict/train", json={
                "session_id": sid,
                "target_col": target,
                "feature_cols": features,
                "task": task,
                "model_configs": model_configs,
                "test_size": test_pct / 100,
            })

        results = r["results"]
        ok = {k: v for k, v in results.items() if v["status"] == "success"}
        fail = {k: v for k, v in results.items() if v["status"] == "error"}

        st.success(f"Trained **{len(ok)}** models  ·  Train {r['train_samples']}  ·  Test {r['test_samples']}")

        if ok:
            # ── Comparison Table ───────────────────────────────
            st.subheader("Results Comparison")
            if task == "regression":
                mkeys = ["MAE", "MSE", "RMSE", "R2"]
                best_key = "R2"
            else:
                mkeys = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
                best_key = "F1"

            rows = []
            for mn, rv in ok.items():
                row = {"Model": mn}
                for mk in mkeys:
                    v = rv["metrics"].get(mk)
                    row[mk] = f"{v:.4f}" if isinstance(v, (int, float)) else v
                rows.append(row)
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

            # ── Comparison Charts ──────────────────────────────
            chart_keys = [m for m in mkeys if m not in ("ROC-AUC",)]
            cols = st.columns(min(len(chart_keys), 2))
            for idx, mk in enumerate(chart_keys):
                vals, names = [], []
                for mn, rv in ok.items():
                    v = rv["metrics"].get(mk)
                    if isinstance(v, (int, float)):
                        vals.append(v)
                        names.append(mn)
                if vals:
                    fig = px.bar(x=names, y=vals, title=mk, labels={"x":"","y":mk},
                                 template="plotly_white", color=vals, color_continuous_scale="Blues")
                    fig.update_layout(showlegend=False, height=320, margin=dict(t=40, b=20))
                    with cols[idx % 2]:
                        st.plotly_chart(fig, use_container_width=True)

            # ── Best Model ─────────────────────────────────────
            best = max(ok.items(), key=lambda x: x[1]["metrics"].get(best_key, -999))
            st.markdown("---")
            st.subheader("Best Model")
            c1, c2 = st.columns(2)
            c1.metric("Model", best[0])
            c2.metric(best_key, f"{best[1]['metrics'][best_key]:.4f}")

            # ── Diagnostic Plots for Best Model ────────────────
            st.subheader(f"Diagnostic Plots — {best[0]}")
            diag = best[1].get("diagnostics", {})
            diag_cols = st.columns(min(len(diag), 2))
            for idx, (pname, pdata) in enumerate(diag.items()):
                with diag_cols[idx % len(diag_cols)]:
                    st.plotly_chart(go.Figure(pdata), use_container_width=True)

            # ── Per-model diagnostics ──────────────────────────
            st.markdown("---")
            st.subheader("Individual Model Diagnostics")
            model_pick = st.selectbox("Select model", list(ok.keys()))
            if model_pick:
                diag = ok[model_pick].get("diagnostics", {})
                d_cols = st.columns(min(len(diag), 2)) if diag else []
                for idx, (pname, pdata) in enumerate(diag.items()):
                    with d_cols[idx % len(d_cols)]:
                        st.plotly_chart(go.Figure(pdata), use_container_width=True)

                st.caption(f"Hyperparameters used: `{ok[model_pick].get('params_used', {})}`")

        if fail:
            st.warning("Some models failed:")
            for mn, rv in fail.items():
                st.error(f"**{mn}**: {rv['error']}")

    nav_buttons(back=3)
