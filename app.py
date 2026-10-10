"""
app.py
------
EnergySense AI — Smart Energy Consumption Prediction & Sustainability Dashboard

Run with:
    streamlit run app.py
"""

import os
import sys
import json
import warnings

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Path setup — must come before local imports
# ---------------------------------------------------------------------------
APP_DIR = os.path.dirname(os.path.abspath(__file__))
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

from src.preprocessing import load_and_clean, validate_dataset, DATA_PATH
from src.feature_engineering import (
    engineer_features,
    build_X_y,
    get_feature_columns,
    LAG_STEPS,
    ROLLING_WINDOWS,
)
from src.explain import get_feature_importance

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="EnergySense AI",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .kpi-card {
        background: #f7f8fa;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 16px 20px;
        text-align: center;
        margin-bottom: 8px;
    }
    .kpi-label { font-size: 12px; color: #57606a; margin-bottom: 4px; }
    .kpi-value { font-size: 26px; font-weight: 700; color: #1f2328; }
    .kpi-unit  { font-size: 12px; color: #57606a; }

    .sdg-card {
        background: #f7f8fa;
        border-left: 4px solid #3b82d4;
        border-radius: 6px;
        padding: 14px 18px;
        margin-bottom: 12px;
    }
    .info-box {
        background: #eef2ff;
        border: 1px solid #c7d2fe;
        border-radius: 6px;
        padding: 12px 16px;
        font-size: 13px;
        color: #1f2328;
        margin-bottom: 8px;
    }
    .warn-box {
        background: #fff7ed;
        border: 1px solid #fed7aa;
        border-radius: 6px;
        padding: 12px 16px;
        font-size: 13px;
        color: #1f2328;
        margin-bottom: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Loading dataset …")
def load_data():
    """Load, clean, and engineer features — cached for the session."""
    df_raw = load_and_clean(DATA_PATH)
    df = engineer_features(df_raw, include_lags=True)
    return df_raw, df


@st.cache_resource(show_spinner="Loading model …")
def load_model():
    """Load the saved best model — cached across reruns."""
    import joblib
    model_path = os.path.join("models", "best_model.joblib")
    if not os.path.exists(model_path):
        return None
    return joblib.load(model_path)


@st.cache_data
def load_metrics():
    path = os.path.join("models", "model_metrics.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


@st.cache_data
def load_feature_columns():
    path = os.path.join("models", "feature_columns.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)["feature_columns"]


# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
PAGES = [
    "🏠 Home",
    "📊 Data Explorer",
    "📈 Consumption Analysis",
    "🤖 ML Prediction",
    "🔍 Model Performance",
    "💡 What-If Simulator",
    "⚠️ Energy Insights",
    "🌍 Sustainability & SDGs",
    "ℹ️ About",
]

with st.sidebar:
    st.markdown("## ⚡ EnergySense AI")
    st.markdown("---")
    page = st.radio("Navigation", PAGES, label_visibility="collapsed")
    st.markdown("---")
    st.markdown(
        "<small style='color:#57606a'>UCI Appliances Energy Prediction<br>"
        "Candanedo, L. (2017)</small>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Helper: KPI card HTML
# ---------------------------------------------------------------------------
def kpi_card(label, value, unit=""):
    return (
        f'<div class="kpi-card">'
        f'<div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{value}</div>'
        f'<div class="kpi-unit">{unit}</div>'
        f"</div>"
    )


# ===========================================================================
# PAGE: HOME
# ===========================================================================
def page_home(df_raw, df_eng, metrics, feature_cols):
    st.markdown("# ⚡ EnergySense AI")
    st.markdown(
        "### Smart Energy Consumption Prediction & Sustainability Dashboard"
    )
    st.markdown("---")

    # KPI row 1
    best = metrics["best_model"]
    best_m = next(m for m in metrics["models"] if m["model"] == best)
    avg_wh = int(df_raw["Appliances"].mean())
    max_wh = int(df_raw["Appliances"].max())

    cols = st.columns(4)
    with cols[0]:
        st.markdown(
            kpi_card("Total Records", f"{len(df_raw):,}", "observations"),
            unsafe_allow_html=True,
        )
    with cols[1]:
        st.markdown(
            kpi_card("Model Features", str(len(feature_cols)), "input variables"),
            unsafe_allow_html=True,
        )
    with cols[2]:
        st.markdown(
            kpi_card("Avg Appliance Load", f"{avg_wh}", "Wh per 10 min"),
            unsafe_allow_html=True,
        )
    with cols[3]:
        st.markdown(
            kpi_card("Peak Appliance Load", f"{max_wh}", "Wh per 10 min"),
            unsafe_allow_html=True,
        )

    # KPI row 2
    cols2 = st.columns(3)
    with cols2[0]:
        st.markdown(
            kpi_card("Best Model", best, ""),
            unsafe_allow_html=True,
        )
    with cols2[1]:
        st.markdown(
            kpi_card("Best Model RMSE", f"{best_m['RMSE']:.2f}", "Wh (test set)"),
            unsafe_allow_html=True,
        )
    with cols2[2]:
        st.markdown(
            kpi_card("Best Model R²", f"{best_m['R2']:.4f}", "on test set"),
            unsafe_allow_html=True,
        )

    st.markdown("---")

    c1, c2 = st.columns([3, 2])
    with c1:
        st.markdown("### Problem Statement")
        st.info(
            "**How can machine learning be used to predict household appliance energy "
            "consumption from indoor environmental conditions, weather conditions, "
            "lighting, and time-related factors, while providing useful insights that "
            "encourage energy efficiency and responsible consumption?**"
        )

        st.markdown("### Project Objectives")
        st.markdown(
            """
- Predict appliance energy consumption (Wh per 10-minute interval) using environmental and temporal features.
- Provide interactive tools for exploring energy consumption patterns.
- Enable scenario simulation to understand how model predictions respond to changed inputs.
- Present sustainability insights aligned with SDG 7, 11, 12, and 13.
            """
        )

        st.markdown("### Machine Learning Approach")
        st.markdown(
            """
Three regression models are compared using a **chronological train/test split**
(earliest 80% → training, latest 20% → test). This avoids temporal data leakage
and produces a more realistic evaluation of prediction performance.

**Lag features** (past appliance consumption values at 10-min, 1-hr, and 24-hr
intervals) are included because the dataset has strong autocorrelation
(Lag-1 correlation ≈ 0.75). The best model is selected by lowest test-set RMSE.
            """
        )

    with c2:
        st.markdown("### Dataset Information")
        st.markdown(
            f"""
| Property | Value |
|---|---|
| Source | UCI ML Repository |
| Records | {len(df_raw):,} |
| Columns | {df_raw.shape[1]} |
| Frequency | 10-minute intervals |
| Period | Jan – May 2016 |
| Target | Appliances (Wh) |
| Missing values | None |
| Duplicate rows | None |
            """
        )

        # Mini time-series preview
        sample = df_raw.set_index("date")["Appliances"].resample("D").mean().reset_index()
        fig = px.line(
            sample, x="date", y="Appliances",
            title="Daily Average Appliance Consumption",
            labels={"Appliances": "Wh (mean)", "date": ""},
        )
        fig.update_layout(height=240, margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig, use_container_width=True)


# ===========================================================================
# PAGE: DATA EXPLORER
# ===========================================================================
def page_data_explorer(df_raw):
    st.markdown("## 📊 Data Explorer")
    st.markdown("Explore the raw dataset, descriptive statistics, and column information.")
    st.markdown("---")

    # Validation report
    with st.expander("Dataset Validation Report", expanded=False):
        try:
            report = validate_dataset()
            col1, col2, col3 = st.columns(3)
            col1.metric("Rows", f"{report['rows']:,}")
            col2.metric("Columns", report["columns"])
            col3.metric("Missing Values", report["total_missing_values"])
            col1.metric("Duplicate Rows", report["duplicate_rows"])
            col2.metric("Date Range Start", report["date_range_start"][:10])
            col3.metric("Date Range End", report["date_range_end"][:10])
            st.success("All validation checks passed. Dataset is clean and complete.")
        except Exception as e:
            st.error(f"Validation error: {e}")

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Sample Data", "Statistics", "Column Info", "Distributions"]
    )

    with tab1:
        st.markdown("#### Sample Records")
        cols_sel = st.multiselect(
            "Select columns to display",
            options=df_raw.columns.tolist(),
            default=["date", "Appliances", "lights", "T1", "RH_1", "T_out"],
        )
        n_rows = st.slider("Rows to display", 5, 200, 20, key="explorer_rows")
        filter_col = st.selectbox("Filter by column (optional)", ["— None —"] + df_raw.columns.tolist(), key="filter_col")
        df_display = df_raw[cols_sel] if cols_sel else df_raw
        if filter_col != "— None —" and pd.api.types.is_numeric_dtype(df_raw[filter_col]):
            lo = float(df_raw[filter_col].min())
            hi = float(df_raw[filter_col].max())
            rng = st.slider(f"{filter_col} range", lo, hi, (lo, hi), key="filter_rng")
            mask = df_raw[filter_col].between(rng[0], rng[1])
            df_display = df_display[mask]
        st.dataframe(df_display.head(n_rows), use_container_width=True)
        st.caption(f"Showing {min(n_rows, len(df_display))} of {len(df_display):,} matching rows.")

    with tab2:
        st.markdown("#### Descriptive Statistics")
        numeric_cols = df_raw.select_dtypes(include="number").columns.tolist()
        stat_cols = st.multiselect("Select columns", numeric_cols, default=["Appliances", "lights", "T1", "RH_1", "T_out", "Windspeed"])
        if stat_cols:
            st.dataframe(df_raw[stat_cols].describe().T.round(3), use_container_width=True)
        else:
            st.info("Select at least one column.")

    with tab3:
        st.markdown("#### Column Information")
        col_info = pd.DataFrame({
            "Column": df_raw.columns,
            "Type": [str(t) for t in df_raw.dtypes],
            "Non-Null": df_raw.notnull().sum().values,
            "Null": df_raw.isnull().sum().values,
            "Unique": [df_raw[c].nunique() for c in df_raw.columns],
            "Min": [df_raw[c].min() if pd.api.types.is_numeric_dtype(df_raw[c]) else "—" for c in df_raw.columns],
            "Max": [df_raw[c].max() if pd.api.types.is_numeric_dtype(df_raw[c]) else "—" for c in df_raw.columns],
        })
        st.dataframe(col_info, use_container_width=True, hide_index=True)

    with tab4:
        st.markdown("#### Column Distribution")
        dist_col = st.selectbox("Select column", df_raw.select_dtypes(include="number").columns.tolist(), key="dist_col")
        fig = px.histogram(
            df_raw, x=dist_col, nbins=60,
            title=f"Distribution of {dist_col}",
            color_discrete_sequence=["#3b82d4"],
        )
        fig.update_layout(height=380)
        st.plotly_chart(fig, use_container_width=True)


# ===========================================================================
# PAGE: CONSUMPTION ANALYSIS
# ===========================================================================
def page_consumption_analysis(df_raw, df_eng):
    st.markdown("## 📈 Consumption Analysis")
    st.markdown("Interactive charts exploring appliance energy consumption patterns.")
    st.markdown("---")

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["Time Series", "Hourly & Weekly", "Environmental", "Lights vs Appliances", "Monthly"]
    )

    with tab1:
        st.markdown("#### Appliance Consumption Over Time")
        resample_opt = st.select_slider(
            "Aggregation period",
            options=["10min", "1H", "6H", "D", "W"],
            value="D",
        )
        ts = df_raw.set_index("date")["Appliances"].resample(resample_opt).mean().reset_index()
        fig = px.line(ts, x="date", y="Appliances",
                      title=f"Appliance Consumption — {resample_opt} average",
                      labels={"Appliances": "Wh", "date": "Date"},
                      color_discrete_sequence=["#3b82d4"])
        fig.update_layout(height=380)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        c1, c2 = st.columns(2)
        with c1:
            hourly = df_eng.groupby("hour")["Appliances"].mean().reset_index()
            fig = px.bar(hourly, x="hour", y="Appliances",
                         title="Avg Consumption by Hour of Day",
                         labels={"Appliances": "Avg Wh", "hour": "Hour"},
                         color="Appliances", color_continuous_scale="Blues")
            fig.update_layout(height=340, coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            dow_map = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"}
            dow_df = df_eng.groupby("day_of_week")["Appliances"].mean().reset_index()
            dow_df["day_name"] = dow_df["day_of_week"].map(dow_map)
            fig = px.bar(dow_df, x="day_name", y="Appliances",
                         title="Avg Consumption by Day of Week",
                         labels={"Appliances": "Avg Wh", "day_name": "Day"},
                         color="Appliances", color_continuous_scale="Greens",
                         category_orders={"day_name": list(dow_map.values())})
            fig.update_layout(height=340, coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

        pod_df = df_eng.groupby("part_of_day")["Appliances"].mean().reset_index()
        order = ["Morning", "Late Morning", "Afternoon", "Evening", "Night"]
        pod_df["part_of_day"] = pd.Categorical(pod_df["part_of_day"], categories=order, ordered=True)
        pod_df = pod_df.sort_values("part_of_day")
        fig = px.bar(pod_df, x="part_of_day", y="Appliances",
                     title="Avg Consumption by Part of Day",
                     labels={"Appliances": "Avg Wh", "part_of_day": ""},
                     color="Appliances", color_continuous_scale="Oranges")
        fig.update_layout(height=300, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        env_col = st.selectbox(
            "Environmental variable",
            ["T1", "T2", "T3", "T_out", "RH_1", "RH_out", "Windspeed", "Visibility", "Tdewpoint", "Press_mm_hg"],
        )
        sample = df_raw.sample(min(3000, len(df_raw)), random_state=42)
        fig = px.scatter(sample, x=env_col, y="Appliances",
                         title=f"Appliances vs {env_col}",
                         labels={"Appliances": "Appliances (Wh)", env_col: env_col},
                         opacity=0.4, trendline="ols",
                         color_discrete_sequence=["#3b82d4"])
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Scatter plot uses a random sample of 3,000 observations for performance. Trend line = OLS regression.")

    with tab4:
        fig = px.scatter(
            df_raw.sample(min(3000, len(df_raw)), random_state=1),
            x="lights", y="Appliances",
            title="Lighting vs Appliance Consumption",
            labels={"lights": "Lights (Wh)", "Appliances": "Appliances (Wh)"},
            opacity=0.4, trendline="ols",
            color_discrete_sequence=["#7c5cd8"],
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown(
            "**Note:** Lighting (`lights`) is a separate energy measurement for "
            "lighting fixtures and is distinct from appliance consumption. Both can "
            "be elevated at the same time (e.g., evenings)."
        )

    with tab5:
        mon_df = df_eng.groupby("month")["Appliances"].agg(["mean", "median", "std"]).reset_index()
        mon_df.columns = ["month", "Mean", "Median", "Std"]
        mon_map = {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May"}
        mon_df["Month"] = mon_df["month"].map(mon_map)
        fig = go.Figure()
        fig.add_trace(go.Bar(x=mon_df["Month"], y=mon_df["Mean"], name="Mean", marker_color="#3b82d4"))
        fig.add_trace(go.Bar(x=mon_df["Month"], y=mon_df["Median"], name="Median", marker_color="#7c5cd8"))
        fig.update_layout(
            title="Monthly Appliance Consumption",
            yaxis_title="Wh",
            barmode="group",
            height=360,
        )
        st.plotly_chart(fig, use_container_width=True)


# ===========================================================================
# PAGE: ML PREDICTION
# ===========================================================================
def page_ml_prediction(df_eng, model, feature_cols):
    st.markdown("## 🤖 ML Prediction")
    st.markdown(
        "Configure environmental and time conditions to get a **model-estimated "
        "appliance energy consumption** for a single 10-minute interval."
    )
    st.markdown("---")

    if model is None:
        st.error("Trained model not found. Please run `python src/train.py` first.")
        return

    st.markdown(
        '<div class="info-box">This page uses the saved trained model to estimate '
        'appliance energy consumption. It does not retrain the model. The prediction '
        'is a model estimate and may not reflect actual consumption in a specific '
        'household.</div>',
        unsafe_allow_html=True,
    )

    # Median values from training data as defaults
    med = df_eng[feature_cols].median()

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("#### Indoor Conditions")
        T1 = st.slider("Kitchen Temp T1 (°C)", 16.0, 30.0, float(round(med.get("T1", 21.0), 1)), 0.1)
        RH_1 = st.slider("Kitchen Humidity RH_1 (%)", 20.0, 70.0, float(round(med.get("RH_1", 40.0), 1)), 0.5)
        T2 = st.slider("Living Room Temp T2 (°C)", 16.0, 30.0, float(round(med.get("T2", 20.0), 1)), 0.1)
        RH_2 = st.slider("Living Room Humidity RH_2 (%)", 20.0, 60.0, float(round(med.get("RH_2", 40.0), 1)), 0.5)
        T3 = st.slider("Laundry Temp T3 (°C)", 16.0, 30.0, float(round(med.get("T3", 22.0), 1)), 0.1)
        lights = st.slider("Lighting Load (Wh)", 0, 70, int(med.get("lights", 0)))

    with c2:
        st.markdown("#### Outdoor Conditions")
        T_out = st.slider("Outdoor Temp T_out (°C)", -6.0, 27.0, float(round(med.get("T_out", 7.0), 1)), 0.1)
        RH_out = st.slider("Outdoor Humidity (%)", 20.0, 100.0, float(round(med.get("RH_out", 80.0), 1)), 1.0)
        Windspeed = st.slider("Wind Speed (m/s)", 0.0, 14.0, float(round(med.get("Windspeed", 3.5), 1)), 0.5)
        Visibility = st.slider("Visibility (km)", 1.0, 66.0, float(round(med.get("Visibility", 40.0), 1)), 1.0)
        Tdewpoint = st.slider("Dew Point (°C)", -7.0, 16.0, float(round(med.get("Tdewpoint", 3.0), 1)), 0.1)
        Press_mm_hg = st.slider("Pressure (mm Hg)", 729.0, 773.0, float(round(med.get("Press_mm_hg", 756.0), 1)), 0.5)

    with c3:
        st.markdown("#### Time")
        hour = st.slider("Hour of Day", 0, 23, int(med.get("hour", 12)))
        dow = st.slider("Day of Week (0=Mon, 6=Sun)", 0, 6, int(med.get("day_of_week", 2)))
        month = st.slider("Month", 1, 5, int(med.get("month", 3)))
        is_weekend = 1 if dow >= 5 else 0
        st.markdown(f"**Is Weekend:** {'Yes' if is_weekend else 'No'}")

        st.markdown("#### Recent Appliance Use (Lag values)")
        st.markdown("<small>Set to approximate recent consumption level</small>", unsafe_allow_html=True)
        lag_default = int(med.get("appliances_lag_1", 60))
        recent_load = st.slider("Approx. recent consumption (Wh)", 10, 500, lag_default, 10)

    # Build input row
    input_row = {col: float(med.get(col, 0.0)) for col in feature_cols}
    # Override with user inputs
    for key, val in {
        "T1": T1, "RH_1": RH_1, "T2": T2, "RH_2": RH_2, "T3": T3,
        "T_out": T_out, "RH_out": RH_out, "Windspeed": Windspeed,
        "Visibility": Visibility, "Tdewpoint": Tdewpoint,
        "Press_mm_hg": Press_mm_hg, "lights": lights,
        "hour": hour, "day_of_week": dow, "month": month,
        "is_weekend": is_weekend,
    }.items():
        if key in input_row:
            input_row[key] = float(val)

    # Set cyclical features
    input_row["hour_sin"] = float(np.sin(2 * np.pi * hour / 24))
    input_row["hour_cos"] = float(np.cos(2 * np.pi * hour / 24))
    input_row["dow_sin"] = float(np.sin(2 * np.pi * dow / 7))
    input_row["dow_cos"] = float(np.cos(2 * np.pi * dow / 7))
    input_row["month_sin"] = float(np.sin(2 * np.pi * month / 12))
    input_row["month_cos"] = float(np.cos(2 * np.pi * month / 12))

    # Fill lag features with user's "recent load" value
    for lag in LAG_STEPS:
        key = f"appliances_lag_{lag}"
        if key in input_row:
            input_row[key] = float(recent_load)
    for w in ROLLING_WINDOWS:
        for stat in ["mean", "std"]:
            key = f"appliances_roll_{stat}_{w}"
            if key in input_row:
                input_row[key] = float(recent_load if stat == "mean" else recent_load * 0.3)

    st.markdown("---")
    if st.button("🔮 Predict Energy Consumption", type="primary"):
        try:
            X_input = pd.DataFrame([input_row])[feature_cols]
            prediction = float(model.predict(X_input)[0])
            prediction = max(10.0, prediction)

            col_r1, col_r2, col_r3 = st.columns(3)
            col_r1.metric(
                "Model-Estimated Appliance Consumption",
                f"{prediction:.1f} Wh",
                help="Per 10-minute interval",
            )
            col_r2.metric(
                "Estimated Hourly Rate",
                f"{prediction * 6:.0f} Wh/hr",
            )
            col_r3.metric(
                "Estimated Daily Rate",
                f"{prediction * 144 / 1000:.2f} kWh/day",
                help="Assumes this level sustained all day",
            )

            if prediction > 300:
                st.warning("The model estimates relatively high appliance consumption for these conditions.")
            elif prediction < 50:
                st.success("The model estimates low appliance consumption for these conditions.")
            else:
                st.info("The model estimates moderate appliance consumption for these conditions.")

            st.caption(
                "This is a model-estimated value based on the trained machine learning model. "
                "It is not a guarantee of actual energy consumption. Prediction accuracy varies."
            )
        except Exception as e:
            st.error(f"Prediction error: {e}")


# ===========================================================================
# PAGE: MODEL PERFORMANCE
# ===========================================================================
def page_model_performance(df_eng, model, metrics, feature_cols):
    st.markdown("## 🔍 Model Performance")
    st.markdown("Evaluation results for all trained models on the chronological test set.")
    st.markdown("---")

    if model is None or metrics is None:
        st.error("Model artifacts not found. Run `python src/train.py` first.")
        return

    # Model comparison table
    st.markdown("### Model Comparison")
    comp_df = pd.DataFrame(metrics["models"])[["model", "MAE", "RMSE", "R2", "train_R2", "train_time_s"]]
    comp_df.columns = ["Model", "MAE (Wh)", "RMSE (Wh)", "Test R²", "Train R²", "Train Time (s)"]
    comp_df = comp_df.sort_values("RMSE (Wh)")
    st.dataframe(comp_df.style.highlight_min(subset=["RMSE (Wh)", "MAE (Wh)"], color="#d1fae5")
                 .highlight_max(subset=["Test R²"], color="#d1fae5"), use_container_width=True)

    st.success(f"**Best model:** {metrics['best_model']} (lowest test RMSE)")

    with st.expander("What do these metrics mean?"):
        st.markdown(
            """
- **MAE (Mean Absolute Error):** Average absolute difference between predicted and actual values. Lower is better.
- **RMSE (Root Mean Squared Error):** Square root of average squared errors. More sensitive to large errors. Lower is better.
- **Test R²:** Proportion of variance in the test target explained by the model. Closer to 1.0 is better; values above 0 mean the model beats a naive mean prediction.
- **Train R²:** R² on the training set. Much higher train R² than test R² indicates overfitting.

*The train/test split is chronological (not random): earlier observations are training data, later observations are test data.*
            """
        )

    st.markdown("---")

    # Actual vs Predicted plot
    st.markdown("### Actual vs Predicted (Test Set)")
    X, y = build_X_y(df_eng)
    n = len(X)
    split_idx = int(n * (1 - metrics["test_ratio"]))
    X_test = X.iloc[split_idx:]
    y_test = y.iloc[split_idx:]
    dates_test = df_eng["date"].iloc[split_idx:] if "date" in df_eng.columns else pd.RangeIndex(len(y_test))

    y_pred = model.predict(X_test)
    y_pred = np.maximum(10.0, y_pred)

    n_plot = min(500, len(y_test))
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(n_plot)), y=y_test.values[:n_plot],
        mode="lines", name="Actual", line=dict(color="#3b82d4", width=1.5)
    ))
    fig.add_trace(go.Scatter(
        x=list(range(n_plot)), y=y_pred[:n_plot],
        mode="lines", name="Predicted", line=dict(color="#f97316", width=1.5, dash="dash")
    ))
    fig.update_layout(
        title=f"Actual vs Predicted — first {n_plot} test observations",
        xaxis_title="Observation index (test set)",
        yaxis_title="Appliances (Wh)",
        height=380,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        # Scatter: actual vs predicted
        fig2 = px.scatter(
            x=y_test.values[:1000], y=y_pred[:1000],
            labels={"x": "Actual (Wh)", "y": "Predicted (Wh)"},
            title="Scatter: Actual vs Predicted (first 1,000 test points)",
            opacity=0.4,
            color_discrete_sequence=["#3b82d4"],
        )
        max_val = max(y_test.values[:1000].max(), y_pred[:1000].max())
        fig2.add_shape(type="line", x0=0, y0=0, x1=max_val, y1=max_val,
                       line=dict(color="#ef4444", dash="dash"))
        fig2.update_layout(height=370)
        st.plotly_chart(fig2, use_container_width=True)

    with c2:
        residuals = y_test.values - y_pred
        fig3 = px.histogram(
            x=residuals, nbins=60,
            title="Residual Distribution (Actual − Predicted)",
            labels={"x": "Residual (Wh)", "y": "Count"},
            color_discrete_sequence=["#7c5cd8"],
        )
        fig3.add_vline(x=0, line_color="#ef4444", line_dash="dash")
        fig3.update_layout(height=370)
        st.plotly_chart(fig3, use_container_width=True)

    # Feature importance
    st.markdown("---")
    st.markdown("### Feature Importance")
    st.markdown(
        '<div class="info-box">Feature importance shows how much each variable '
        'contributes to the model\'s predictions. It does <strong>not</strong> prove '
        'that a feature causes changes in energy consumption — only that it is '
        'useful for prediction.</div>',
        unsafe_allow_html=True,
    )

    try:
        with st.spinner("Computing feature importance …"):
            fi_df = get_feature_importance(model, X_test.iloc[:500], y_test.iloc[:500])
        top_n = st.slider("Show top N features", 5, min(30, len(fi_df)), 15)
        top_fi = fi_df.head(top_n)
        fig_fi = px.bar(
            top_fi.sort_values("importance"),
            x="importance", y="feature",
            orientation="h",
            title=f"Top {top_n} Features by Importance ({top_fi['method'].iloc[0]})",
            color="importance",
            color_continuous_scale="Blues",
        )
        fig_fi.update_layout(height=top_n * 24 + 60, coloraxis_showscale=False, yaxis_title="")
        st.plotly_chart(fig_fi, use_container_width=True)
    except Exception as e:
        st.warning(f"Could not compute feature importance: {e}")


# ===========================================================================
# PAGE: WHAT-IF SIMULATOR
# ===========================================================================
def page_what_if(df_eng, model, feature_cols):
    st.markdown("## 💡 What-If Simulator")
    st.markdown(
        "Compare a **Baseline scenario** with a **Modified scenario** to see how "
        "the trained model responds to changed input conditions."
    )
    st.markdown("---")

    st.markdown(
        '<div class="warn-box"><strong>Important:</strong> This simulation shows how '
        'the trained model responds to a changed input scenario. It does <strong>not</strong> '
        'establish that changing this variable will causally change real-world energy consumption. '
        'Correlation in the training data influences model behaviour, but correlation does '
        'not imply causation.</div>',
        unsafe_allow_html=True,
    )

    if model is None:
        st.error("Trained model not found. Run `python src/train.py` first.")
        return

    med = df_eng[feature_cols].median()

    def build_input(overrides: dict) -> pd.DataFrame:
        row = {col: float(med.get(col, 0.0)) for col in feature_cols}
        for k, v in overrides.items():
            if k in row:
                row[k] = float(v)
        return pd.DataFrame([row])[feature_cols]

    st.markdown("### Configure Scenarios")
    c_b, c_m = st.columns(2)

    with c_b:
        st.markdown("#### Baseline Scenario")
        b_temp = st.slider("Outdoor Temp (°C)", -6.0, 27.0, 7.0, 0.5, key="b_temp")
        b_hour = st.slider("Hour of Day", 0, 23, 8, key="b_hour")
        b_lights = st.slider("Lights (Wh)", 0, 70, 0, key="b_lights")
        b_wind = st.slider("Wind Speed (m/s)", 0.0, 14.0, 3.5, 0.5, key="b_wind")
        b_lag = st.slider("Recent Consumption (Wh)", 10, 500, 60, 10, key="b_lag")
        b_rh = st.slider("Outdoor Humidity (%)", 20.0, 100.0, 80.0, 1.0, key="b_rh")

    with c_m:
        st.markdown("#### Modified Scenario")
        m_temp = st.slider("Outdoor Temp (°C)", -6.0, 27.0, 15.0, 0.5, key="m_temp")
        m_hour = st.slider("Hour of Day", 0, 23, 19, key="m_hour")
        m_lights = st.slider("Lights (Wh)", 0, 70, 40, key="m_lights")
        m_wind = st.slider("Wind Speed (m/s)", 0.0, 14.0, 7.0, 0.5, key="m_wind")
        m_lag = st.slider("Recent Consumption (Wh)", 10, 500, 200, 10, key="m_lag")
        m_rh = st.slider("Outdoor Humidity (%)", 20.0, 100.0, 60.0, 1.0, key="m_rh")

    def scenario_overrides(temp, hour, lights, wind, lag, rh):
        dow = 2  # Wednesday default
        month = 3
        return {
            "T_out": temp,
            "hour": hour, "day_of_week": dow, "month": month,
            "is_weekend": 0,
            "lights": lights,
            "Windspeed": wind,
            "RH_out": rh,
            "hour_sin": np.sin(2 * np.pi * hour / 24),
            "hour_cos": np.cos(2 * np.pi * hour / 24),
            "dow_sin": np.sin(2 * np.pi * dow / 7),
            "dow_cos": np.cos(2 * np.pi * dow / 7),
            "month_sin": np.sin(2 * np.pi * month / 12),
            "month_cos": np.cos(2 * np.pi * month / 12),
            **{f"appliances_lag_{s}": lag for s in LAG_STEPS},
            **{f"appliances_roll_mean_{w}": lag for w in ROLLING_WINDOWS},
            **{f"appliances_roll_std_{w}": lag * 0.3 for w in ROLLING_WINDOWS},
        }

    b_pred = max(10.0, float(model.predict(build_input(scenario_overrides(b_temp, b_hour, b_lights, b_wind, b_lag, b_rh)))[0]))
    m_pred = max(10.0, float(model.predict(build_input(scenario_overrides(m_temp, m_hour, m_lights, m_wind, m_lag, m_rh)))[0]))
    delta = m_pred - b_pred
    delta_pct = (delta / b_pred) * 100

    st.markdown("---")
    st.markdown("### Comparison Results")
    r1, r2, r3 = st.columns(3)
    r1.metric("Baseline Estimate", f"{b_pred:.1f} Wh")
    r2.metric("Modified Estimate", f"{m_pred:.1f} Wh")
    r3.metric("Difference", f"{delta:+.1f} Wh", delta=f"{delta_pct:+.1f}%",
              delta_color="inverse" if delta > 0 else "normal")

    # Bar comparison
    fig = go.Figure([
        go.Bar(name="Baseline", x=["Appliance Consumption"], y=[b_pred], marker_color="#3b82d4"),
        go.Bar(name="Modified", x=["Appliance Consumption"], y=[m_pred], marker_color="#f97316"),
    ])
    fig.update_layout(
        title="Scenario Comparison — Model-Estimated Consumption",
        yaxis_title="Wh per 10-min interval",
        barmode="group",
        height=320,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "Values shown are the machine learning model's output for the configured input scenarios. "
        "They represent model estimates, not real measurements."
    )


# ===========================================================================
# PAGE: ENERGY INSIGHTS
# ===========================================================================
def page_energy_insights(df_raw, df_eng):
    st.markdown("## ⚠️ Energy Insights")
    st.markdown("Analysis of high-consumption observations in the dataset.")
    st.markdown("---")

    # Threshold selection
    method_opt = st.radio(
        "High-consumption threshold method",
        ["90th Percentile", "95th Percentile", "IQR (Q3 + 1.5×IQR)"],
        horizontal=True,
    )

    if method_opt == "90th Percentile":
        threshold = float(df_raw["Appliances"].quantile(0.90))
        label = "90th percentile"
    elif method_opt == "95th Percentile":
        threshold = float(df_raw["Appliances"].quantile(0.95))
        label = "95th percentile"
    else:
        q1 = df_raw["Appliances"].quantile(0.25)
        q3 = df_raw["Appliances"].quantile(0.75)
        iqr = q3 - q1
        threshold = float(q3 + 1.5 * iqr)
        label = "Q3 + 1.5×IQR"

    high_df = df_raw[df_raw["Appliances"] > threshold].copy()
    n_high = len(high_df)
    pct_high = n_high / len(df_raw) * 100

    col1, col2, col3 = st.columns(3)
    col1.metric("High-Consumption Threshold", f"{threshold:.0f} Wh", f"({label})")
    col2.metric("High-Consumption Observations", f"{n_high:,}")
    col3.metric("% of All Observations", f"{pct_high:.1f}%")

    st.markdown(
        '<div class="info-box">These are observations where appliance consumption '
        'exceeded the selected threshold. They are called <strong>high-consumption '
        'observations</strong> — not failures or faults. Higher consumption may reflect '
        'normal household activity (cooking, laundry, etc.).</div>',
        unsafe_allow_html=True,
    )

    st.markdown("---")
    tab1, tab2, tab3 = st.tabs(["Time Patterns", "Environmental Conditions", "Raw Data"])

    with tab1:
        high_eng = df_eng[df_eng["Appliances"] > threshold].copy()

        c1, c2 = st.columns(2)
        with c1:
            hourly_high = high_eng.groupby("hour").size().reset_index(name="count")
            fig = px.bar(hourly_high, x="hour", y="count",
                         title="High-Consumption Observations by Hour",
                         labels={"count": "Count", "hour": "Hour"},
                         color="count", color_continuous_scale="Reds")
            fig.update_layout(height=340, coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            dow_map = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"}
            dow_high = high_eng.groupby("day_of_week").size().reset_index(name="count")
            dow_high["day"] = dow_high["day_of_week"].map(dow_map)
            fig = px.bar(dow_high, x="day", y="count",
                         title="High-Consumption Observations by Day",
                         labels={"count": "Count", "day": ""},
                         color="count", color_continuous_scale="Oranges",
                         category_orders={"day": list(dow_map.values())})
            fig.update_layout(height=340, coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

        pod_high = high_eng.groupby("part_of_day").size().reset_index(name="count")
        order = ["Morning", "Late Morning", "Afternoon", "Evening", "Night"]
        pod_high["part_of_day"] = pd.Categorical(pod_high["part_of_day"], categories=order, ordered=True)
        fig = px.bar(pod_high.sort_values("part_of_day"), x="part_of_day", y="count",
                     title="High-Consumption Observations by Part of Day",
                     labels={"count": "Count", "part_of_day": ""},
                     color="count", color_continuous_scale="YlOrRd")
        fig.update_layout(height=300, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        env_var = st.selectbox(
            "Environmental variable to compare",
            ["T_out", "T1", "RH_out", "RH_1", "Windspeed", "Visibility", "Tdewpoint"],
            key="insight_env",
        )
        comp_df = pd.DataFrame({
            "Group": (["All"] * len(df_raw)) + (["High Consumption"] * len(high_df)),
            env_var: pd.concat([df_raw[env_var], high_df[env_var]]).values,
        })
        fig = px.box(comp_df, x="Group", y=env_var,
                     title=f"{env_var}: All vs High-Consumption Observations",
                     color="Group",
                     color_discrete_map={"All": "#3b82d4", "High Consumption": "#ef4444"})
        fig.update_layout(height=380)
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        display_cols = ["date", "Appliances", "lights", "T1", "T_out", "RH_out"]
        st.dataframe(
            high_df[display_cols].sort_values("Appliances", ascending=False).head(100),
            use_container_width=True,
        )
        st.caption("Showing top 100 high-consumption observations, sorted by consumption.")


# ===========================================================================
# PAGE: SUSTAINABILITY & SDGs
# ===========================================================================
def page_sustainability():
    st.markdown("## 🌍 Sustainability & SDGs")
    st.markdown(
        "This project aligns with four United Nations Sustainable Development Goals. "
        "The connections described below represent areas where the project's insights "
        "*can contribute to* or *support awareness of* broader sustainability objectives."
    )
    st.markdown("---")

    sdgs = [
        {
            "icon": "⚡",
            "num": "SDG 7",
            "title": "Affordable and Clean Energy",
            "color": "#f59e0b",
            "about": (
                "SDG 7 aims to ensure access to affordable, reliable, sustainable and modern "
                "energy for all. A key component is improving energy efficiency."
            ),
            "project": (
                "This project supports awareness of household electricity consumption by "
                "making consumption patterns visible and understandable. The prediction model "
                "provides estimates of appliance energy use under different environmental "
                "conditions, which can inform energy-efficiency decisions."
            ),
            "provides": (
                "Interactive consumption analysis, hourly/daily patterns, and "
                "model-estimated predictions for different environmental scenarios."
            ),
        },
        {
            "icon": "🏙️",
            "num": "SDG 11",
            "title": "Sustainable Cities and Communities",
            "color": "#3b82f6",
            "about": (
                "SDG 11 focuses on making cities inclusive, safe, resilient, and sustainable. "
                "Building energy efficiency is a key factor in sustainable urban development."
            ),
            "project": (
                "This project aligns with SDG 11 by providing tools that can contribute to "
                "smarter energy monitoring in buildings. Understanding which time periods and "
                "environmental conditions are associated with high appliance consumption can "
                "support more efficient energy management in residential settings."
            ),
            "provides": (
                "High-consumption analysis by time of day, day of week, and environmental "
                "conditions, providing context relevant to building energy management."
            ),
        },
        {
            "icon": "♻️",
            "num": "SDG 12",
            "title": "Responsible Consumption and Production",
            "color": "#10b981",
            "about": (
                "SDG 12 promotes sustainable consumption and production patterns, including "
                "efficient use of resources and reduction of waste."
            ),
            "project": (
                "This project provides insights related to SDG 12 by helping users understand "
                "energy consumption patterns. Awareness of when and under what conditions "
                "appliances consume the most energy can support more conscious and responsible "
                "energy use in households."
            ),
            "provides": (
                "The What-If Simulator and Consumption Analysis pages allow users to explore "
                "how different conditions relate to model-estimated energy consumption."
            ),
        },
        {
            "icon": "🌱",
            "num": "SDG 13",
            "title": "Climate Action",
            "color": "#22c55e",
            "about": (
                "SDG 13 calls for urgent action to combat climate change. Reducing energy "
                "consumption in buildings is one of many factors that can contribute to "
                "lower energy-related emissions."
            ),
            "project": (
                "This project aligns with SDG 13 in spirit: by increasing awareness of "
                "household energy consumption patterns, it can contribute to individual "
                "decisions that support energy efficiency. The project does not quantify "
                "carbon emission reductions or make claims about direct climate impact."
            ),
            "provides": (
                "Pattern analysis and scenario simulation that can raise awareness of "
                "energy consumption, supporting informed energy-use decisions."
            ),
        },
    ]

    for sdg in sdgs:
        st.markdown(
            f"""
<div class="sdg-card" style="border-left-color:{sdg['color']}">
<h3>{sdg['icon']} {sdg['num']} — {sdg['title']}</h3>
<p><strong>About this SDG:</strong> {sdg['about']}</p>
<p><strong>How this project relates:</strong> {sdg['project']}</p>
<p><strong>What the application provides:</strong> {sdg['provides']}</p>
</div>
""",
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown(
        """
**Important disclaimer:** This project is a machine learning research and education application.
It does not directly achieve any SDG. The connections described above represent areas where the
project's analytical outputs *can contribute to* or *support awareness of* sustainability goals.
Real-world impact would require integration into actual energy management systems and broader
behavioural and policy interventions.
        """
    )


# ===========================================================================
# PAGE: ABOUT
# ===========================================================================
def page_about():
    st.markdown("## ℹ️ About EnergySense AI")
    st.markdown("---")

    st.markdown(
        """
### Project Title
**EnergySense AI — Smart Energy Consumption Prediction & Sustainability Dashboard**

### Problem Statement
How can machine learning be used to predict household appliance energy consumption from
indoor environmental conditions, weather conditions, lighting, and time-related factors,
while providing useful insights that encourage energy efficiency and responsible consumption?

### Objectives
- Predict appliance energy consumption (Wh per 10-minute interval) using environmental and temporal features.
- Provide interactive exploration of energy consumption patterns.
- Enable scenario simulation to understand model behaviour under different conditions.
- Present insights aligned with SDG 7, 11, 12, and 13.

---

### Dataset
**UCI Appliances Energy Prediction**

| Property | Details |
|---|---|
| Records | 19,735 observations |
| Frequency | Every 10 minutes |
| Period | January 11 – May 27, 2016 |
| Location | A house in Stambruges, Belgium |
| Columns | 29 (date, Appliances, lights, 9 indoor T/RH pairs, outdoor weather, 2 noise columns) |
| Missing values | None |
| Duplicate rows | None |

**Official source:**
https://archive.ics.uci.edu/dataset/374/appliances%2Benergy%2Bprediction

**Citation:**
Candanedo, L. (2017). *Appliances Energy Prediction*. UCI Machine Learning Repository.
DOI: [10.24432/C5VC8G](https://doi.org/10.24432/C5VC8G)

---

### Feature Engineering
Temporal features extracted from the date/time column:
- Hour, minute, day of week, month, day of month
- Weekend indicator
- Part of day (Morning / Late Morning / Afternoon / Evening / Night)
- Cyclical encodings (sine/cosine) for hour, day of week, and month

Lag features (past appliance consumption):
- Past values at 10 min, 20 min, 30 min, 1 hr, 2 hr, 3 hr, 4 hr, 6 hr, 8 hr, 12 hr, 24 hr intervals
- Rolling mean and standard deviation over 1 hr, 2 hr, 4 hr, 6 hr, and 24 hr windows

All lag features use `.shift()` to ensure no future information is used.

---

### Machine Learning Methodology
| Model | Notes |
|---|---|
| Linear Regression | StandardScaler pipeline; good generalisation on time-series data |
| Random Forest | Shallow trees (depth=6, min_samples_leaf=30) to prevent overfitting |
| Gradient Boosting | Slow learning rate (0.02), shallow trees (depth=3) for regularisation |

---

### Train/Test Evaluation
A **chronological split** is used: the earliest 80% of observations are used for training
and the latest 20% for testing. This avoids temporal data leakage and produces a more
realistic evaluation of how the model would generalise to future data.

The best model is selected programmatically by **lowest test-set RMSE**.

---

### Technologies
- **Python** — core language
- **pandas / numpy** — data processing
- **scikit-learn** — machine learning models and evaluation
- **plotly** — interactive visualisations
- **Streamlit** — web application framework
- **joblib** — model serialisation

---

### SDG Alignment
- **SDG 7** — Affordable and Clean Energy: supports awareness of household electricity use
- **SDG 11** — Sustainable Cities: contributes insights for smarter building energy monitoring
- **SDG 12** — Responsible Consumption: promotes understanding of consumption patterns
- **SDG 13** — Climate Action: energy efficiency awareness as one contributing factor

---

### Limitations
- The dataset covers a single house over approximately 4.5 months (Jan–May 2016). Findings may not generalise to other households, building types, climates, or time periods.
- The model predicts based on patterns in historical data. It is an estimate, not a measurement.
- Lag features mean the model requires recent past consumption values for real-time prediction.
- Correlation in the training data drives model behaviour; correlation does not imply causation.
- External factors (occupant behaviour, appliance changes, renovations) are not represented.
- The chronological test set (May 2016) shows seasonal differences from training months (Jan–Apr), which limits R² on the test set compared to a random split.

---

### Future Improvements
- Include additional households and longer time periods for more robust generalisation.
- Explore LSTM or other sequence models that can more naturally handle temporal autocorrelation.
- Add occupancy data or smart-meter data for richer feature engineering.
- Build a real-time prediction API connected to live sensor data.
- Incorporate energy pricing data to provide cost-based insights.
- Extend SDG analysis with energy benchmark comparisons.

---

### License & Attribution
Dataset: UCI Machine Learning Repository — Appliances Energy Prediction
Candanedo, L. (2017). DOI: 10.24432/C5VC8G
https://archive.ics.uci.edu/dataset/374/appliances%2Benergy%2Bprediction

This project is for educational and research purposes.
        """
    )


# ===========================================================================
# MAIN ROUTER
# ===========================================================================

def main():
    # Load data and model
    try:
        df_raw, df_eng = load_data()
    except FileNotFoundError as e:
        st.error(str(e))
        st.stop()
    except Exception as e:
        st.error(f"Failed to load dataset: {e}")
        st.stop()

    model = load_model()
    metrics = load_metrics()
    feature_cols = load_feature_columns()

    if page == "🏠 Home":
        if metrics and feature_cols:
            page_home(df_raw, df_eng, metrics, feature_cols)
        else:
            st.warning("Model artifacts not found. Please run `python src/train.py` first.")
            page_home(df_raw, df_eng, {"best_model": "N/A", "models": []}, [])

    elif page == "📊 Data Explorer":
        page_data_explorer(df_raw)

    elif page == "📈 Consumption Analysis":
        page_consumption_analysis(df_raw, df_eng)

    elif page == "🤖 ML Prediction":
        if feature_cols:
            page_ml_prediction(df_eng, model, feature_cols)
        else:
            st.error("Feature columns not found. Run `python src/train.py` first.")

    elif page == "🔍 Model Performance":
        if feature_cols:
            page_model_performance(df_eng, model, metrics, feature_cols)
        else:
            st.error("Model artifacts not found. Run `python src/train.py` first.")

    elif page == "💡 What-If Simulator":
        if feature_cols:
            page_what_if(df_eng, model, feature_cols)
        else:
            st.error("Model artifacts not found. Run `python src/train.py` first.")

    elif page == "⚠️ Energy Insights":
        page_energy_insights(df_raw, df_eng)

    elif page == "🌍 Sustainability & SDGs":
        page_sustainability()

    elif page == "ℹ️ About":
        page_about()


if __name__ == "__main__":
    main()
