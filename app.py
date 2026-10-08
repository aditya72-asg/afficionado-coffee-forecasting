import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(page_title="Afficionado Coffee Forecasting", layout="wide")

@st.cache_data
def load():
    daily = pd.read_csv("daily_ts.csv", parse_dates=["datetime"])
    hourly = pd.read_csv("hourly_ts.csv", parse_dates=["datetime"])
    preds = pd.read_csv("daily_predictions.csv", parse_dates=["datetime"])
    ci = pd.read_csv("prophet_intervals.csv", parse_dates=["ds"])
    results = pd.read_csv("model_results.csv")
    hpred = pd.read_csv("hourly_predictions.csv", parse_dates=["datetime"])
    peaks = pd.read_csv("peak_results.csv")
    return daily, hourly, preds, ci, results, hpred, peaks

daily, hourly, preds, ci, results, hpred, peaks = load()

# ---------------- Sidebar controls ----------------
st.sidebar.title("Controls")
store = st.sidebar.selectbox("Store", sorted(daily["store_location"].unique()))
horizon = st.sidebar.slider("Forecast horizon (days)", 1, 28, 14)
metric = st.sidebar.radio("Metric", ["Revenue", "Quantity"])
all_models = sorted(preds["model"].unique())
default_models = [m for m in ["Prophet", "SARIMA", "Gradient Boosting"] if m in all_models]
models = st.sidebar.multiselect("Models", all_models, default=default_models)

# Forecasts are made for revenue; quantity is approximated with the store's average revenue per unit
d_store = daily[daily["store_location"] == store].sort_values("datetime")
ratio = d_store["revenue"].sum() / d_store["quantity"].sum()
def conv(x):
    return x if metric == "Revenue" else x / ratio
actual_col = "revenue" if metric == "Revenue" else "quantity"

st.title("Afficionado Coffee Roasters: Demand Forecasting Dashboard")
st.caption(f"Store: {store} | Metric: {metric} | Test window forecast, first {horizon} days")

tab1, tab2, tab3, tab4 = st.tabs(
    ["Sales forecast", "Model comparison", "Confidence interval", "Hourly demand and peaks"])

# ---------------- Tab 1: Forecast chart ----------------
with tab1:
    p = preds[(preds["store"] == store) & (preds["model"].isin(models))]
    first_day = p["datetime"].min()
    last_day = first_day + pd.Timedelta(days=horizon - 1)
    p = p[p["datetime"] <= last_day]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=d_store["datetime"], y=d_store[actual_col],
                             name="Actual", line=dict(color="black")))
    for m in models:
        pm = p[p["model"] == m]
        fig.add_trace(go.Scatter(x=pm["datetime"], y=conv(pm["forecast"]), name=m))
    fig.add_vline(x=first_day, line_dash="dash", line_color="grey")
    fig.update_layout(height=450, yaxis_title=metric, xaxis_title="Date")
    st.plotly_chart(fig, use_container_width=True)

    # Accuracy within the selected horizon
    rows = []
    for m in models:
        pm = p[p["model"] == m]
        mae = (pm["actual"] - pm["forecast"]).abs().mean()
        mape = ((pm["actual"] - pm["forecast"]).abs() / pm["actual"]).mean() * 100
        rows.append([m, round(mae, 1), round(mape, 1)])
    st.subheader(f"Accuracy over the first {horizon} forecast days (revenue)")
    st.dataframe(pd.DataFrame(rows, columns=["Model", "MAE", "MAPE %"]), hide_index=True)

# ---------------- Tab 2: Model comparison ----------------
with tab2:
    st.subheader("Average accuracy across stores (28-day test)")
    avg = results.groupby("model")[["MAE", "RMSE", "MAPE_%"]].mean().sort_values("MAPE_%").round(2)
    st.dataframe(avg)
    st.plotly_chart(px.bar(avg.reset_index(), x="model", y="MAPE_%", title="MAPE % by model"),
                    use_container_width=True)

    st.subheader(f"Results for {store}")
    st.dataframe(results[results["store"] == store].sort_values("MAPE_%").round(2), hide_index=True)

    best = results.loc[results.groupby("store")["MAPE_%"].idxmin(), ["store", "model", "MAPE_%"]]
    st.subheader("Best model per store")
    st.dataframe(best.round(2), hide_index=True)

# ---------------- Tab 3: Confidence interval ----------------
with tab3:
    c = ci[ci["store"] == store].sort_values("ds").head(horizon)
    a = preds[(preds["store"] == store) & (preds["model"] == "Prophet")].sort_values("datetime").head(horizon)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=c["ds"], y=conv(c["yhat_upper"]), line=dict(width=0), showlegend=False))
    fig.add_trace(go.Scatter(x=c["ds"], y=conv(c["yhat_lower"]), fill="tonexty",
                             fillcolor="rgba(0,120,200,0.2)", line=dict(width=0), name="95% interval"))
    fig.add_trace(go.Scatter(x=c["ds"], y=conv(c["yhat"]), name="Prophet forecast", line=dict(color="blue")))
    fig.add_trace(go.Scatter(x=a["datetime"], y=conv(a["actual"]), name="Actual", line=dict(color="black")))
    fig.update_layout(height=450, yaxis_title=metric)
    st.plotly_chart(fig, use_container_width=True)

    inside = ((a["actual"].values >= c["yhat_lower"].values) & (a["actual"].values <= c["yhat_upper"].values)).mean() * 100
    st.info(f"Observed coverage in this window: {inside:.1f}% (nominal 95%). "
            "The interval is narrower than the real daily noise, so treat it as a minimum uncertainty range.")

# ---------------- Tab 4: Hourly demand and peaks ----------------
with tab4:
    h = hpred[hpred["store_location"] == store].copy()
    h["date"] = h["datetime"].dt.date.astype(str)
    heat = h.pivot_table(index="hour", columns="date", values="pred_gb")
    st.subheader("Predicted hourly transactions (last 7 days of the test window)")
    st.plotly_chart(px.imshow(heat, aspect="auto", color_continuous_scale="YlOrRd",
                              labels=dict(x="Date", y="Hour", color="Transactions")),
                    use_container_width=True)

    hh = hourly[hourly["store_location"] == store].copy()
    hh["hour"] = hh["datetime"].dt.hour
    hh["dow"] = hh["datetime"].dt.day_name()
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    hist = hh.pivot_table(index="hour", columns="dow", values="transactions", aggfunc="mean")
    hist = hist.reindex(columns=order)
    st.subheader("Average hourly transactions by weekday (full history)")
    st.plotly_chart(px.imshow(hist, aspect="auto", color_continuous_scale="YlOrRd",
                              labels=dict(x="Weekday", y="Hour", color="Avg transactions")),
                    use_container_width=True)

    st.subheader("Predicted peak hours (model forecast >= store peak threshold)")
    pk = h[h["pred_gb"] >= h["thr"]][["datetime", "pred_gb", "thr", "transactions"]]
    pk = pk.rename(columns={"pred_gb": "predicted", "thr": "peak threshold", "transactions": "actual"})
    st.dataframe(pk.round(1), hide_index=True)

    st.subheader("Peak detection quality (all stores)")
    st.dataframe(peaks.round(1), hide_index=True)