"""Guided Class 12 interface with the original research view still available."""
import hashlib
import io
import json
import runpy
from dataclasses import asdict, replace
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from crashlab.data import SYMBOLS, fetch_market, validate
from crashlab.estimation import calculus, fit_all
from crashlab.risk import summarize
from crashlab.simulation import Config, simulate

st.set_page_config(page_title="CrashLab | Learn market risk", page_icon="📊", layout="wide")
view = st.sidebar.radio("Choose your view", ["Student view", "Advanced view"])
if view == "Advanced view":
    runpy.run_path(str(Path(__file__).with_name("advanced_app.py")), run_name="__main__")
    st.stop()

st.sidebar.title("Quick guide")
st.sidebar.markdown("**1.** Choose a market and load its prices.\n\n**2.** Choose a time period and size of fall.\n\n**3.** Run the simulation and read the results.")
st.sidebar.caption("Advanced view contains the original model settings, stress tests and historical replay.")
st.markdown("""<style>
div[data-testid="stMetric"] {background: #101d30; border: 1px solid #233752; padding: 18px; border-radius: 12px;}
div[data-testid="stMetricLabel"] {white-space: normal;}
</style>""", unsafe_allow_html=True)
st.title("CrashLab")
st.write("**Explore stock market risk with Class 12 maths.**")
st.caption("Use real prices to explore possible futures. No model can tell you with certainty when a crash will happen.")

MODEL_NAMES = {
    "GBM": "Constant volatility (GBM)",
    "GARCH": "Changing volatility (GARCH)",
    "Jump": "Sudden price moves (Jump)",
    "Regime": "Different market conditions (Regime)",
    "Bootstrap": "Resampled historical returns (Bootstrap)",
}
HORIZONS = {"1 month · 21 trading days": 21, "3 months · 63 trading days": 63,
            "6 months · 126 trading days": 126, "1 year · 252 trading days": 252}


@st.cache_data(ttl=3600, max_entries=12, show_spinner=False)
def load_prices(symbol, start, end):
    return fetch_market(symbol, start, end)


@st.cache_data(ttl=3600, max_entries=4, show_spinner=False)
def fit_prices(returns):
    return fit_all(returns, student=True, asymmetric=False)


st.subheader("1 · Choose a market")
market = st.selectbox("Which market would you like to explore?", list(SYMBOLS),
                      help="An index tracks a group of companies. Choose NIFTY 50 for India, or one of the US indices.")
st.caption("We will request the past five years of real closing prices. Availability depends on the data provider.")
load = st.button("Load market prices", type="primary")
with st.expander("Optional: use your own historical prices CSV"):
    st.write("Upload a CSV with **Date** and **Close** columns, at least 252 rows, unique dates and positive prices. For shares, use consistently adjusted prices.")
    upload = st.file_uploader("Historical prices CSV", type=["csv"])
    st.caption("An uploaded file takes priority over the selected market. Remove it to use downloaded market prices.")

if load and upload is None:
    end = date.today() + timedelta(days=1)
    start = date.today() - timedelta(days=5 * 365 + 2)
    try:
        with st.spinner("Loading real prices…"):
            downloaded = load_prices(SYMBOLS[market], start, end)
        st.session_state["student_prices"] = downloaded
        st.session_state["student_market"] = market
        st.session_state["student_origin"] = "download"
    except Exception:
        st.error("We could not download these prices. Try again, choose another market, or upload a real CSV using the optional section above.")

upload_valid = True
if upload is not None:
    file_key = hashlib.sha256(upload.getvalue()).hexdigest()
    if st.session_state.get("student_upload_key") != file_key or st.session_state.get("student_origin") != "upload":
        try:
            uploaded = validate(pd.read_csv(io.BytesIO(upload.getvalue())))
            uploaded.attrs["source"] = "Your uploaded CSV; source not independently verified"
            st.session_state.update(student_prices=uploaded, student_market="Your CSV",
                                    student_origin="upload", student_upload_key=file_key)
        except Exception as exc:
            upload_valid = False
            st.error(f"Please check your CSV: {exc}")

prices = st.session_state.get("student_prices")
ready = prices is not None and upload_valid and (
    upload is not None or (st.session_state.get("student_origin") == "download"
                           and st.session_state.get("student_market") == market))
if ready:
    data_label = st.session_state["student_market"]
    st.success(f"Ready: {data_label} · {len(prices):,} prices · latest close {prices.index[-1]:%d %b %Y}")
    if prices.index[-1] < pd.Timestamp(date.today()) - pd.Timedelta(days=7):
        st.warning("These prices are more than a week old. Results will describe that historical snapshot, not today's market.")
    with st.expander("Where did these prices come from?"):
        st.write(prices.attrs.get("source", "Your uploaded CSV"))
        st.write(f"Available history: {prices.index[0]:%d %b %Y} to {prices.index[-1]:%d %b %Y}.")
else:
    st.info("Load historical market prices or upload a real CSV to begin. If you change the market, load its prices again.")

st.subheader("2 · Choose what to explore")
left, right = st.columns(2)
period = left.selectbox("How far ahead?", list(HORIZONS), index=1,
                        help="Trading days are days with a market observation, excluding weekends and market holidays.")
fall_percent = right.selectbox("What size of fall should we count?", [10, 20, 30], index=1,
                                format_func=lambda value: f"{value}% fall",
                                help="A 20% fall means a starting value of 100 reaches 80 or below.")
horizon = HORIZONS[period]
threshold = fall_percent / 100
st.write(f"**Our question:** How often do simulated prices fall at least **{fall_percent}%** below their starting value within **{horizon} trading days**?")
st.caption("We count a fall at any simulated daily close, even if the market later recovers.")
with st.expander("How the simulation is set up"):
    st.write("We fit five mathematical models using up to the latest five years of returns. Each available model generates 10,000 possible futures. Student view uses fixed settings so you can focus on understanding the results.")
    st.write("Volatility means how much prices fluctuate. A simulated future is called a path. Advanced view lets you change the underlying assumptions.")

st.subheader("3 · Run and understand")
run = st.button("Run my simulation", type="primary", disabled=not ready)
if not ready:
    st.caption("Start with the Load market prices button in step 1.")
    st.stop()

r = np.diff(np.log(prices.iloc[-1261:].values))
s0 = float(prices.iloc[-1])
cfg = Config(paths=10000, horizon=horizon, threshold=threshold, seed=42, definition="starting loss", block=10)
data_key = hashlib.sha256(prices.values.tobytes() + prices.index.values.tobytes()
                          + data_label.encode()).hexdigest()
experiment_key = (data_key, horizon, threshold)
if run:
    with st.spinner("Learning from the price history and generating possible futures…"):
        fitted, errors = fit_prices(r)
        results, visuals = {}, {}
        for idx, name in enumerate(MODEL_NAMES):
            if name not in fitted:
                continue
            mc = replace(cfg, seed=cfg.seed + idx * 1009)
            try:
                arr, extra = simulate(name, fitted[name], s0, mc)
                results[name] = {**summarize(arr, mc), **extra}
                visuals[name] = {"bands": np.quantile(arr, [.05, .5, .95], axis=0),
                                 "sample": arr[:30].copy()}
                del arr
            except ValueError as exc:
                errors[name] = str(exc)
        st.session_state.update(student_results=results, student_visuals=visuals,
                                student_errors=errors, student_result_key=experiment_key,
                                student_config=asdict(cfg))

if st.session_state.get("student_result_key") != experiment_key:
    if "student_results" in st.session_state:
        st.warning("Your choices or data changed. Run my simulation again to update the results.")
    else:
        st.info("Your data is ready. Click Run my simulation to see the results.")
    st.stop()

results = st.session_state["student_results"]
if not results:
    st.error("No model could complete this simulation. Try another dataset or review the data in Advanced view.")
    st.stop()

results_tab, graphs_tab, maths_tab = st.tabs(["Your results", "Price graphs", "Class 12 maths"])
with results_tab:
    probabilities = [result["probability"] for result in results.values()]
    a, b, c = st.columns(3)
    a.metric("Estimates across models", f"{min(probabilities):.1%} – {max(probabilities):.1%}")
    b.metric("Possible futures per model", "10,000")
    c.metric("Fall being counted", f"{fall_percent}%")
    st.write(f"Each model gives a different estimate of a **{fall_percent}% fall within {horizon} trading days**. The range above compares their estimates; it is not one combined probability.")
    rows = [{"Model": MODEL_NAMES[name], "Estimated chance of this fall": f"{value['probability']:.2%}",
             "Paths with this fall / 10,000": f"{round(value['probability'] * cfg.paths):,}"}
            for name, value in results.items()]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    name = st.selectbox("Explain one model's result", list(results), format_func=MODEL_NAMES.get)
    result = results[name]
    count = round(result["probability"] * cfg.paths)
    st.info(f"For {MODEL_NAMES[name]}, {count:,} out of 10,000 simulated futures had this fall. That is an estimated chance of {result['probability']:.2%} under this model's assumptions.")
    if count == 0:
        st.write("No fall was observed in this sample. That does not mean the true chance is zero.")
    st.caption("These are mathematical estimates, not a certainty or a buy/sell recommendation. More simulations reduce random noise; they do not fix incorrect assumptions.")
    with st.expander("How to read uncertainty"):
        lo, hi = result["mc_ci95"]
        st.write(f"This model's 95% simulation sampling interval is {lo:.2%} to {hi:.2%}. It describes random variation from generating a limited number of paths. It does not include uncertainty about future market conditions or the fitted model.")
    if st.session_state["student_errors"]:
        with st.expander("Why are some models missing?"):
            st.write("Some models could not be fitted or simulated reliably, so their results are omitted.")
            for model, error in st.session_state["student_errors"].items():
                st.write(f"**{MODEL_NAMES.get(model, model)}:** {error}")
    export = {"market": data_label, "latest_observation": str(prices.index[-1].date()),
              "source": prices.attrs.get("source", "CSV"),
              "configuration": st.session_state["student_config"], "models": results}
    st.download_button("Download results as CSV", pd.DataFrame(rows).to_csv(index=False),
                       "crashlab_student_results.csv", "text/csv")
    with st.expander("Download full mathematical results"):
        st.download_button("Download full results JSON", json.dumps(export, indent=2),
                           "crashlab_student_results.json", "application/json")

with graphs_tab:
    st.write("**Past prices: what actually happened**")
    history = prices / prices.iloc[0] * 100
    fig = go.Figure(go.Scatter(x=history.index, y=history.values, name="Actual history"))
    fig.update_layout(xaxis_title="Date", yaxis_title="Market value (first observation = 100)")
    st.plotly_chart(fig, width="stretch")
    st.caption("The first observed price is rescaled to 100 to make changes easier to compare. For example, 120 means a 20% increase from that observation.")
    st.write("**Possible futures: what the selected model simulated**")
    trajectory = st.selectbox("Choose a model for the future graph", list(results), format_func=MODEL_NAMES.get)
    v = st.session_state["student_visuals"][trajectory]
    fig = go.Figure()
    x = np.arange(horizon + 1)
    for row in v["sample"]:
        fig.add_trace(go.Scatter(x=x, y=row / s0 * 100, line=dict(color="rgba(53,208,192,.18)", width=1), showlegend=False))
    for label, band in zip(["5th percentile", "Middle outcome (median)", "95th percentile"], v["bands"]):
        fig.add_trace(go.Scatter(x=x, y=band / s0 * 100, name=label, line=dict(width=3)))
    fig.add_hline(y=100 * (1 - threshold), line_dash="dash", line_color="#f45c6b",
                  annotation_text=f"{fall_percent}% fall level")
    fig.update_layout(xaxis_title="Trading days from now", yaxis_title="Market value (latest starting close = 100)")
    st.plotly_chart(fig, width="stretch")
    st.write("Thin lines show 30 example futures. The red line is the fall level. The median and percentile curves summarise all 10,000 paths at each day; they are not individual predicted journeys.")
    st.caption("The band describes simulated values under the model. Actual future prices can lie outside it.")

with maths_tab:
    st.write("**Connect the results to maths you study in Class 12.**")
    with st.expander("Percentages: what does a fall mean?", expanded=True):
        st.latex(r"\text{Percentage change} = \frac{\text{New value} - \text{Old value}}{\text{Old value}}\times 100")
        st.write(f"Starting at 100, a {fall_percent}% fall gives 100 × (1 − {threshold:g}) = {100 * (1 - threshold):g}.")
    with st.expander("Probability: how is the chance calculated?"):
        st.latex(r"\text{Estimated probability} = \frac{\text{Simulated paths with the fall}}{\text{Total simulated paths}}")
        st.write(f"For {MODEL_NAMES[name]}: {count:,} ÷ 10,000 = {result['probability']:.4f}, or {result['probability']:.2%}.")
        st.write("We count whole futures that crossed the fall level at least once. We do not count individual falling days. This is a simulation frequency, not proof of the actual future probability.")
    with st.expander("Graphs: reading the axes and the median"):
        st.write("The horizontal axis shows time. The vertical axis shows the market's value. The median is the middle simulated value at each day: approximately half the paths are above it and half below.")
        st.write("At each day, the 5th to 95th percentile band contains approximately the middle 90% of simulated values. It is not a guarantee that the whole future path stays inside the band.")
    with st.expander("Derivatives: how quickly did past prices change?"):
        st.write("If P(t) is price, P′(t) describes its rate of change and P″(t) describes how that rate is changing. We smooth past prices so short-term noise is easier to see.")
        d = calculus(prices.values)
        fig = go.Figure(go.Scatter(x=prices.index, y=d["velocity"], name="Rate of change"))
        fig.add_hline(y=0, line_dash="dash")
        fig.update_layout(xaxis_title="Date", yaxis_title="Smoothed price change per observation")
        st.plotly_chart(fig, width="stretch")
        st.caption("Above zero means the smoothed price was rising; below zero means it was falling. This uses neighbouring historical points and is descriptive, not a crash predictor.")
    with st.expander("Why do the mathematical models disagree?"):
        st.dataframe(pd.DataFrame([
            {"Model": MODEL_NAMES["GBM"], "Main idea": "Random price changes with a constant volatility level."},
            {"Model": MODEL_NAMES["GARCH"], "Main idea": "Calm and turbulent periods can persist."},
            {"Model": MODEL_NAMES["Jump"], "Main idea": "Occasional sudden shocks accompany ordinary changes."},
            {"Model": MODEL_NAMES["Regime"], "Main idea": "The market can switch between different fitted conditions."},
            {"Model": MODEL_NAMES["Bootstrap"], "Main idea": "Resample 10-day chunks of historical returns."},
        ]), hide_index=True, width="stretch")
        st.write("Different assumptions produce different estimates. Comparing them reveals uncertainty; agreeing models are not proof that a forecast is correct.")
