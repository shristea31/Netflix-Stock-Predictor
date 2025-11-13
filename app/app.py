# app/app.py
import os
import io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from statsmodels.tsa.arima.model import ARIMA
import joblib
from datetime import timedelta

# Optional Groq client (agent). Only import if available.
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except Exception:
    GROQ_AVAILABLE = False

# ----------------------
# Theme / Colors (Netflix Red theme)
# ----------------------
NETFLIX_RED = "#E50914"
BG_COLOR = "#121213"
CARD_COLOR = "#1f1f1f"
TEXT_COLOR = "#E6E6E6"
ACCENT = NETFLIX_RED

# Streamlit page config
st.set_page_config(page_title="Netflix Stock Predictor", layout="wide")

# Small CSS for the Netflix look
st.markdown(
    f"""
    <style>
    .reportview-container {{background-color: {BG_COLOR}; color: {TEXT_COLOR};}}
    .stApp {{ background-color: {BG_COLOR}; color: {TEXT_COLOR}; }}
    .big-red {{ color: {NETFLIX_RED}; font-weight:700; }}
    .card {{ background: {CARD_COLOR}; padding: 12px; border-radius: 8px; }}
    .footer {{ color: #888888; font-size:12px; }}
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------
# Helper functions
# -----------------------
def load_default_data():
    """Load cleaned monthly file if present, else resample raw daily file, else sample."""
    # Prefer monthly prepared file
    if os.path.exists("results/netflix_monthly_close.csv"):
        dfm = pd.read_csv("results/netflix_monthly_close.csv", index_col=0, parse_dates=True)
        if dfm.shape[1] == 1:
            dfm.columns = ["Close"]
        return dfm

    # Fallback: load daily raw and resample
    raw_path = "data/netflix_stock_data.csv"
    if os.path.exists(raw_path):
        df = pd.read_csv(raw_path, skiprows=[0])
        # Defensive column normalization
        try:
            df.columns = ['Date', 'Close', 'High', 'Low', 'Open', 'Volume', 'MA50', 'MA200'][:len(df.columns)]
        except Exception:
            pass
        df = df[df['Date'] != 'Date']
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
        df = df.dropna(subset=['Date', 'Close']).set_index('Date').sort_index()
        dfm = df['Close'].resample('M').mean().to_frame(name='Close')
        return dfm

    # If nothing found, create a tiny sample to keep UI working
    idx = pd.date_range(start="2019-01-31", periods=36, freq='M')
    sample = pd.Series(100 + np.cumsum(np.random.randn(len(idx))*2), index=idx, name='Close')
    return sample.to_frame()

def sanitize_uploaded_csv(uploaded_file):
    """Parse uploaded CSV (handles the junk first row pattern). Returns monthly frame."""
    try:
        df = pd.read_csv(uploaded_file)
    except Exception:
        uploaded_file.seek(0)
        df = pd.read_csv(io.StringIO(uploaded_file.read().decode('utf-8')))
    # If Date header missing, try skipping first row like earlier
    if 'Date' not in df.columns:
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file, skiprows=[0])
        if 'Date' not in df.columns and df.shape[1] >= 2:
            df.columns = ['Date','Close','High','Low','Open','Volume','MA50','MA200'][:df.shape[1]]
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    df = df.dropna(subset=['Date','Close']).set_index('Date').sort_index()
    dfm = df['Close'].resample('M').mean().to_frame(name='Close')
    return dfm

def arima_forecast(series, months_ahead, order=(5,1,2)):
    """Fit ARIMA and forecast months_ahead months. Returns forecast Series and fitted model."""
    model = ARIMA(series, order=order)
    model_fit = model.fit()
    pred = model_fit.get_forecast(steps=months_ahead)
    forecast = pred.predicted_mean
    # monthly index starting after last month-end
    start = series.index[-1] + pd.offsets.MonthEnd(1)
    forecast.index = pd.date_range(start=start, periods=months_ahead, freq='M')
    return forecast, model_fit

def save_forecast_results(series_monthly, forecast, fname_prefix="results/netflix"):
    """Save monthly historical and forecast CSVs into results/ and return paths."""
    os.makedirs("results", exist_ok=True)
    series_monthly.to_csv(f"{fname_prefix}_monthly_close.csv")
    forecast.to_frame(name='predicted_mean').to_csv(f"{fname_prefix}_forecast.csv")
    return f"{fname_prefix}_forecast.csv", f"{fname_prefix}_monthly_close.csv"

def run_agent_analysis(lstm_path, forecast_path, model_name="llama-3.3-70b-versatile"):
    """Call Groq agent and return text. Handles missing files gracefully."""
    if not GROQ_AVAILABLE:
        return "Agent (Groq) package not installed in environment. Agent disabled."

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return "Groq API key not set. Add GROQ_API_KEY in Streamlit Secrets."

    # Load files robustly
    try:
        lstm_df = pd.read_csv(lstm_path) if (lstm_path and os.path.exists(lstm_path)) else pd.DataFrame()
    except Exception:
        lstm_df = pd.DataFrame()
    try:
        forecast_df = pd.read_csv(forecast_path) if os.path.exists(forecast_path) else pd.DataFrame()
    except Exception:
        forecast_df = pd.DataFrame()

    client = Groq(api_key=api_key)

    prompt = f"""
You are an expert Netflix (NFLX) stock analyst AI. Use ONLY the data below.

--- LSTM Short-Term Prediction (last 10 rows) ---
{lstm_df.tail(10).to_string(index=False) if not lstm_df.empty else "NO LSTM DATA FOUND"}

--- ARIMA Forecast (last 12 rows) ---
{forecast_df.tail(12).to_string(index=False) if not forecast_df.empty else "NO ARIMA FORECAST FOUND"}

Tasks:
1. Short-term movement summary (stable / volatile / trending).
2. Long-term trend summary (up / down / flat).
3. Risk rating (Low/Medium/High) + 1-line reason.
4. Where the models agree/disagree.
5. Final beginner-friendly 4-6 sentence summary.
"""
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role":"user","content":prompt}],
            temperature=0.2
        )
        out_text = response.choices[0].message.content
    except Exception as e:
        out_text = f"Agent call failed: {e}"

    # save agent output
    os.makedirs("results", exist_ok=True)
    fname = "results/netflix_agent_analysis.txt"
    with open(fname, "w", encoding="utf-8") as f:
        f.write(out_text)
    return out_text

# -----------------------
# Streamlit UI
# -----------------------
st.markdown(f"<h1 class='big-red'>Netflix Stock Predictor</h1>", unsafe_allow_html=True)
st.markdown("<div class='card'><small>ARIMA live forecasting + LSTM (offline) context + Agentic AI explanation</small></div>", unsafe_allow_html=True)
st.markdown("")

col1, col2 = st.columns([1,3])

with col1:
    st.header("Data & Forecast")
    st.write("Default dataset is loaded automatically. You can also upload your own CSV (Date, Close, ...).")
    uploaded = st.file_uploader("Upload Netflix CSV", type=["csv"])
    if uploaded:
        try:
            series_monthly = sanitize_uploaded_csv(uploaded)
            st.success("Uploaded CSV parsed and resampled to monthly.")
        except Exception as e:
            st.error("Failed to parse uploaded CSV: " + str(e))
            series_monthly = load_default_data()
    else:
        series_monthly = load_default_data()
        st.info("Using default dataset (from repository).")

    st.markdown("---")
    st.subheader("Forecast Options")
    horizon_mode = st.radio("Horizon type", ("Months", "Years"), index=1)
    if horizon_mode == "Months":
        months = st.slider("Forecast months", min_value=1, max_value=120, value=12)
    else:
        years = st.slider("Forecast years", min_value=1, max_value=5, value=1)
        months = years * 12

    st.write("ARIMA order: (5,1,2). Change code to tune or replace with auto_arima later.")
    run_button = st.button("Run Forecast", help="Run ARIMA forecast for chosen horizon")

with col2:
    st.header("Visualization & Results")
    if run_button:
        with st.spinner("Running ARIMA forecast..."):
            try:
                forecast, arima_model = arima_forecast(series_monthly['Close'], months)
                # Save results
                forecast_path, monthly_path = save_forecast_results(series_monthly['Close'], forecast, fname_prefix="results/netflix")
                # Save model object
                os.makedirs("models", exist_ok=True)
                joblib.dump(arima_model, "models/arima_trend_model.pkl")
                # Plotting (Netflix-red theme)
                fig, ax = plt.subplots(figsize=(11,5), facecolor=BG_COLOR)
                ax.plot(series_monthly.index, series_monthly['Close'], label="Historical (monthly)", linewidth=2, color="#f0f0f0")
                ax.plot(forecast.index, forecast.values, linestyle='--', color=NETFLIX_RED, label=f"Forecast next {months} months", linewidth=2)
                ax.set_facecolor(BG_COLOR)
                ax.spines['bottom'].set_color(TEXT_COLOR)
                ax.spines['left'].set_color(TEXT_COLOR)
                ax.tick_params(colors=TEXT_COLOR)
                ax.yaxis.label.set_color(TEXT_COLOR)
                ax.xaxis.label.set_color(TEXT_COLOR)
                ax.set_title("Netflix — Historical vs Forecast (Monthly)", color=TEXT_COLOR)
                ax.set_ylabel("Close Price (USD)")
                ax.legend(facecolor=CARD_COLOR)
                st.pyplot(fig)

                st.subheader("Forecast (first 24 rows)")
                df_forecast = forecast.to_frame(name='predicted_mean')
                st.dataframe(df_forecast.head(24).style.format("{:.2f}"))

                st.success("Forecast completed and saved to results/ and models/.")
            except Exception as e:
                st.error("Forecast failed: " + str(e))
    else:
        st.info("Press **Run Forecast** to generate ARIMA forecast using chosen horizon.")

    st.markdown("---")
    # Agent button placed on same page under the forecast
    st.markdown("### AI Analysis")
    # show last saved agent output if exists
    if os.path.exists("results/netflix_agent_analysis.txt"):
        with open("results/netflix_agent_analysis.txt", "r", encoding="utf-8") as f:
            existing = f.read()
        st.markdown("**Last saved AI Analysis:**")
        st.code(existing)

    agent_trigger = st.button("Get AI Analysis (uses ARIMA + LSTM context if available)", key="agent_btn")

# -----------------------
# Agent execution (on same page)
# -----------------------
if agent_trigger:
    st.info("Generating AI analysis. This calls Groq — ensure GROQ_API_KEY is set in Streamlit Secrets.")
    with st.spinner("Calling the Netflix Agent..."):
        # robust detection of LSTM predictions file
        lstm_path = "results/lstm_predictions.csv" if os.path.exists("results/lstm_predictions.csv") else None
        forecast_path = "results/netflix_forecast.csv" if os.path.exists("results/netflix_forecast.csv") else "results/netflix_forecast.csv"
        # prefer the more descriptive file from save_forecast_results function earlier
        if os.path.exists("results/netflix_forecast.csv"):
            forecast_path = "results/netflix_forecast.csv"
        elif os.path.exists("results/netflix_12_month_forecast.csv"):
            forecast_path = "results/netflix_12_month_forecast.csv"
        elif os.path.exists("results/netflix_forecast.csv"):
            forecast_path = "results/netflix_forecast.csv"
        agent_text = run_agent_analysis(lstm_path, forecast_path)
    st.markdown("### Netflix Agent Output")
    st.write(agent_text)
    st.success("Agent output saved to results/netflix_agent_analysis.txt")

st.markdown("---")
st.markdown("<div class='footer'>Notes: ARIMA used for live forecasting (CPU). LSTM is offline and used for agent context if present. Set GROQ_API_KEY as a Streamlit Secret to enable the AI agent.</div>", unsafe_allow_html=True)
