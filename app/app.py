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

st.set_page_config(page_title="Netflix Stock Predictor", layout="wide")

# -----------------------
# Helper functions
# -----------------------
def load_default_data():
    # Attempt to load cleaned dataset shipped in repo (prefer monthly CSV if present)
    if os.path.exists("results/netflix_monthly_close.csv"):
        dfm = pd.read_csv("results/netflix_monthly_close.csv", index_col=0, parse_dates=True)
        dfm.index.name = "Date"
        dfm = dfm.rename(columns={dfm.columns[0]: "Close"}) if dfm.columns.size == 1 else dfm
        return dfm
    # Fallback: load daily raw and resample
    raw_path = "data/netflix_stock_data.csv"
    if os.path.exists(raw_path):
        df = pd.read_csv(raw_path, skiprows=[0])
        # Normalize columns if necessary
        df.columns = ['Date', 'Close', 'High', 'Low', 'Open', 'Volume', 'MA50', 'MA200'][:len(df.columns)]
        df = df[df['Date'] != 'Date']
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
        df = df.dropna(subset=['Date','Close']).set_index('Date').sort_index()
        dfm = df['Close'].resample('M').mean()
        dfm = dfm.to_frame(name='Close')
        return dfm
    # If nothing found, create a tiny sample to keep UI working
    idx = pd.date_range(start="2019-01-31", periods=24, freq='M')
    sample = pd.Series(100 + np.cumsum(np.random.randn(len(idx))*2), index=idx, name='Close')
    return sample.to_frame()

def sanitize_uploaded_csv(uploaded_file):
    try:
        df = pd.read_csv(uploaded_file)
    except Exception:
        uploaded_file.seek(0)
        df = pd.read_csv(io.StringIO(uploaded_file.read().decode('utf-8')))
    # Try to detect header junk and fix like earlier workflow
    if 'Date' not in df.columns:
        # maybe first row is header-like; try skipping first row
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file, skiprows=[0])
        if df.shape[1] >= 2 and 'Date' not in df.columns:
            df.columns = ['Date','Close','High','Low','Open','Volume','MA50','MA200'][:df.shape[1]]
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    df = df.dropna(subset=['Date','Close']).set_index('Date').sort_index()
    dfm = df['Close'].resample('M').mean().to_frame(name='Close')
    return dfm

def arima_forecast(series, months_ahead):
    # Fit ARIMA on the monthly series (simple general order)
    # You can change the (p,d,q) to tune; (5,1,2) is a reasonable default
    model = ARIMA(series, order=(5,1,2))
    model_fit = model.fit()
    pred = model_fit.get_forecast(steps=months_ahead)
    forecast = pred.predicted_mean
    # ensure monthly index starting after last month-end
    start = series.index[-1] + pd.offsets.MonthEnd(1)
    forecast.index = pd.date_range(start=start, periods=months_ahead, freq='M')
    return forecast, model_fit

def save_forecast_results(series_monthly, forecast):
    os.makedirs("results", exist_ok=True)
    # Save historical monthly closes
    series_monthly.to_csv("results/netflix_monthly_close.csv")
    # Save forecast
    forecast.to_frame(name='predicted_mean').to_csv("results/netflix_12_month_forecast.csv")
    return "results/netflix_12_month_forecast.csv"

def run_agent_analysis(lstm_path, forecast_path):
    # Loads agent code + calls Groq API to analyze Netflix (if key set)
    if not GROQ_AVAILABLE:
        return "Agent (Groq) package not installed in environment."
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return "Groq API key not set. Add GROQ_API_KEY as a Streamlit secret or environment variable."
    # Load files for prompt
    try:
        lstm_df = pd.read_csv(lstm_path) if os.path.exists(lstm_path) else pd.DataFrame()
        forecast_df = pd.read_csv(forecast_path) if os.path.exists(forecast_path) else pd.DataFrame()
    except Exception as e:
        return f"Error loading model files: {e}"
    client = Groq(api_key=api_key)
    prompt = f"""
You are an expert Netflix (NFLX) stock analyst AI. Use ONLY the data below.

--- LSTM Short-Term Prediction (last 10 rows) ---
{lstm_df.tail(10).to_string(index=False)}

--- ARIMA Forecast (last 12 rows) ---
{forecast_df.tail(12).to_string(index=False)}

Tasks:
1. Short-term movement summary (stable / volatile / trending).
2. Long-term trend summary (up / down / flat).
3. Risk rating (Low/Medium/High) + 1-line reason.
4. Where the models agree/disagree.
5. Final beginner-friendly 4-6 sentence summary.
"""
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role":"user","content":prompt}],
            temperature=0.2
        )
        # Groq response object: use .choices[0].message.content
        out_text = response.choices[0].message.content
    except Exception as e:
        out_text = f"Agent call failed: {e}"
    # save agent output
    os.makedirs("results", exist_ok=True)
    fname = "results/netflix_agent_analysis.txt"
    with open(fname, "w") as f:
        f.write(out_text)
    return out_text

# -----------------------
# Streamlit UI
# -----------------------
st.title("📈 Netflix Stock Predictor — ARIMA + Agent")
st.markdown("Upload your Netflix CSV or use the default dataset. Select a forecast horizon and click **Run Forecast**. Click **Get AI Analysis** to have the agent explain the results on the same page.")

col1, col2 = st.columns([1,3])

with col1:
    st.header("Data Input")
    st.write("Default dataset is loaded automatically. You can also upload your own CSV.")
    uploaded = st.file_uploader("Upload Netflix CSV (Date, Close,...)", type=["csv"])
    if uploaded:
        try:
            series_monthly = sanitize_uploaded_csv(uploaded)
            st.success("Uploaded CSV parsed successfully.")
        except Exception as e:
            st.error("Failed to parse uploaded CSV: " + str(e))
            series_monthly = load_default_data()
    else:
        series_monthly = load_default_data()
        st.info("Using default dataset (from repository).")

    st.markdown("---")
    st.header("Forecast Options")
    horizon_mode = st.radio("Pick horizon type", ("Months", "Years"), index=0)
    if horizon_mode == "Months":
        months = st.slider("Forecast months", min_value=1, max_value=60, value=12)
    else:
        years = st.slider("Forecast years", min_value=1, max_value=5, value=1)
        months = years * 12

    st.markdown("---")
    st.write("ARIMA model order used: (5,1,2). You can change code to tune this.")

    run_button = st.button("Run Forecast")

with col2:
    st.header("Forecast & Visualization")
    if run_button:
        with st.spinner("Running ARIMA forecast..."):
            try:
                forecast, arima_model = arima_forecast(series_monthly['Close'], months)
                # Save results
                save_forecast_results(series_monthly['Close'], forecast)
                # Display plot
                fig, ax = plt.subplots(figsize=(10,5))
                ax.plot(series_monthly.index, series_monthly['Close'], label="Historical (monthly)", linewidth=2)
                ax.plot(forecast.index, forecast.values, linestyle='--', color='orange', label=f'Forecast next {months} months', linewidth=2)
                ax.set_title("Netflix — Historical vs Forecast (Monthly)")
                ax.set_ylabel("Close Price (USD)")
                ax.legend()
                st.pyplot(fig)
                # Show forecast table (last N)
                st.subheader("Forecast (first 12 rows shown)")
                df_forecast = forecast.to_frame(name='predicted_mean')
                st.dataframe(df_forecast.head(12).style.format("{:.2f}"))
                # Save model file
                os.makedirs("models", exist_ok=True)
                joblib.dump(arima_model, "models/arima_trend_model.pkl")
                st.success("Forecast completed and saved to results/ and models/.")
            except Exception as e:
                st.error("Forecast failed: " + str(e))
    else:
        st.info("Press **Run Forecast** to generate ARIMA forecast using the chosen horizon.")

    st.markdown("---")
    # Agent button & area
    st.button("Get AI Analysis", key="agent_button")  # placeholder - will be captured below

    # Show last saved agent output if it exists
    agent_display = st.empty()
    if os.path.exists("results/netflix_agent_analysis.txt"):
        with open("results/netflix_agent_analysis.txt", "r") as f:
            existing = f.read()
        agent_display.markdown("### AI Analysis (last saved)")
        agent_display.text(existing)

# Listen for the agent button click (same page)
if st.session_state.get("agent_button", False) or st.button("Get AI Analysis"):
    st.info("Generating AI analysis (calls Groq). Make sure GROQ_API_KEY is set in Secrets.")
    with st.spinner("Calling the Netflix Agent..."):
        lstm_path = "results/lstm_predictions.csv" # expected file (if present) from your LSTM stage
        forecast_path = "results/netflix_12_month_forecast.csv"
        agent_text = run_agent_analysis(lstm_path, forecast_path)
    st.markdown("### Netflix Agent Output")
    st.write(agent_text)
    st.success("Agent output saved to results/netflix_agent_analysis.txt")

# Footer
st.markdown("---")
st.markdown("**Notes:** ARIMA is used for fast live forecasting on CPU (Streamlit Cloud). The AI Agent is powered by Groq LLaMA models — set `GROQ_API_KEY` in Streamlit Secrets to enable. LSTM model outputs are used for agent context if `lstm_predictions.csv` is present.")

