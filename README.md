<h1 align="center">📈**Netflix Stock Predictor with AI Analysis**</h1>


A complete end-to-end financial forecasting and analysis system.

This project predicts Netflix (NFLX) stock prices using a combination of:

👉Machine Learning

👉Deep Learning (LSTM)

👉Statistical Time-Series (ARIMA)

👉Agentic AI (Groq LLaMA-3.3 70B)

👉Interactive Streamlit Web App

*The project integrates forecasting with AI-driven explanations, enabling users to understand short-term volatility, long-term trends, risks, and model agreement.*

<h2>🚀 **Live Demo**</h2>

👉 https://netflix-stock-predictor-bgyapczbmbcv5atvbwip9x.streamlit.app/

<h2>🎥 **Project Overview** </h2>

This system performs:

🔹 LSTM — Deep Learning Short-Term Forecasting (Offline)

- Multivariate LSTM trained on technical indicators

**R² ≈ 0.84**

🔹 ARIMA — Real-Time Forecasting (Online)

- Used inside the Streamlit app

- Predicts 1 month to 5 years

- CPU-friendly and deployable on free hosting

🔹 Agentic AI — Netflix Stock Analyst

Powered by Groq's LLaMA-3.3-70B-Versatile
Takes LSTM + ARIMA outputs and generates:

- Short-term movement

- Long-term trend

- Risk rating

- Model agreement/disagreement

🔹 Streamlit App (Netflix Theme)

- Dark UI with Netflix-red highlights

- Upload custom CSV or use built-in dataset

- Select forecast horizon

 -Plot trends & forecast

**Click “Get AI Analysis” to generate full financial insights.**

 <h2>🧩 **Project Features** </h2>

☑️ Data Pipeline:

✓Cleans corrupted headers

✓Normalizes types

✓Removes junk rows

✓Resamples daily → monthly

☑️Forecasting Engine:

✓ ARIMA(5,1,2)

✓ Supports 1–60 months

✓ Real-time inference in browser

☑️Deep Learning

✓ LSTM with engineered features:

✓ Moving averages

✓ Momentum

✓ Returns

✓ Volatility

 ☑️Agentic Intelligence

✓ Custom financial prompt

✓ Uses Groq API

✓ Human-style, structured analysis

☑️Deployment

✓ Free Streamlit Cloud

<h2>📊 **Screenshots**</h2>

1) Forecast Graph
<img width="1908" height="854" alt="image" src="https://github.com/user-attachments/assets/79211292-536c-4591-b806-e852132e8d15" />

2)Forecast
<img width="1740" height="709" alt="image" src="https://github.com/user-attachments/assets/521d4e79-9b7f-4f4d-a4ab-312456f3e1b7" />

3)Analysis
<img width="1696" height="689" alt="image" src="https://github.com/user-attachments/assets/b611ca0b-93f9-404b-ab07-c8f4e4d5e7c4" />

 <h2>🧪 **Results Summary** </h2>

| Model                 | Result                           |
| --------------------- | -------------------------------- |
| **Linear Regression** | R² = 0.996                       |
| **Random Forest**     | Poor generalization              |
| **LSTM**              | **R² ≈ 0.84** (best deep model)  |
| **ARIMA**             | Stable long-term forecast        |
| **Agentic AI**        | Human-readable financial reports |

 <h2>🛠 **Tech Stack** </h2>

Languages: Python
Libraries: Pandas, NumPy, Matplotlib, Statsmodels, Scikit-Learn, TensorFlow/Keras
Models: ARIMA, LSTM
AI: Groq LLaMA-3.3 70B
Deployment: Streamlit Cloud
Version Control: GitHub
Environment: Google Colab + Streamlit Cloud

