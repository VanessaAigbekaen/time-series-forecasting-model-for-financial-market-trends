import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="SPY AI Market Dashboard",
    page_icon="📊",
    layout="wide"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.stApp {
    background-color: #080d18;
    color: #ffffff;
}

#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
header { visibility: hidden; }

.main-title {
    font-size: 38px;
    font-weight: 700;
    color: white;
    margin-bottom: 5px;
}

.subtitle {
    color: #8b95a7;
    font-size: 16px;
    margin-bottom: 30px;
}

.metric-card {
    background: linear-gradient(145deg, #101827, #0b1220);
    border: 1px solid #1d2a3a;
    border-radius: 15px;
    padding: 22px;
    box-shadow: 0px 5px 20px rgba(0,0,0,0.25);
    min-height: 150px;
}

.metric-title {
    color: #8b95a7;
    font-size: 14px;
    font-weight: 600;
}

.metric-value {
    color: white;
    font-size: 30px;
    font-weight: bold;
    margin-top: 10px;
}

.metric-green { color: #20e0a5; font-size: 16px; }
.metric-red { color: #ff5c70; font-size: 16px; }

.dashboard-card {
    background-color: #0e1624;
    border: 1px solid #1d2a3a;
    border-radius: 15px;
    padding: 20px;
    margin-top: 15px;
}

section[data-testid="stSidebar"] {
    background-color: #0b1220;
    border-right: 1px solid #1d2a3a;
}

.stButton > button {
    background: linear-gradient(90deg, #16c79a, #14a67f);
    color: white;
    border: none;
    border-radius: 10px;
    height: 50px;
    font-size: 16px;
    font-weight: bold;
    width: 100%;
}

.stButton > button:hover {
    background: linear-gradient(90deg, #14a67f, #16c79a);
    color: white;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# FEATURE ORDER — MUST MATCH THE NOTEBOOK EXACTLY
# =========================================================
# The model in best_spy_model.joblib was trained on these 8 columns,
# in this exact order: Close, High, Low, Open, Volume, TR, ATR, RSI.
# If you ever change the notebook's feature engineering, update this
# list (and FEATURE_COLUMNS below) to match, or predictions will fail
# or silently be wrong.
FEATURE_COLUMNS = ["Close", "High", "Low", "Open", "Volume", "TR", "ATR", "RSI"]


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():
    return joblib.load("best_spy_model.joblib1")


# =========================================================
# LOAD DATASET
# =========================================================

@st.cache_data
def load_data():

    df = pd.read_csv("SPY_data.csv", skiprows=2)

    df.columns = ["Date", "Close", "High", "Low", "Open", "Volume"]

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    numeric_columns = ["Close", "High", "Low", "Open", "Volume"]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna()
    df = df.sort_values("Date")

    return df


# =========================================================
# TECHNICAL INDICATORS
# =========================================================

def add_indicators(df):

    df = df.copy()

    df["Previous_Close"] = df["Close"].shift(1)

    df["TR"] = np.maximum.reduce([
        df["High"] - df["Low"],
        abs(df["High"] - df["Previous_Close"]),
        abs(df["Low"] - df["Previous_Close"])
    ])

    df["ATR"] = df["TR"].rolling(window=14, min_periods=1).mean()

    change = df["Close"].diff()

    gain = change.where(change > 0, 0)
    loss = -change.where(change < 0, 0)

    average_gain = gain.rolling(window=14, min_periods=14).mean()
    average_loss = loss.rolling(window=14, min_periods=14).mean()

    rs = average_gain / average_loss
    df["RSI"] = 100 - (100 / (1 + rs))

    df = df.dropna()

    return df


# =========================================================
# PREDICTION
# =========================================================

def predict_market(df, model):

    latest = df.iloc[-1]

    features = np.array([[latest[col] for col in FEATURE_COLUMNS]])

    expected = getattr(model, "n_features_in_", None)
    if expected is not None and features.shape[1] != expected:
        st.error(
            f"Feature mismatch: the app is sending {features.shape[1]} features "
            f"({', '.join(FEATURE_COLUMNS)}), but the loaded model expects "
            f"{expected}. Retrain the notebook and re-save best_spy_model.joblib "
            f"so the feature counts match, then restart this app."
        )
        st.stop()

    prediction = model.predict(features)[0]

    confidence = None
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features)[0]
        confidence = np.max(probabilities) * 100

    return prediction, confidence


# =========================================================
# RISK CALCULATION
# =========================================================

def calculate_risk(df):

    latest_atr = df["ATR"].iloc[-1]
    latest_close = df["Close"].iloc[-1]

    volatility = (latest_atr / latest_close) * 100

    if volatility < 1:
        risk = "LOW"
        description = "Low market volatility"
    elif volatility < 2:
        risk = "MEDIUM"
        description = "Moderate market volatility"
    else:
        risk = "HIGH"
        description = "High market volatility"

    return risk, description, volatility


# =========================================================
# LOAD EVERYTHING
# =========================================================

try:
    model = load_model()
    df = load_data()
    df = add_indicators(df)
except Exception as e:
    st.error(f"Error loading project files: {e}")
    st.stop()


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown("# 📊 MarketAI")
    st.markdown("---")

    page = st.radio(
        "Navigation",
        ["Dashboard", "Market Charts", "Technical Indicators", "About"]
    )

    st.markdown("---")
    st.markdown("### 🤖 AI Prediction System")
    st.write("Machine learning analysis of historical SPY market data.")

    st.markdown("---")
    st.caption("Educational purposes only. Not financial advice.")


# =========================================================
# DASHBOARD
# =========================================================

if page == "Dashboard":

    st.markdown('<div class="main-title">Market Forecast Dashboard</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">AI-powered SPY market analysis and prediction system</div>',
        unsafe_allow_html=True
    )

    prediction, confidence = predict_market(df, model)

    if prediction == 1:
        direction = "📈 UP"
        direction_text = "Bullish prediction"
        direction_colour = "metric-green"
    else:
        direction = "📉 DOWN"
        direction_text = "Bearish prediction"
        direction_colour = "metric-red"

    risk, risk_description, volatility = calculate_risk(df)

    latest_close = df["Close"].iloc[-1]
    latest_rsi = df["RSI"].iloc[-1]
    latest_atr = df["ATR"].iloc[-1]

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">MARKET DIRECTION</div>
            <div class="metric-value">{direction}</div>
            <div class="{direction_colour}">{direction_text}</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        confidence_display = f"{confidence:.1f}%" if confidence is not None else "N/A"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">MODEL CONFIDENCE</div>
            <div class="metric-value">{confidence_display}</div>
            <div class="metric-green">Prediction probability</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">RISK LEVEL</div>
            <div class="metric-value">⚠️ {risk}</div>
            <div class="metric-green">{volatility:.2f}% volatility</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">LATEST SPY PRICE</div>
            <div class="metric-value">${latest_close:.2f}</div>
            <div class="metric-green">Latest available data</div>
        </div>
        """, unsafe_allow_html=True)

    st.write("")

    left, right = st.columns([2, 1])

    with left:
        st.markdown("### 📈 Market Performance")

        recent_data = df.tail(60)

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=recent_data["Date"],
                y=recent_data["Close"],
                mode="lines",
                name="SPY Close Price",
                line=dict(color="#20e0a5", width=3),
                fill="tozeroy"
            )
        )
        fig.update_layout(
            paper_bgcolor="#0e1624",
            plot_bgcolor="#0e1624",
            font=dict(color="white"),
            height=400,
            margin=dict(l=20, r=20, t=20, b=20),
            xaxis=dict(showgrid=False),
            yaxis=dict(gridcolor="#1d2a3a")
        )
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.markdown("### 📊 Technical Indicators")

        st.metric("RSI", f"{latest_rsi:.2f}")

        if latest_rsi > 70:
            st.error("Overbought")
        elif latest_rsi < 30:
            st.success("Oversold")
        else:
            st.info("Neutral Range")

        st.metric("Average True Range", f"{latest_atr:.2f}")
        st.metric("Volatility", f"{volatility:.2f}%")

    st.markdown("### 🤖 AI Market Analysis")

    if prediction == 1:
        analysis = (
            "The machine learning model predicts that the market is more likely "
            "to move UP based on the latest historical market data and technical indicators."
        )
    else:
        analysis = (
            "The machine learning model predicts that the market is more likely "
            "to move DOWN based on the latest historical market data and technical indicators."
        )

    st.markdown(
        f"""
        <div class="dashboard-card">
        {analysis}
        <br><br>
        <b>Risk:</b> {risk_description}
        </div>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# MARKET CHARTS
# =========================================================

elif page == "Market Charts":

    st.title("📈 SPY Market Charts")

    chart_data = df.tail(100)

    fig = go.Figure(
        data=[
            go.Candlestick(
                x=chart_data["Date"],
                open=chart_data["Open"],
                high=chart_data["High"],
                low=chart_data["Low"],
                close=chart_data["Close"],
                name="SPY"
            )
        ]
    )

    fig.update_layout(
        title="SPY Candlestick Chart",
        paper_bgcolor="#080d18",
        plot_bgcolor="#0e1624",
        font=dict(color="white"),
        height=600,
        xaxis_rangeslider_visible=False
    )

    st.plotly_chart(fig, use_container_width=True)


# =========================================================
# TECHNICAL INDICATORS
# =========================================================

elif page == "Technical Indicators":

    st.title("📊 Technical Indicators")

    chart_data = df.tail(100)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=chart_data["Date"],
            y=chart_data["RSI"],
            mode="lines",
            name="RSI",
            line=dict(color="#9b5cff", width=3)
        )
    )
    fig.add_hline(y=70, line_dash="dash", line_color="red")
    fig.add_hline(y=30, line_dash="dash", line_color="green")

    fig.update_layout(
        title="Relative Strength Index (RSI)",
        paper_bgcolor="#080d18",
        plot_bgcolor="#0e1624",
        font=dict(color="white"),
        height=450
    )
    st.plotly_chart(fig, use_container_width=True)

    fig_atr = go.Figure()
    fig_atr.add_trace(
        go.Scatter(
            x=chart_data["Date"],
            y=chart_data["ATR"],
            mode="lines",
            name="ATR",
            line=dict(color="#20e0a5", width=3)
        )
    )
    fig_atr.update_layout(
        title="Average True Range (ATR)",
        paper_bgcolor="#080d18",
        plot_bgcolor="#0e1624",
        font=dict(color="white"),
        height=450
    )
    st.plotly_chart(fig_atr, use_container_width=True)


# =========================================================
# ABOUT
# =========================================================

elif page == "About":

    st.title("About This Project")

    st.write("""
    This project uses machine learning and historical SPY
    market data to analyse technical indicators and predict
    potential market direction.

    Technical indicators include:

    - True Range (TR)
    - Average True Range (ATR)
    - Relative Strength Index (RSI)

    The project compares machine learning models and uses
    the best-performing model for prediction.
    """)

    st.warning(
        "This project is for educational purposes only and "
        "does not provide financial advice."
    )
