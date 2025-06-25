# Import libraries
import datetime
import warnings
from datetime import date, timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import statsmodels.api as sm
import streamlit as st
import yfinance as yf
from keras.layers import LSTM, Dense
from keras.models import Sequential
from prophet import Prophet
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller

warnings.filterwarnings("ignore")

# Set Streamlit layout
st.set_page_config(layout="wide", initial_sidebar_state="collapsed")

# Title and banner
st.title("Stock Market Forecasting App")
st.subheader(
    "This app is created to forecast the stock market price of the selected company."
)
st.image(
    "https://img.freepik.com/free-vector/gradient-stock-market-concept_23-2149166910.jpg"
)

# Sidebar
st.sidebar.header("Select the parameters from below")
start_date = st.sidebar.date_input("Start date", date(2020, 1, 1))
end_date = st.sidebar.date_input("End date", date(2020, 12, 31))

ticker_list = [
    "AAPL",
    "MSFT",
    "GOOG",
    "GOOGL",
    "META",
    "TSLA",
    "NVDA",
    "ADBE",
    "PYPL",
    "INTC",
    "CMCSA",
    "NFLX",
    "PEP",
]
ticker = st.sidebar.selectbox("Select the company", ticker_list)

# Fetch stock data
data = yf.download(ticker, start=start_date, end=end_date)

# Flatten column index if needed
if isinstance(data.columns, pd.MultiIndex):
    data.columns = [" ".join(col).strip() for col in data.columns]

data.insert(0, "Date", data.index)
data.reset_index(drop=True, inplace=True)

st.write("Data from", start_date, "to", end_date)
st.write(data)

# Visualization
st.header("Data Visualization")
st.subheader("Plot of the data")
st.write(
    "**Note:** Select your specific date range on the sidebar, or zoom in on the plot and select your specific column"
)

fig = px.line(
    data, x="Date", y=data.columns[1:], title="Stock Data", width=1000, height=600
)
st.plotly_chart(fig)

# Column selection
column = st.selectbox("Select the column to be used for forecasting", data.columns[1:])
data = data[["Date", column]]
st.write("Selected Data")
st.write(data)

# ADF Test
st.header("Is data Stationary?")
st.write(adfuller(data[column])[1] < 0.05)

# Decomposition
st.header("Decomposition of the data")
decomposition = seasonal_decompose(data[column], model="additive", period=12)
st.write(decomposition.plot())
st.plotly_chart(
    px.line(x=data["Date"], y=decomposition.trend, title="Trend").update_traces(
        line_color="Blue"
    )
)
st.plotly_chart(
    px.line(
        x=data["Date"], y=decomposition.seasonal, title="Seasonality"
    ).update_traces(line_color="Green")
)
st.plotly_chart(
    px.line(x=data["Date"], y=decomposition.resid, title="Residuals").update_traces(
        line_color="Red", line_dash="dot"
    )
)

# Model selection
models = ["SARIMA", "Random Forest", "LSTM", "Prophet"]
selected_model = st.sidebar.selectbox("Select the model for forecasting", models)

if selected_model == "SARIMA":
    p = st.slider("Select the value of p", 0, 5, 2)
    d = st.slider("Select the value of d", 0, 5, 1)
    q = st.slider("Select the value of q", 0, 5, 2)
    seasonal_order = st.number_input(
        "Select the value of seasonal p (period)", 0, 24, 12
    )

    model = sm.tsa.statespace.SARIMAX(
        data[column], order=(p, d, q), seasonal_order=(p, d, q, seasonal_order)
    )
    model = model.fit()

    st.header("Model Summary")
    st.write(model.summary())

    forecast_period = st.number_input(
        "Select the number of days to forecast", 1, 365, 10
    )

    predictions = model.get_prediction(
        start=len(data), end=len(data) + forecast_period - 1
    )
    predictions_mean = predictions.predicted_mean
    prediction_dates = pd.date_range(
        start=end_date + timedelta(days=1), periods=forecast_period, freq="D"
    )

    predicted_df = pd.DataFrame(
        {"Date": prediction_dates, "Predicted": predictions_mean.values}
    )

    # Show actual data table (last 10 rows)
    st.subheader("Actual Data")
    st.write(data.tail(10))

    # Show predicted data table
    st.subheader("Predicted Data")
    st.write(predicted_df)

    # Plot actual vs predicted
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=data["Date"],
            y=data[column],
            mode="lines",
            name="Actual",
            line=dict(color="blue"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=predicted_df["Date"],
            y=predicted_df["Predicted"],
            mode="lines",
            name="Predicted",
            line=dict(color="red"),
        )
    )
    fig.update_layout(
        title="Actual vs Predicted",
        xaxis_title="Date",
        yaxis_title="Price",
        width=1000,
        height=400,
    )
    st.plotly_chart(fig)

elif selected_model == "Random Forest":
    st.header("Random Forest Regression")

    train_size = int(len(data) * 0.8)
    train_data, test_data = data[:train_size], data[train_size:]

    train_X, train_y = train_data["Date"], train_data[column]
    test_X, test_y = test_data["Date"], test_data[column]

    rf_model = RandomForestRegressor(n_estimators=100, random_state=0)
    rf_model.fit(np.array(train_X.index).reshape(-1, 1), train_y.values)
    predictions = rf_model.predict(np.array(test_X.index).reshape(-1, 1))

    mse = mean_squared_error(test_y, predictions)
    st.write(f"Root Mean Squared Error (RMSE): {np.sqrt(mse)}")

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=data["Date"],
            y=data[column],
            mode="lines",
            name="Actual",
            line=dict(color="blue"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=test_data["Date"],
            y=predictions,
            mode="lines",
            name="Predicted",
            line=dict(color="red"),
        )
    )
    fig.update_layout(
        title="Actual vs Predicted (Random Forest)",
        xaxis_title="Date",
        yaxis_title="Price",
        width=1000,
        height=400,
    )
    st.plotly_chart(fig)

elif selected_model == "LSTM":
    st.header("Long Short-Term Memory (LSTM)")

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data[column].values.reshape(-1, 1))

    train_size = int(len(scaled_data) * 0.8)
    train_data, test_data = scaled_data[:train_size], scaled_data[train_size:]

    def create_sequences(dataset, seq_length):
        X, y = [], []
        for i in range(len(dataset) - seq_length):
            X.append(dataset[i : i + seq_length, 0])
            y.append(dataset[i + seq_length, 0])
        return np.array(X), np.array(y)

    seq_length = st.slider("Select sequence length", 1, 30, 10)
    train_X, train_y = create_sequences(train_data, seq_length)
    test_X, test_y = create_sequences(test_data, seq_length)

    train_X = np.reshape(train_X, (train_X.shape[0], train_X.shape[1], 1))
    test_X = np.reshape(test_X, (test_X.shape[0], test_X.shape[1], 1))

    lstm_model = Sequential()
    lstm_model.add(LSTM(50, return_sequences=True, input_shape=(train_X.shape[1], 1)))
    lstm_model.add(LSTM(50))
    lstm_model.add(Dense(1))

    lstm_model.compile(optimizer="adam", loss="mean_squared_error")
    lstm_model.fit(train_X, train_y, epochs=10, batch_size=16)

    test_predictions = scaler.inverse_transform(lstm_model.predict(test_X))
    test_dates = data["Date"][train_size + seq_length :]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=data["Date"],
            y=data[column],
            mode="lines",
            name="Actual",
            line=dict(color="blue"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=test_dates,
            y=test_predictions.flatten(),
            mode="lines",
            name="Predicted",
            line=dict(color="red"),
        )
    )
    fig.update_layout(
        title="Actual vs Predicted (LSTM)",
        xaxis_title="Date",
        yaxis_title="Price",
        width=1000,
        height=400,
    )
    st.plotly_chart(fig)

elif selected_model == "Prophet":
    st.header("Facebook Prophet")

    prophet_data = data.rename(columns={"Date": "ds", column: "y"})
    prophet_model = Prophet()
    prophet_model.fit(prophet_data)

    future = prophet_model.make_future_dataframe(periods=365)
    forecast = prophet_model.predict(future)

    fig = prophet_model.plot(forecast)
    plt.title("Forecast with Facebook Prophet")
    plt.xlabel("Date")
    plt.ylabel("Price")
    st.pyplot(fig)

st.write("Model selected:", selected_model)

# Footer CSS
st.markdown(
    """
    <style>
    .footer {
        position: fixed;
        left: 0;
        bottom: 0; 
        width: 100%;
        background-color: #f5f5f5;
        color: #000000;
        text-align: center;
        padding: 10px;
        z-index: 1000;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Your URLs
GITHUB_URL = "https://github.com/hammadhanif267"
LINKEDIN_URL = "https://www.linkedin.com/in/hammad-hanif-153a182bb/"

# Footer HTML
st.markdown(
    f"""
    <div class="footer">
        <b>Created by Hammad Hanif:</b> &nbsp;
        <a href="{GITHUB_URL}" target="_blank" rel="noopener noreferrer">
            <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/github/github-original.svg" width="30" alt="GitHub">
        </a> &nbsp;
        <a href="{LINKEDIN_URL}" target="_blank" rel="noopener noreferrer">
            <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/linkedin/linkedin-original.svg" width="30" alt="LinkedIn">
        </a>
    </div>
    """,
    unsafe_allow_html=True,
)

# Sidebar contact
st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    **Created By Hammad Hanif**:<br>
    <a href='https://github.com/hammadhanif267' target='_blank'>
        <i class='fab fa-github'></i> GitHub
    </a> &nbsp;|&nbsp;
    <a href='https://www.linkedin.com/in/hammad-hanif-153a182bb/' target='_blank'>
        <i class='fab fa-linkedin'></i> LinkedIn
    </a>
    """,
    unsafe_allow_html=True,
)
st.sidebar.markdown(
    '<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">',
    unsafe_allow_html=True,
)
st.sidebar.markdown(
    "<i class='fas fa-envelope'></i> <b>Contact Me</b>: <a href='mailto:hamadhanif267@gmail.com' target='_blank'>Email</a>",
    unsafe_allow_html=True,
)
