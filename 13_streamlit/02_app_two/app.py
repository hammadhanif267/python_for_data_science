import hashlib
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import requests

# ------------------ Helper Functions ------------------

def get_df_hash(df):
    return hashlib.md5(pd.util.hash_pandas_object(df, index=True).values).hexdigest()

def load_builtin_dataset(name):
    return sns.load_dataset(name)

def load_uploaded_file(uploaded_file):
    if uploaded_file.name.endswith('.csv'):
        return pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith('.xlsx'):
        return pd.read_excel(uploaded_file)
    else:
        return None

def filter_dataframe(df):
    with st.sidebar.expander("🔎 Filter Data", expanded=False):
        filter_cols = st.multiselect("Select columns to filter", df.columns)
        for col in filter_cols:
            if pd.api.types.is_numeric_dtype(df[col]):
                min_val, max_val = float(df[col].min()), float(df[col].max())
                selected = st.slider(f"Filter {col}", min_val, max_val, (min_val, max_val))
                df = df[df[col].between(selected[0], selected[1])]
            elif pd.api.types.is_categorical_dtype(df[col]) or df[col].dtype == object:
                options = df[col].dropna().unique().tolist()
                selected = st.multiselect(f"Filter {col}", options, default=options)
                df = df[df[col].isin(selected)]
    return df

def download_csv(df, label="⬇️ Download CSV", filename="data.csv"):
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(label, csv, file_name=filename, mime="text/csv")

# ------------------ Weather Fetch Function ------------------

def get_weather_data(city):
    api_key = "2fd698a51f38f72de6882765f9019f97"
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        temp = data['main']['temp']
        description = data['weather'][0]['description']
        humidity = data['main']['humidity']
        return {
            'City': city,
            'Temperature (°C)': temp,
            'Weather': description,
            'Humidity (%)': humidity
        }
    except requests.RequestException as e:
        return f"❌ Failed to get weather: {e}"

# ------------------ App UI ------------------

st.set_page_config(page_title="Online Data Collection App", layout="wide")
st.markdown("""
    <div style="
        border: 6px solid #1976D2;
        border-radius: 20px;
        padding: 0px;
        background-color: #f5faff;
        margin-bottom: 18px;
        text-align: center;
        box-sizing: inherit;">
        <h1 style='color:#1976D2; margin-bottom: -1.5em; font-weight: 720;'>📊 Online Weather Forecast Data Collection Application</h1>
        <hr style="border:1px solid #ddd;">
    </div>
""", unsafe_allow_html=True)

# ------------------ Sidebar ------------------

st.sidebar.header("📁 Dataset Options")
dataset_options = sns.get_dataset_names()
selected_dataset = st.sidebar.selectbox('Built-in dataset:', dataset_options)
uploaded_file = st.sidebar.file_uploader('Or upload your custom dataset', type=['csv', 'xlsx'])

# Sidebar weather input
st.sidebar.header("🌤️ Real-Time Weather Data")
sidebar_city = st.sidebar.text_input("Enter your city name", "New York", key="sidebar_city_input")

# Store city name in session_state to use across pages
if sidebar_city:
    st.session_state["selected_city"] = sidebar_city
    weather_data = get_weather_data(sidebar_city)
    st.session_state["weather_data"] = weather_data
else:
    st.session_state["weather_data"] = None

# Show weather on sidebar
if isinstance(st.session_state.get("weather_data"), dict):
    weather = st.session_state["weather_data"]
    st.sidebar.metric("🌡️ Temp (°C)", f"{weather['Temperature (°C)']} °C")
    st.sidebar.metric("💧 Humidity", f"{weather['Humidity (%)']}%")
    st.sidebar.caption(f"**{weather['Weather'].capitalize()}** in {weather['City']}")
elif isinstance(st.session_state["weather_data"], str):
    st.sidebar.error(st.session_state["weather_data"])

# ------------------ Load Dataset ------------------

if 'df' not in st.session_state or 'df_hash' not in st.session_state:
    df = load_builtin_dataset(selected_dataset)
    st.session_state.df = df
    st.session_state.df_hash = get_df_hash(df)

if uploaded_file:
    df_new = load_uploaded_file(uploaded_file)
    if df_new is not None:
        st.session_state.df = df_new
        st.session_state.df_hash = get_df_hash(df_new)
else:
    df_builtin = load_builtin_dataset(selected_dataset)
    df_builtin_hash = get_df_hash(df_builtin)
    if df_builtin_hash != st.session_state.df_hash:
        st.session_state.df = df_builtin
        st.session_state.df_hash = df_builtin_hash

df = st.session_state.df.copy()
current_hash = get_df_hash(df)

with st.sidebar:
    st.markdown("### 📌 Dataset Status")
    if current_hash != st.session_state.df_hash:
        st.session_state.df_hash = current_hash
        st.info("Update detected in the dataset.")
    else:
        st.success("No updates detected.")

page = st.sidebar.radio("🔹 Navigation", ["Overview", "Visualization", "Data Cleaning", "Real-Time Weather"])

# ------------------ Main Pages ------------------

if page == "Overview":
    st.header("📋 Dataset Overview")
    filtered_df = filter_dataframe(df)
    st.dataframe(filtered_df, use_container_width=True)
    st.markdown(f"**🔢 Rows:** {filtered_df.shape[0]} &nbsp;&nbsp; **🔠 Columns:** {filtered_df.shape[1]}")
    st.markdown("### 📋 Column Types:")
    st.write(filtered_df.dtypes)

    if filtered_df.isnull().sum().sum() > 0:
        st.warning('⚠️ Null Values Found:')
        st.write(filtered_df.isnull().sum().sort_values(ascending=False))
    else:
        st.success('✅ No Null Values')

    st.markdown("### 📊 Summary Statistics:")
    st.write(filtered_df.describe())
    download_csv(filtered_df, "⬇️ Download Filtered Dataset", "filtered_data.csv")

    if isinstance(st.session_state["weather_data"], dict):
        st.markdown("### 🌍 Real-Time Weather Data")
        st.dataframe(pd.DataFrame([st.session_state["weather_data"]]))
        download_csv(pd.DataFrame([st.session_state["weather_data"]]), "⬇️ Download Weather Data", "weather_data.csv")

elif page == "Visualization":
    st.header("📈 Data Visualization")
    filtered_df = filter_dataframe(df)
    numeric_cols = filtered_df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = filtered_df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

    viz_type = st.selectbox("Select Visualization Type", ["Pairplot", "Heatmap", "Histogram", "Scatter", "Boxplot"])

    if viz_type == "Pairplot":
        hue_col = st.selectbox('Hue (color grouping):', ["None"] + categorical_cols)
        with st.spinner("Generating pairplot..."):
            fig = sns.pairplot(filtered_df, hue=hue_col if hue_col != "None" else None)
            st.pyplot(fig.figure)

    elif viz_type == "Heatmap":
        selected_cols = st.multiselect("Select numeric columns for heatmap", numeric_cols, default=numeric_cols)
        if len(selected_cols) >= 2:
            corr_matrix = filtered_df[selected_cols].corr()
            fig, ax = plt.subplots(figsize=(1.5*len(selected_cols), 1.2*len(selected_cols)))
            sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='viridis', ax=ax)
            st.pyplot(fig)
        else:
            st.info("Select at least two numerical columns for heatmap.")

    elif viz_type == "Histogram":
        col = st.selectbox("Select column for histogram", numeric_cols)
        bins = st.slider("Number of bins", 5, 100, 20)
        fig, ax = plt.subplots()
        ax.hist(filtered_df[col].dropna(), bins=bins, color='skyblue', edgecolor='black')
        ax.set_title(f'Histogram of {col}')
        st.pyplot(fig)

    elif viz_type == "Scatter":
        x_col = st.selectbox("X-axis", numeric_cols)
        y_col = st.selectbox("Y-axis", numeric_cols, index=1 if len(numeric_cols) > 1 else 0)
        color_col = st.selectbox("Color by", ["None"] + categorical_cols)
        fig = go.Figure()
        if color_col != "None":
            for cat in filtered_df[color_col].dropna().unique():
                df_cat = filtered_df[filtered_df[color_col] == cat]
                fig.add_trace(go.Scatter(x=df_cat[x_col], y=df_cat[y_col], mode='markers', name=str(cat)))
        else:
            fig.add_trace(go.Scatter(x=filtered_df[x_col], y=filtered_df[y_col], mode='markers'))
        fig.update_layout(title=f"{y_col} vs {x_col}")
        st.plotly_chart(fig, use_container_width=True)

    elif viz_type == "Boxplot":
        col = st.selectbox("Select column for boxplot", numeric_cols)
        group_col = st.selectbox("Group by (optional)", ["None"] + categorical_cols)
        fig, ax = plt.subplots()
        if group_col != "None":
            sns.boxplot(x=filtered_df[group_col], y=filtered_df[col], ax=ax)
        else:
            sns.boxplot(y=filtered_df[col], ax=ax)
        ax.set_title(f'Boxplot of {col}')
        st.pyplot(fig)

elif page == "Data Cleaning":
    st.header("🧹 Data Cleaning Tools")
    st.markdown("### 🔍 Preview of Data:")
    st.dataframe(df.head(), use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Drop Rows with Nulls"):
            df = df.dropna()
            st.session_state.df = df
            st.success("Dropped rows with null values.")
    with col2:
        fill_col = st.selectbox("🧯 Fill Nulls in Column", ["None"] + df.columns.tolist())
        if fill_col != "None":
            fill_val = st.text_input(f"Fill nulls in `{fill_col}` with:")
            if st.button("✅ Fill Nulls"):
                df[fill_col] = df[fill_col].fillna(fill_val)
                st.session_state.df = df
                st.success(f"Filled nulls in `{fill_col}`.")

    st.markdown("### 🔄 Convert Column Data Type:")
    dtype_col = st.selectbox("Column to Convert", ["None"] + df.columns.tolist())
    dtype_type = st.selectbox("Target Type", ["None", "int", "float", "str", "category"])
    if dtype_col != "None" and dtype_type != "None":
        if st.button("🔁 Convert Type"):
            try:
                if dtype_type == "int":
                    converted = pd.to_numeric(df[dtype_col], errors='coerce').astype('Int64')
                    df[dtype_col] = converted
                    st.warning(f"{converted.isna().sum()} entries could not be converted to int.")
                elif dtype_type == "float":
                    converted = pd.to_numeric(df[dtype_col], errors='coerce').astype(float)
                    df[dtype_col] = converted
                    st.warning(f"{converted.isna().sum()} entries could not be converted to float.")
                elif dtype_type == "str":
                    df[dtype_col] = df[dtype_col].astype(str)
                elif dtype_type == "category":
                    df[dtype_col] = df[dtype_col].astype('category')
                st.session_state.df = df
                st.success(f"Converted `{dtype_col}` to `{dtype_type}`.")
            except Exception as e:
                st.error(f"Conversion failed: {e}")

    st.markdown("### 💾 Download Cleaned Data:")
    download_csv(df, "⬇️ Download Cleaned Data", "cleaned_data.csv")

elif page == "Real-Time Weather":
    st.header("🌦️ Real-Time Weather Data")
    if isinstance(st.session_state["weather_data"], dict):
        st.dataframe(pd.DataFrame([st.session_state["weather_data"]]), use_container_width=True)
        download_csv(pd.DataFrame([st.session_state["weather_data"]]), "⬇️ Download Weather Data", "weather_data.csv")
    elif isinstance(st.session_state["weather_data"], str):
        st.error(st.session_state["weather_data"])
    else:
        st.info("Enter a city in the sidebar to fetch weather data.")
