import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC, SVR
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay,
    mean_absolute_error, mean_squared_error, r2_score
)

# ------------------ Caching ------------------
@st.cache_data
def load_example_data(name):
    return sns.load_dataset(name)

@st.cache_data
def preprocess_data(df, features, target):
    original_cols = features + [target]
    df = df[original_cols].copy()

    # Drop columns with >50% missing
    missing_thresh = 0.5
    cols_to_drop = df.columns[df.isnull().mean() > missing_thresh].tolist()
    if cols_to_drop:
        st.warning(f"Columns dropped due to excessive missing values: {cols_to_drop}")
        df.drop(columns=cols_to_drop, inplace=True)

    # Update feature list to exclude dropped ones
    features = [f for f in features if f in df.columns]
    if target not in df.columns:
        st.error("Target column was dropped due to excessive missing values.")
        st.stop()

    # Detect types
    categorical_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    numerical_cols = df.select_dtypes(include=["number"]).columns.tolist()

    # Label Encoding for categoricals
    encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = df[col].astype(str)
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    # Impute missing values
    if categorical_cols:
        df[categorical_cols] = SimpleImputer(strategy="most_frequent").fit_transform(df[categorical_cols])
    if numerical_cols:
        df[numerical_cols] = IterativeImputer().fit_transform(df[numerical_cols])

    # Scale features
    scaler = StandardScaler()
    df[features] = scaler.fit_transform(df[features])

    return df, encoders, scaler, features

@st.cache_resource
def train_model(_model, X_train, y_train):
    _model.fit(X_train, y_train)
    return _model

# ------------------ UI ------------------
st.title("🧠 Smart AutoML App")
st.markdown("Upload your dataset or use a sample. Choose features, model and get predictions easily.")

# ------------------ Load Data ------------------
st.sidebar.header("1. Data Source")
source = st.sidebar.radio("Select source:", ["Upload", "Example"])

if source == "Upload":
    uploaded_file = st.sidebar.file_uploader("Upload CSV/Excel file", type=["csv", "xlsx"])
    if uploaded_file:
        if uploaded_file.name.endswith("csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
    else:
        st.warning("Please upload a file.")
        st.stop()
else:
    dataset = st.sidebar.selectbox("Choose a dataset", ["iris", "titanic", "tips"])
    df = load_example_data(dataset)

st.subheader("📄 Dataset Preview")
st.dataframe(df.head())
st.write(f"Shape: {df.shape}")

# ------------------ Feature Selection ------------------
st.sidebar.header("2. Features and Target")
features = st.sidebar.multiselect("Select features", df.columns.tolist())
target = st.sidebar.selectbox("Select target", df.columns.tolist())

if not features or not target:
    st.warning("Please select both features and target to continue.")
    st.stop()

# ------------------ Problem Type ------------------
st.sidebar.header("3. Problem Type")
task = st.sidebar.radio("Choose task:", ["Classification", "Regression"])

# ------------------ Model Selection ------------------
st.sidebar.header("4. Model")
if task == "Regression":
    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree Regressor": DecisionTreeRegressor(max_depth=5),
        "Random Forest Regressor": RandomForestRegressor(n_estimators=100),
        "SVR": SVR()
    }
else:
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree Classifier": DecisionTreeClassifier(max_depth=5),
        "Random Forest Classifier": RandomForestClassifier(n_estimators=100),
        "SVC": SVC(probability=True)
    }

chosen_model_name = st.sidebar.selectbox("Select model", list(models.keys()))
model = models[chosen_model_name]

test_size = st.sidebar.slider("Test size (%)", 10, 50, 20) / 100

# ------------------ Train Model ------------------
if st.button("Train Model"):
    st.info("Preprocessing data and training...")
    df_clean, encoders, scaler, features = preprocess_data(df, features, target)
    X = df_clean[features]
    y = df_clean[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    model = train_model(model, X_train, y_train)
    y_pred = model.predict(X_test)

    st.success(f"Model Trained: {chosen_model_name}")
    st.subheader("🔍 Evaluation")

    if task == "Regression":
        st.metric("R² Score", round(r2_score(y_test, y_pred), 3))
        st.write({
            "MAE": round(mean_absolute_error(y_test, y_pred), 3),
            "MSE": round(mean_squared_error(y_test, y_pred), 3),
            "R2": round(r2_score(y_test, y_pred), 3)
        })
    else:
        acc = accuracy_score(y_test, y_pred)
        st.metric("Accuracy", f"{acc:.2%}")
        st.write({
            "Precision": round(precision_score(y_test, y_pred, average="weighted", zero_division=0), 3),
            "Recall": round(recall_score(y_test, y_pred, average="weighted"), 3),
            "F1 Score": round(f1_score(y_test, y_pred, average="weighted"), 3)
        })

        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        ConfusionMatrixDisplay(cm).plot(ax=ax)
        st.pyplot(fig)

    # ------------------ Feature Importance ------------------
    if hasattr(model, "feature_importances_"):
        st.subheader("📊 Feature Importance")
        fi = pd.Series(model.feature_importances_, index=features).sort_values()
        st.bar_chart(fi)

    # ------------------ Predict ------------------
    st.subheader("🔮 Make Predictions")
    pred_mode = st.radio("Prediction input method", ["Manual", "Upload CSV"])

    if pred_mode == "Manual":
        input_data = {f: st.number_input(f"Input for {f}", value=0.0) for f in features}
        input_df = pd.DataFrame([input_data])
    else:
        file = st.file_uploader("Upload prediction CSV", type=["csv"])
        if file:
            input_df = pd.read_csv(file)
        else:
            st.stop()

    try:
        input_df = pd.DataFrame(scaler.transform(input_df[features]), columns=features)
        prediction = model.predict(input_df)
        st.write("Predictions:", prediction)
    except Exception as e:
        st.error(f"Prediction failed: {e}")

    # ------------------ Download Model ------------------
    st.download_button("Download Model", pickle.dumps(model), file_name=f"{chosen_model_name}.pkl")

# ------------------ Tip ------------------
st.sidebar.markdown("---")
st.sidebar.info("📈 **Tip:** For better results, remove noisy features and ensure you handle missing values appropriately.")
