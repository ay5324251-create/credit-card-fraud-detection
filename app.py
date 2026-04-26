import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="AI Credit Card Fraud Detection", layout="wide")


@st.cache_data
def load_data():
    return pd.read_csv("creditcard.csv")


@st.cache_resource
def train_model(df):
    X = df.drop("Class", axis=1)
    y = df["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced"
    )
    model.fit(X_train, y_train)
    return model, X.columns


df = load_data()
model, feature_columns = train_model(df)

st.title("💳 AI Credit Card Fraud Detection System")
st.write("Enter transaction details and predict whether it is Fraud or Normal.")

st.sidebar.header("Transaction Input")

user_input = {}

for col in feature_columns:
    if col == "Time":
        user_input[col] = st.sidebar.number_input("Time", value=0.0)
    elif col == "Amount":
        user_input[col] = st.sidebar.number_input("Amount", value=100.0)
    else:
        user_input[col] = st.sidebar.number_input(col, value=0.0, format="%.4f")

input_df = pd.DataFrame([user_input])

if st.sidebar.button("Predict Fraud"):
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("🚨 Fraud Detected")
        st.write(f"Fraud Probability: **{probability:.2%}**")
    else:
        st.success("✅ Normal Transaction")
        st.write(f"Fraud Probability: **{probability:.2%}**")

st.subheader("Dataset Preview")
st.dataframe(df.head())

st.subheader("Fraud Distribution")
st.bar_chart(df["Class"].value_counts())