import streamlit as st
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

DATA_PATH = Path(__file__).resolve().parent / "car data.csv"


@st.cache_data
def load_dataset(path: Path):
    if not path.exists() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Dataset not found or empty: {path}")

    df = pd.read_csv(path)

    required = {"Year", "Selling_Price", "Present_Price", "Kms_Driven", "Fuel_Type", "Seller_Type", "Transmission", "Owner"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in dataset: {sorted(missing)}")

    df_proc = df.drop(columns=["Car_Name"], errors="ignore")
    df_proc = pd.get_dummies(df_proc, drop_first=True)

    X = df_proc.drop(columns=["Selling_Price"], errors="ignore")
    y = df_proc["Selling_Price"]

    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=0)

    return X, y, X_train, y_train


@st.cache_resource
def train_model(X_train, y_train):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model


def build_input_frame(feature_columns):
    year = st.session_state.year
    present_price = st.session_state.present_price
    kms_driven = st.session_state.kms_driven
    owner = st.session_state.owner
    fuel = st.session_state.fuel
    seller = st.session_state.seller
    trans = st.session_state.trans

    row = {col: 0 for col in feature_columns}
    row["Year"] = year
    row["Present_Price"] = present_price
    row["Kms_Driven"] = kms_driven
    row["Owner"] = owner

    if "Fuel_Type_Diesel" in row:
        row["Fuel_Type_Diesel"] = 1 if fuel == "Diesel" else 0
    if "Fuel_Type_Petrol" in row:
        row["Fuel_Type_Petrol"] = 1 if fuel == "Petrol" else 0
    if "Seller_Type_Individual" in row:
        row["Seller_Type_Individual"] = 1 if seller == "Individual" else 0
    if "Transmission_Manual" in row:
        row["Transmission_Manual"] = 1 if trans == "Manual" else 0

    return pd.DataFrame([row], columns=feature_columns)


st.title("🚗 Car Price Predictor")
st.write("Enter car details to estimate the selling price based on the Random Forest model.")

try:
    X, y, X_train, y_train = load_dataset(DATA_PATH)
    model = train_model(X_train, y_train)
    feature_columns = list(X.columns)

    col1, col2 = st.columns(2)

    with col1:
        st.session_state.year = st.number_input("Year", min_value=2000, max_value=2024, value=2015)
        st.session_state.present_price = st.number_input("Present Price (Lakhs)", min_value=0.0, value=5.0)
        st.session_state.kms_driven = st.number_input("Kms Driven", min_value=0, value=20000)
        st.session_state.owner = st.selectbox("Owner Type", [0, 1, 3])

    with col2:
        st.session_state.fuel = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG"])
        st.session_state.seller = st.selectbox("Seller Type", ["Dealer", "Individual"])
        st.session_state.trans = st.selectbox("Transmission", ["Manual", "Automatic"])

    if st.button("Predict Price"):
        input_df = build_input_frame(feature_columns)
        prediction = model.predict(input_df)[0]
        st.success(f"Estimated Selling Price: ₹{prediction:.2f} Lakhs")

except FileNotFoundError as e:
    st.error(str(e))
    st.info("Place the dataset file named 'car data.csv' in the same folder as this app and make sure it contains the car data.")
except ValueError as e:
    st.error(str(e))
except Exception as e:
    st.error(f"Unexpected error: {e}")