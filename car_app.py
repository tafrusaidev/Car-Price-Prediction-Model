
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# --- Load and Preprocess Data (to match notebook state) ---
df = pd.read_csv('/content/car data.csv')
df_proc = df.drop(['Car_Name'], axis=1)
df_proc = pd.get_dummies(df_proc, drop_first=True)

X = df_proc.drop(['Selling_Price'], axis=1)
y = df_proc['Selling_Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# --- Train Model ---
model_rf = RandomForestRegressor(n_estimators=100, random_state=42)
model_rf.fit(X_train, y_train)

# --- Streamlit UI ---
st.title("🚗 Car Price Predictor")
st.write("Enter car details to estimate the selling price based on the Random Forest model.")

col1, col2 = st.columns(2)

with col1:
    year = st.number_input("Year", min_value=2000, max_value=2024, value=2015)
    present_price = st.number_input("Present Price (Lakhs)", min_value=0.0, value=5.0)
    kms_driven = st.number_input("Kms Driven", min_value=0, value=20000)
    owner = st.selectbox("Owner Type", [0, 1, 3])

with col2:
    fuel = st.selectbox("Fuel Type", ['Petrol', 'Diesel', 'CNG'])
    seller = st.selectbox("Seller Type", ['Dealer', 'Individual'])
    trans = st.selectbox("Transmission", ['Manual', 'Automatic'])

# Prepare input for prediction
# Logic follows the get_dummies(drop_first=True) structure
fuel_diesel = 1 if fuel == 'Diesel' else 0
fuel_petrol = 1 if fuel == 'Petrol' else 0
seller_ind = 1 if seller == 'Individual' else 0
trans_man = 1 if trans == 'Manual' else 0

input_df = pd.DataFrame([[year, present_price, kms_driven, owner, fuel_diesel, fuel_petrol, seller_ind, trans_man]],
                        columns=X.columns)

if st.button("Predict Price"):
    prediction = model_rf.predict(input_df)[0]
    st.success(f"Estimated Selling Price: ₹{prediction:.2f} Lakhs")
