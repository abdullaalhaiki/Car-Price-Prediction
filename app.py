
import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Car Price Predictor", page_icon="🚗")

model = joblib.load("car_price_model.pkl")
categories = joblib.load("car_categories.pkl")

st.title("🚗 Car Price Prediction")
st.write("Enter the car details to estimate its Manufacturer's Suggested Retail Price (MSRP).")

with st.form("car_form"):
    make = st.selectbox("Manufacturer", categories["Make"])
    car_model = st.selectbox("Model", categories["Model"])
    year = st.number_input("Year", min_value=1990, max_value=2030, value=2016)

    fuel_type = st.selectbox("Engine Fuel Type", categories["Engine Fuel Type"])
    engine_hp = st.number_input("Engine Horsepower", min_value=0.0, value=200.0)
    cylinders = st.number_input("Engine Cylinders", min_value=0.0, value=4.0)

    transmission = st.selectbox("Transmission Type", categories["Transmission Type"])
    driven_wheels = st.selectbox("Driven Wheels", categories["Driven_Wheels"])
    doors = st.number_input("Number of Doors", min_value=0.0, value=4.0)

    market_category = st.selectbox(
        "Market Category",
        ["Missing"] + categories["Market Category"]
    )

    vehicle_size = st.selectbox("Vehicle Size", categories["Vehicle Size"])
    vehicle_style = st.selectbox("Vehicle Style", categories["Vehicle Style"])

    highway_mpg = st.number_input("Highway MPG", min_value=0, value=30)
    city_mpg = st.number_input("City MPG", min_value=0, value=22)
    popularity = st.number_input("Popularity", min_value=0, value=1000)

    submitted = st.form_submit_button("Predict Car Price")

if submitted:
    hp_per_cylinder = (
        engine_hp / cylinders if cylinders > 0 else np.nan
    )

    average_mpg = (city_mpg + highway_mpg) / 2

    car = pd.DataFrame([{
        "Make": make,
        "Model": car_model,
        "Year": year,
        "Engine Fuel Type": fuel_type,
        "Engine HP": engine_hp,
        "Engine Cylinders": cylinders,
        "Transmission Type": transmission,
        "Driven_Wheels": driven_wheels,
        "Number of Doors": doors,
        "Market Category": (
            np.nan if market_category == "Missing" else market_category
        ),
        "Vehicle Size": vehicle_size,
        "Vehicle Style": vehicle_style,
        "highway MPG": highway_mpg,
        "city mpg": city_mpg,
        "Popularity": popularity,
        "HP_per_Cylinder": hp_per_cylinder,
        "Average_MPG": average_mpg
    }])

    prediction = model.predict(car)[0]

    st.success(f"Estimated MSRP: ${prediction:,.2f}")
    st.caption("This is an estimated price, not a guaranteed market valuation.")
