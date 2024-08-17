import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.title("Predict Your Car Price")

encode_dict = {
    "fuel_type" : {"Diesel": 1, "Petrol": 2, "CNG": 3, "LPG": 4, "Electric":5},
    "seller_type" : {"Dealer": 1, "Individual":2, "Trustmark Dealer":3},
    "transmission_type" : {"Manual": 1, "Automatic": 2}
}

fuel_type = st.selectbox("Select fuel type", ["Diesel", "Petrol", "CNG", "LPG", "Electric"])
transmission_type = st.selectbox("Select transmission type", ["Manual", "Automatic"])
engine = st.slider("Select engine power", 500, 5000, step=100)
seats = st.selectbox("Select # of seats", [2, 4, 5, 7, 9, 11])

with open("car_pred", "rb") as file:
    reg_model = pickle.load(file)

def model_predict(fuel_type, transmission_type, engine, seats):
        input_features = [[2018.0, 1, 4000, fuel_type, transmission_type, 19.70, engine, 86.30, seats]]
        return reg_model.predict(input_features)

if st.button("Predict Car Price", "primary"):
      fuel_type = encode_dict["fuel_type"][fuel_type]
      transmission_type = encode_dict["transmission_type"][transmission_type]
      price = model_predict(fuel_type, transmission_type, engine, seats)
      st.text("The price of the selected model is: " + str(np.round(price,2)) + " Lakhs")
    
