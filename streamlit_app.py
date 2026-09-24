import pickle
import numpy as np
import streamlit as st

st.set_page_config(page_title="FWI Prediction", page_icon="🔥")


@st.cache_resource
def load_model():
    ridge_model = pickle.load(open('models/ridge.pkl', 'rb'))
    standard_scaler = pickle.load(open('models/scaler.pkl', 'rb'))
    return ridge_model, standard_scaler


ridge_model, standard_scaler = load_model()

st.title("🔥 Forest Fire Weather Index (FWI) Prediction")
st.write("Enter the weather and fire index readings below to predict the FWI.")

col1, col2 = st.columns(2)

with col1:
    Temperature = st.number_input("Temperature (°C)", value=29.0)
    RH = st.number_input("Relative Humidity (%)", value=57.0)
    Ws = st.number_input("Wind Speed (km/h)", value=18.0)
    Rain = st.number_input("Rain (mm)", value=0.0)
    FFMC = st.number_input("FFMC", value=65.7)

with col2:
    DMC = st.number_input("DMC", value=3.4)
    ISI = st.number_input("ISI", value=1.3)
    Classes = st.selectbox("Classes", [0, 1], help="0 = not fire, 1 = fire")
    Region = st.selectbox("Region", [0, 1], help="0 = Bejaia, 1 = Sidi-Bel Abbes")

if st.button("Predict FWI"):
    # Feature order must match training: Temperature, RH, Ws, Rain, FFMC, DMC, ISI, Classes, Region
    input_data = np.array([[Temperature, RH, Ws, Rain, FFMC, DMC, ISI, Classes, Region]])
    scaled_data = standard_scaler.transform(input_data)
    prediction = ridge_model.predict(scaled_data)
    st.success(f"Predicted FWI: {round(prediction[0], 2)}")
