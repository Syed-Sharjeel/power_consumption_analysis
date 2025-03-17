import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
import streamlit as st

data = pd.read_csv(r'C:\Users\Syed_Sharjeel\Desktop\SS\DS\Elecricity Consumption\smart_meter_data.csv')
data = data.drop('Timestamp', axis=1)

encoder = LabelEncoder()
data['Anomaly_Label'] = encoder.fit_transform(data['Anomaly_Label'])

X = data.drop('Anomaly_Label', axis=1)
y = data['Anomaly_Label']

X_test, X_train, y_test, y_train = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=500)
model.fit(X_train, y_train)

st.title("Electrical Power Consumption Analysis")
with st.form('form'):
    electricity_consumed = st.number_input("Electricity Consumed (kWh)", placeholder='Electricity Comsumed')
    temperature = st.number_input("Temperature (°C)", placeholder='Temperature')
    humidity = st.number_input('Humidity (%)',placeholder='Humidity')
    wind_speed = st.number_input('Wind Speed (km/h)', placeholder='Wind Speed')
    avg_past_consumption = st.number_input('Average Past Electricity Consumption (kWh)', placeholder='Avg, Past Consumption')
    submitted = st.form_submit_button('Check Consumption Status')

if submitted:
    X_new = np.array([electricity_consumed, temperature, humidity, wind_speed, avg_past_consumption]).reshape(1,-1)
    result = model.predict(X_new)

    if result < 0.5:
        st.write("Abnormal Power Consumption")
    elif result < 0.5:
        st.write("Normal Power Consumption")    
