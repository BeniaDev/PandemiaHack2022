import pandas as pd
import streamlit as st

LAT_LON = ['lat', 'lon']


df = pd.read_csv('data/covid_data_train.csv')
df['lon'] = df['lng']

st.title("Hi Anber! Welcome to Streamlit!")

st.write("My first DataFrame")



st.write(
    df.iloc[:6]
)
st.map(df)