import streamlit as st
import altair as alt
from superstore_analysis.pipelines.inference_pipeline import OrderPredictionInfer
from superstore_analysis.datasets import DataLoader

st.header("Order Count Prediction")
"""
The main objective of this prediction is to discover which day that might have the highest order count in the next N days,
It is important to know the fartest day that the model will predict might decrease the accuracy of the prediction,
because the model will predict the next day based on the previous day, 
and if the previous day is not accurate, the next day will be even more inaccurate.
"""

N_NEXT = 10


@st.cache_data
def get_data():
	data_loader = DataLoader()
	return data_loader.from_gdrive()

def predict(data):
	regressor = OrderPredictionInfer()
	return regressor.infer(data, n_next=N_NEXT)

data = get_data()

if st.button('Predict', type='primary'):
	data_predicts = predict(data)
	next_n_days_data = data_predicts.iloc[-N_NEXT:]
	st.dataframe(next_n_days_data)
