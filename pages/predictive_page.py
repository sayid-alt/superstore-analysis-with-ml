import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from superstore_analysis.datasets import DataLoader
from superstore_analysis.pipelines.inference_pipeline import ClassifyInfer

st.set_page_config(layout='wide')
st.header("Cluster Prediction")


class AltairDrawer:
	def __init__(self, df):
		self._df = df

	def draw_proba(self):
		base = alt.Chart(data_pred).mark_bar() \
				.encode(
				alt.X('Clusters:N'),
				alt.Y('Proba:Q'),
				color='Clusters:N',
				tooltip=[alt.Tooltip('Proba', format='.3f')],
				text=alt.Text('Proba:Q', format='.3f')
			)


		text = base.mark_text(
			align='center',
			dy=-10,
		)

		full_chart = (base + text).properties(
			width=400,
			title=alt.Title(
				text="Probabilites",
				subtitle="Higher probability is the chosen for cluster number"
			)
		)

		return full_chart


@st.cache_data
def get_data():
	data = DataLoader().from_gdrive()
	return data

data = get_data()

pred_box = st.columns(2)

with pred_box[0]:
	with st.form('prediction_form'):
		pred_placeholder = st.empty()

		cols = st.columns(2)
		with cols[0]:
			sales = st.number_input(
				label='Sales',
			)
			category = st.selectbox(
				label='Category',
				options=data['Category'].unique()
			)
			ship_mode = st.selectbox(
				label='Ship Mode',
				options=data['Ship_Mode'].unique()
			)

		with cols[1]:
			segment = st.selectbox(
				label='Segment',
				options=data['Segment'].unique()
			)
			region = st.selectbox(
				label='Region',
				options=data['Region'].unique()
			)

		submit = st.form_submit_button("Predict")

with pred_box[1]:
	if submit:
		X = pd.DataFrame(
			data=[[sales, category, ship_mode, segment, region]],
			columns=['Sales', 'Category', 'Ship_Mode', 'Segment', 'Region']
		)

		classifier = ClassifyInfer()
		prediction = classifier.infer(X)

		proba_pred = classifier.get_proba_prediction


		data_pred = pd.DataFrame({
				'Clusters': [i for i in range(len(proba_pred[0]))],
				'Proba': proba_pred[0],
			})


		drawer = AltairDrawer(data_pred)
		proba_chart = drawer.draw_proba()

		st.altair_chart(proba_chart, theme='streamlit')
