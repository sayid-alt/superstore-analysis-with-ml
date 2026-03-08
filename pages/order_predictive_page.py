import pathlib
import dill as pickle
import streamlit as st
import numpy as np
import altair as alt
import pandas as pd
from superstore_analysis.pipelines.inference_pipeline import OrderPredictionInfer
from superstore_analysis.pipelines.training_pipeline import OrderPredictivePipeline
from superstore_analysis.datasets import DataLoader


script_dir = pathlib.Path(__file__).parent.resolve()


with st.sidebar:
	st.title("Superstore Analysis")
	st.image(script_dir.parent / "images/logo_store.png")  # Replace with the actual path to your logo
	st.markdown("""
	## Welcome to the Superstore Analysis App! 🙋🏻‍♂️
	\nThis application provides insights and predictions based on the Superstore dataset. 
	Use the navigation menu at the top to explore different pages, including order count prediction, cluster analysis, and cluster prediction.
	""")

	N_NEXT = st.number_input("Number of Days to Predict", min_value=1, value=10)

@st.cache_data
def get_data():
	data_loader = DataLoader()
	return data_loader.from_gdrive()

@st.cache_resource
def load_model(path=None):
	model_path = path if path else script_dir.parents[0] / "models/regression_model.pkl"

	with open(model_path, 'rb') as f:
		model = pickle.load(f)

	return model

def predict(data):
	regressor = OrderPredictionInfer()
	return regressor.infer(data, n_next=N_NEXT)


def plot_line_chart(data, var_1, var_2, n_next):
    """
    Creates a flexible multi-line chart for comparing two variables.
    
    Args:
        data (pd.DataFrame): The input dataframe.
        var_1 (str): Name of the first column.
        var_2 (str): Name of the second column.
        n_next (int): Number of days for the title.
    """
    # 1. Reset index to make 'index' a column for the x-axis
    chart_data = data.reset_index()

    # 2. Use transform_fold with the variable names provided
    final_chart = alt.Chart(chart_data).transform_fold(
        [var_1, var_2],
        as_=['Category', 'Value']
    ).mark_line().encode(
        x=alt.X('index:N', title='Days'),
        y=alt.Y('Value:Q', title='Order Count'),
        color=alt.Color('Category:N',
                        scale=alt.Scale(
                            domain=[var_1, var_2],
                            range=['#1f77b4', 'orange']
                        ),
                        legend=alt.Legend(title=None, orient='top', offset=10)
                        ),
        tooltip=['index:Q', 'Value:Q', 'Category:N']
    ).properties(
        title=f'{n_next} Days {var_2} vs {var_1}',
        width='container'  # Makes it responsive in Streamlit
    )

    return final_chart


def draw_bar_chart_comparison(data):
    # Melt the data for long-form format (required for Altair grouping)
    data_melted = pd.melt(
        data, 
        id_vars=['Days'], 
        value_vars=['Previous', 'Predictions'], 
        var_name='Type', 
        value_name='Order Count'
    )
    
    bar_chart = alt.Chart(data_melted).mark_bar().encode(
        # The main category on the X-axis
        x=alt.X('Days:N', title='Days'),
        
        # The value on the Y-axis
        y=alt.X('Order Count:Q', title='Order Count'),
        
        # This creates the grouping/unstacking effect
        xOffset='Type:N',
        
        # Color bars by Type
        color=alt.Color('Type:N', scale=alt.Scale(
            domain=['Predictions', 'Previous'],
            # Orange for Predictions, Blue for Previous
            range=['orange', '#1f77b0']
        )),
        
        tooltip=['Days:N', 'Order Count:Q', 'Type:N']
    ).properties(
        title=f'{N_NEXT} Days Prediction vs {N_NEXT} Days Previous',
        # height=400
    ).configure_legend(
        orient='top',
        direction='horizontal'
    )
    
    return bar_chart


# ========= Main App =========

# get data
data = get_data()

# predict data
data_predicts = predict(data)

# Load Model
model = load_model()

# UI
st.set_page_config(page_title="Order Count Prediction", layout="wide")

st.header("Order Count Prediction")
"""
The main objective of this prediction is to discover which day that might have the highest order count in the next N days,
It is important to know the fartest day that the model will predict might decrease the accuracy of the prediction,
because the model will predict the next day based on the previous day, 
and if the previous day is not accurate, the next day will be even more inaccurate.
"""

# -----------------------
# PREDICTION GENERAL SUMMARY
# -----------------------

st.header(f"Summary of Predicted Order Count for the Next {N_NEXT} Days")
metrics_list = [
    ("Total Order", "sum"),
    ("Avg Order", "mean"),
    ("Max Order", "max"),
    ("Min Order", "min")
]

cols = st.columns(len(metrics_list))

# Extract the two data windows once to avoid repeated slicing
current_window = data_predicts['Order_Count'].tail(N_NEXT)
previous_window = data_predicts['Order_Count'].iloc[-N_NEXT*2: -N_NEXT]

for col, (label, func_name) in zip(cols, metrics_list):
    # Dynamically call the pandas method (sum, mean, etc.)
    val_pred = getattr(current_window, func_name)()
    val_prev = getattr(previous_window, func_name)()

    delta = val_pred - val_prev

    col.metric(
        label=label,
        value=round(val_pred, 2) if func_name == "mean" else val_pred,
        delta=round(delta, 2)
    )

# -----------------------
# DATAFRAME NEXT N DAYS PREDICTION
# -----------------------
dataframe_container = st.container()
with dataframe_container:
	st.subheader(f"Predicted Order Count for the Next {N_NEXT} Days")
	next_n_days_data = data_predicts.iloc[-N_NEXT:]
	st.dataframe(next_n_days_data)
	st.caption("""
	Order Count column is the prediction value, while Order Diff column is the 
	predicted difference between the current day and the previous day,
	so if the Order Diff is positive, it means that the order count will increase compared to the previous day, 
	and if it is negative, it means that the order count will decrease compared to the previous day.
	""")

st.subheader(f"Prediction Comparison for the next {N_NEXT} days")
columns_chart_container = st.columns(2)
with columns_chart_container[0]:
	line_chart_container = st.container()
	with line_chart_container:
		pred_data = next_n_days_data['Order_Count']

		prev_one_year_index = pred_data.index - pd.Timedelta(days=365)
		prev_one_year_data = data_predicts.reset_index()

		start_prev_index = prev_one_year_data[prev_one_year_data['index'] == prev_one_year_index[0]].index[0]
		end_prev_index = start_prev_index + N_NEXT

		prev_data = prev_one_year_data.iloc[start_prev_index:end_prev_index]

		data_to_plot = pd.concat([pred_data.reset_index()['Order_Count'], prev_data.reset_index()['Order_Count']], axis=1)
		data_to_plot.columns = ['Predicted Order Count','Exact on Year Previous Order Count']

		chart = plot_line_chart(data_to_plot, var_1='Exact on Year Previous Order Count', var_2='Predicted Order Count', n_next=N_NEXT)
		st.altair_chart(chart, use_container_width=True)

		st.caption(f"""The comparison between next {N_NEXT} days prediction with one year previous order count, 
		the one year previous order count is used as a reference to see how the prediction will be 
		compared to the exact order count on the same day one year ago.""")

with columns_chart_container[1]:
	pred_vs_n_prev_container = st.container()
	with pred_vs_n_prev_container:
		prev_n_days = data_predicts['Order_Count'].iloc[-N_NEXT*2: -N_NEXT]
		pred_n_days = data_predicts['Order_Count'].iloc[-N_NEXT:]
		pred_with_prev_n_days = pd.concat( [
			prev_n_days.reset_index()['Order_Count'], 
			pred_n_days.reset_index()['Order_Count']
		], axis=1).reset_index()
		pred_with_prev_n_days.columns = ['Days', 'Previous', 'Predictions']
		st.altair_chart(draw_bar_chart_comparison(pred_with_prev_n_days))
		st.caption(f"""The comparison between next {N_NEXT} days prediction with previous {N_NEXT} days order count, each day in the next {N_NEXT} days is compared to the order count of the previous {N_NEXT} days,""")


st.markdown("---")

# -----------------------
# MODEL QUALITY
# -----------------------
model_quality_container = st.container()
with model_quality_container:
	st.header("Model Quality")
	st.markdown("""The model is trained on the previous data, so it is important to evaluate the model quality before using it for prediction, 
	the model quality can be evaluated using different metrics such as Mean Absolute Error (MAE), Mean Squared Error (MSE), Mean Absolute Percentage Error (MAPE), Root Mean Squared Error (RMSE). 
	so it is important to check the model quality before using it for prediction as it becomes our evaluation reference for the model's performance on prediction results.
	Also it is necessary to check the important feature that the model is using for prediction, as it will help us to understand the model's behavior and how it is making the prediction,
	""")

	# Load estimator and prepare data for evaluation
	estimator = OrderPredictivePipeline(estimator=model)
	data_prepared = estimator.preprocessing(data)
	
	# Get evaluation results
	preds, metrics = estimator.evaluate()
	
	# --- Display metrics ---
	metrics_container, eval_chart_container = st.columns(2)

	with metrics_container:
		metric_cols_num = 4
		metric_rows_num = len(metrics['metrics_eval']) // metric_cols_num
	
		# metric_cols = st.columns(metric_cols_num)
		# for i, (key, value) in enumerate(metrics['metrics_eval'].items()):
		# 	if i >= metric_cols_num:
		# 		break
		# 	with metric_cols[i]:
		# 		st.subheader(key.upper())
		# 		st.metric(label=list(value.keys())[0].upper(), value=round(value['train'], 4))
		# 		st.metric(label=list(value.keys())[1].upper(), value=round(value['test'], 4))
		
		metrics_df = pd.DataFrame(metrics['metrics_eval']).T.reset_index().rename(columns={'index': 'Metric'})
		st.dataframe(metrics_df)

		chart = alt.Chart(metrics_df).transform_fold(
			['train', 'test'],
			as_=['Dataset', 'Value']
		).mark_bar().encode(
			x=alt.X('Metric:N', title='Metric'),
			y=alt.Y('Value:Q', title='Score'),
			color=alt.Color('Dataset:N', scale=alt.Scale(domain=['train', 'test'], range=['#1f77b4', 'orange'])),
			xOffset='Dataset:N',
			tooltip=['Metric:N', 'Value:Q', 'Dataset:N']
		).properties(
			title='Model Evaluation Metrics',
			width='container'
		)

		st.altair_chart(chart, use_container_width=True)


	# --- Display comparison between training set and prediction for the first N days --- 
	with eval_chart_container:
		train_set, test_set = estimator.train_unshuffled_set, estimator.test_set
		first_n_day = st.number_input("Number of Days to Compare", min_value=1, max_value=len(test_set[0]), value=30)
		
		train_chart_tab, test_chart_tab = st.tabs(['Train', 'Test'])
		with train_chart_tab:
			# training vs prediction comparison
			# get the first n day of the training set and the prediction
			trian_preds_df = preds[0][:first_n_day]

			# get the first n day of the original training set
			train_original_df = train_set[0]['Order_Count'].iloc[:first_n_day]

			# create a dataframe for comparison
			train_comparison_df = pd.DataFrame({
				'Date': train_set[0].index[:first_n_day],
				'Original Order Count': train_original_df,
				'Predicted Order Count': trian_preds_df
			})
			# plot line chart for the comparison
			train_chart = plot_line_chart(
				data=train_comparison_df.reset_index(drop=True)[['Original Order Count', 'Predicted Order Count']], 
				var_1='Original Order Count', 
				var_2='Predicted Order Count', 
				n_next=first_n_day
			)
			st.altair_chart(train_chart, use_container_width=True)

		with test_chart_tab:
			# training vs prediction comparison
			# get the first n day of the training set and the prediction
			test_preds_df = preds[1][:first_n_day]

			# get the first n day of the original training set
			test_original_df = test_set[0]['Order_Count'].iloc[:first_n_day]

			# create a dataframe for comparison
			test_comparison_df = pd.DataFrame({
				'Date': test_set[0].index[:first_n_day],
				'Original Order Count': test_original_df,
				'Predicted Order Count': test_preds_df
			})
			# plot line chart for the comparison
			test_chart = plot_line_chart(
				data=test_comparison_df.reset_index(
					drop=True)[['Original Order Count', 'Predicted Order Count']],
				var_1='Original Order Count',
				var_2='Predicted Order Count',
				n_next=first_n_day
			)
			st.altair_chart(test_chart, use_container_width=True)

	

	# --- Faeture Essentials ---
	st.subheader("Feature Essentiality")
	feat_importance_cols, commulative_cols, corr_cols = st.columns(3)

	# Feature importance
	with feat_importance_cols:
		importances = model.feature_importances_
		importances_df = pd.DataFrame({
			'Feature': estimator.train_set[0].columns.drop('Order_Count'),
			'Importance': importances
		}).sort_values(by='Importance', ascending=False)

		alt_chart = alt.Chart(importances_df).mark_bar().encode(
			x='Importance:Q',
			y=alt.Y('Feature:N', sort='-x')
		).properties(
			title='Feature Importance'
		)

		st.altair_chart(alt_chart, use_container_width=True)

	# Cumulative importance
	with commulative_cols:
		importances_df['Cumulative Importance'] = importances_df['Importance'].cumsum()
		cumulative_chart = alt.Chart(importances_df).mark_line(point=True).encode(
			x='Cumulative Importance:Q',
			y=alt.Y('Feature:N', sort='-x')
		).properties(
			title='Cumulative Feature Importance'
		)
		st.altair_chart(cumulative_chart, use_container_width=True)
	
	# Correlation matrix
	with corr_cols:
		# correlation
		corr_matrix = estimator.train_set[0].corr()
		corr_matrix = corr_matrix.reset_index().melt(id_vars='index')

		corr_chart = alt.Chart(corr_matrix).mark_rect().encode(
			x=alt.X('index:N', title=None),
			y=alt.Y('variable:N', title=None),
			color=alt.Color('value:Q',
							# Darker blues represent higher correlation
							scale=alt.Scale(scheme='blues'),
							legend=alt.Legend(title="Correlation")
							),
			tooltip=['index', 'variable', 'value']
		).properties(
			title='Feature Correlation Matrix',
			width=500,
			height=500
		)
		st.altair_chart(corr_chart, use_container_width=True)




st.markdown("---")
st.markdown("### Note")
st.markdown("""The prediction is based on the previous data, 
so if there is a sudden change in the market or any external factor that might affect the order count, the prediction might not be accurate,
so it is important to use the prediction as a reference and not as a definite value""")


