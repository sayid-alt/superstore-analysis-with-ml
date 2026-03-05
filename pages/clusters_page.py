import streamlit as st
import pandas as pd
import altair as alt
import plotly.express as px
import plotly.figure_factory as ff
import pathlib

import plotly.graph_objects as go
from superstore_analysis.datasets import DataLoader
from superstore_analysis.processor import DataProcessor

script_dir = pathlib.Path(__file__).parent.resolve()

st.set_page_config(
	page_title='Super Store Dashboard',
	layout='wide',
)

@st.cache_resource
def get_processor():
	return DataProcessor()


@st.cache_data
def load_and_process_data(_processor, file_id):
	loader = DataLoader()
	data_path = script_dir.parents[0] / 'data/Clustered.csv'
	raw_data = loader.from_local(file_path=data_path)
	prepared_data = _processor.prepare_features(raw_data)
	processed_data = _processor.preprocess(prepared_data)

	return raw_data, prepared_data, processed_data


# Load Data with clusters
processor = get_processor()
raw_data, prepared_data, processed_data = load_and_process_data(
	processor, file_id='1vuSYru8_JT1Jx6hDSeT1Yo0-6mGK-Rv_')

# Apply cluster names transformations
raw_data['cluster_names'] = raw_data['clusters'].apply(
	lambda x: "cluster_" + str(x))

clusters_summary = raw_data.groupby(['clusters']).agg({
    'Sales': ['min', 'max', 'mean', 'std', 'sum'],
    'Profit': ['min', 'max', 'mean', 'std', 'sum'],
    'Discount': ['min', 'max', 'mean', 'std', 'sum'],
    'Quantity': ['min', 'max', 'mean', 'std', 'sum'],
    'Category': ['max'],
    'Sub-Category': ['max'],
    'City': ['max'],
    'State': ['max'],
    'Region': ['max'],
})


def draw_pie_sales_pct(df):
	profit_pct_df = pd.DataFrame({
		'clusters': df.index,
		'profit': df['Profit']['sum'].values,
       					'percentage': [p/df['Profit']['sum'].sum() for p in df['Profit']['sum']]
	})
	profit_pct_df = profit_pct_df.sort_values(by='clusters')
	profit_pct_chart = px.pie(
		data_frame=profit_pct_df,
		names='clusters',
		values='percentage',
		title='Profit Percentage By Clusters',
		subtitle='Total summation of all profit each clusters',
		category_orders={"clusters": [i for i in range(len(df.index))]}
	)
	return profit_pct_chart


def draw_strip_range_by_clusters(df, column: str):
	sales_dist_by_clusters = df.loc[:, ['clusters', column]]
	sales_dist_by_clusters['avg'] = df.groupby(
		by=['clusters'])[column].transform('mean')
	sales_dist_by_clusters['std'] = df.groupby(
		by=['clusters'])[column].transform('std')
	sales_dist_by_clusters['min'] = df.groupby(
		by=['clusters'])[column].transform('min')
	sales_dist_by_clusters['max'] = df.groupby(
		by=['clusters'])[column].transform('max')
	fig = px.strip(
            data_frame=sales_dist_by_clusters,
            y='clusters',
            x=column,
            color='clusters',
         			hover_data=dict(
                                    min=':.2f',
                                    avg=':.2f',
                                    std=':.2f',
                                    max=':.2f',
                                )
        )

	fig.update_traces(
		marker_size=15
	)

	fig.update_layout(
		title=dict(
			text=f'Min-Max {column} Distribution Range by Clusters'
		)
	)
	return fig


def draw_avg_discount_vs_profit_by_clusters(df):
	source = pd.concat(
            [
                df['Sales']['sum'].reset_index().rename(
                    columns={'sum': 'sales_sum'}),
                df['Discount']['mean'].reset_index().rename(
                    columns={'mean': 'discount_mean'}),
                df['Profit']['sum'].reset_index().rename(columns={
                    'sum': 'profit_sum'})
            ],
            axis=1
        ) \
            .drop(columns=['clusters']) \
            .reset_index().rename(columns={'index': 'clusters'})

	source['profit_pct'] = source['profit_sum'] / source['profit_sum'].sum()
	bar = alt.Chart(source).mark_bar() \
		.encode(
		x='clusters:N',
		y=alt.Y('discount_mean:Q'),
		color=alt.Color('profit_pct:Q').scale(scheme='greens'),
		tooltip=[alt.Text('discount_mean:Q', format='.2f'),
                    alt.Text('profit_pct', format='.2%')]
	)

	return (bar).properties(
		width=400,
		title=alt.Text(
			text="Average Discount Given for Each Clusters",
			subtitle="Avg Discount vs Profit Percentage",
		)
	).interactive()


def draw_item_sales_by_clusters(df):
	df = df[['Quantity']] \
            .unstack() \
            .reset_index() \
            .rename(columns={
                'level_0': 'group',
                'level_1': 'agg',
                0: 'value'
            })

	df = df[df['agg'] == 'sum']

	return alt.Chart(df).mark_bar() \
		.encode(
		x='clusters:N',
		y='value:Q',
		xOffset='group:O',
		color='value:Q',
		tooltip=[alt.Text('value:Q', format='.2f')]
	).properties(
		width=400,
		title="Sum of Quantity for Each Clusters"
	).interactive()


def draw_sales_discount_corr(df):
	heat_source = pd.concat(
            [
                df['Sales']['sum'].reset_index().rename(
                    columns={'sum': 'sales_sum'}),
                df['Discount']['mean'].reset_index().rename(
                    columns={'mean': 'discount_mean'}),
                df['Profit']['sum'].reset_index().rename(columns={
                    'sum': 'profit_sum'})
            ],
            axis=1
        ) \
            .drop(columns=['clusters']) \
            .corr() \
            .unstack() \
            .reset_index() \
            .rename(columns={
                'level_0': 'x',
                'level_1': 'y',
                0: 'value',
            })

	base = alt.Chart(heat_source).mark_rect() \
		.encode(
		x='x:O',
		y='y:O',
		color=alt.Color('value:Q').scale(scheme='greens').title('Score'),
		tooltip=['value:Q']
	)

	text = base.mark_text(size=15) \
		.encode(
		text=alt.Text('value:Q', format='.4f'),
		color=alt.condition(
			alt.datum.value > float(heat_source['value'].mean()),
			alt.value('white'),
			alt.value('black')
		),
	)

	return (base + text) \
		.properties(
		width=300,
		height=500,
		title=alt.Title(
			text='Correlation Scores',
			subtitle='Sales, Profit, Discount Correlation Scores'
		)
	)

# ======================== UI ========================

main_box = st.container(width="stretch")
with main_box:
	st.title("Super Store - Cluster Analysis 📊")
	st.space()

	"""This analysis from superstore selling data over 4 years. 
		During the sales, there is some unnesessary discount applied at some moment, 
		causes the minimum profit gained. The problem can be solved by looking the behaviour of the sales, and cluster it.
		So we can clearly see the clusters and be able to put the right price and discount for them. As the following chart exhibits the distributions from the sales. 
		Clustering is done by machine learning algorithm using the features of each data rows as an input to be fed up to the model.
		"""

	# Distribution Chart
	distribution_box = st.container(border=True)
	with distribution_box:
		col1, col2 = st.columns([0.6, 0.4])
		with col1:
			data_point_distribution_chart = px.scatter(
				data_frame=raw_data,
				x='pca_1',
				y='pca_2',
				color='cluster_names',
				hover_data=['Sales', 'Profit', 'Discount',
                                    'Month_Order', 'Segment', 'Sub-Category', 'Ship_Mode'],
				title='Data Point Distribution',
				subtitle='Distribution of clusters gained to be used as identify cluster behaviour through descriptive analytics',

			)

			data_point_distribution_chart.update_layout(
				legend=dict(
					orientation='h',
					yanchor='bottom',
					y=1,
					xanchor='center',
					x=0.5,
					title_text='',
				),
				margin=dict(t=120, l=20, r=20, b=20),
				title=dict(
					y=0.95,
				)
			)

			st.plotly_chart(data_point_distribution_chart,
                            theme='streamlit', width='stretch')

		with col2:
			total_by_clusters = raw_data.groupby(by=['cluster_names']).agg(
				total=('cluster_names', 'count')
			).reset_index()
			print(total_by_clusters)
			total_by_clusters_chart = px.bar(
				data_frame=total_by_clusters,
				x='cluster_names',
				y='total',
				color='cluster_names'
			)

			st.plotly_chart(total_by_clusters_chart, theme='streamlit')

	# Summary Chart analysis Box
	cols = st.columns(2)
	with cols[0]:
		# Summary dataframe
		st.subheader('Clusters Summary')
		st.dataframe(clusters_summary.style.highlight_max(axis=0))

	with cols[1]:
		# Profit Percentage
		profit_pct_chart = draw_pie_sales_pct(clusters_summary)
		st.plotly_chart(profit_pct_chart, theme='streamlit', width='stretch')

	# MinMax Range Box
	with st.container(width='stretch', border=True):
		st.subheader("Min Max Range Distribution")
		col1, col2 = st.columns([1, 1])

		with col1:
			minmax_sales_chart = draw_strip_range_by_clusters(
				raw_data, column='Sales')
			st.plotly_chart(minmax_sales_chart, theme='streamlit', height='content')

		with col2:
			minmax_profit_chart = draw_strip_range_by_clusters(
				raw_data, column='Profit')
			st.plotly_chart(minmax_profit_chart, theme='streamlit', height='content')

	# Correlation box
	with st.container(width='stretch', border=True):
		st.subheader("Correlations")
		cols = st.columns(2)
		with cols[0]:
			avg_discount_vs_price_chart = draw_avg_discount_vs_profit_by_clusters(
				clusters_summary)
			st.altair_chart(avg_discount_vs_price_chart, theme='streamlit')

			item_sales_by_clusters_chart = draw_item_sales_by_clusters(clusters_summary)
			st.altair_chart(item_sales_by_clusters_chart, theme='streamlit')

		with cols[1]:
			sales_discount_corr_chart = draw_sales_discount_corr(clusters_summary)
			st.altair_chart(sales_discount_corr_chart, theme='streamlit')
