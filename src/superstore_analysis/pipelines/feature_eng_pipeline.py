import pandas as pd

from loguru import logger
from sklearn.base import TransformerMixin, BaseEstimator
from superstore_analysis.datasets import DataLoader

class FeatureEng(TransformerMixin, BaseEstimator):
	def __init__(self, exclude_unused: bool = True):
		self.exclude_features = ['Order_ID', 'Customer_Name','Product_Name', 'Order_Date', 'Ship_Date', 'Country/Region']
	
	def fit(self, X, y=None):
		return self
	
	def transform(self, X):
		logger.info("Start Feature Engineering...")
		X['Month_Order'] = X['Order_Date'].dt.month
		X['Day_Order'] = X['Order_Date'].dt.day
		X['Days_Shipping'] = (X['Ship_Date'] - X['Order_Date']).dt.days

		# Add min, max, mean of sales by months
		sales_agg_in_month = X.groupby('Month_Order').agg(
			Max_Sales_Month=pd.NamedAgg(column='Sales', aggfunc='max'),
			Min_Sales_Month=pd.NamedAgg(column='Sales', aggfunc='min'),
			Mean_Sales_Month=pd.NamedAgg(column='Sales', aggfunc='max'),
		).reset_index()

		X = X.merge(sales_agg_in_month, on='Month_Order')

		# Add sum of sales by product IDs
		total_sales_product = X.groupby(
			by='Product_ID').agg(product_sales=('Sales', 'sum'))
		X = X.merge(total_sales_product, on='Product_ID')

		total_sales_sub_category = X.groupby(
			by='Sub-Category').agg(sub_category_sales=('Sales', 'sum'))
		X = X.merge(total_sales_sub_category, on='Sub-Category')

		X = X.drop(columns=self.exclude_features)
		return X


