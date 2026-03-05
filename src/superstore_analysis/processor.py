import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler, OneHotEncoder
from typing import List


class Encoder(BaseEstimator, TransformerMixin):
	def __init__(self, one_hot_cols: List[str], ordinal_enc_cols: List[str]):
		self._one_hot_cols = one_hot_cols
		self._ordinal_enc_cols = ordinal_enc_cols
		self.ct = None  # Initialize ColumnTransformer here

	def fit(self, X, y=None):
		# Initialize ColumnTransformer in fit to learn categories from training data
		self.ct = ColumnTransformer([
			('one_hot_encoder', OneHotEncoder(handle_unknown='ignore'), self._one_hot_cols),
			('ordinal_encoder', OrdinalEncoder(), self._ordinal_enc_cols)
		])
		self.ct.fit(X)  # Fit the ColumnTransformer
		return self

	def transform(self, X) -> pd.DataFrame:
		if self.ct is None:
			raise RuntimeError("Encoder not fitted. Call fit() first.")

		# Transform using the fitted ColumnTransformer
		transformed_data = self.ct.transform(X)
		if hasattr(transformed_data, "toarray"):
			return transformed_data.toarray()
		return transformed_data

class DataProcessor():
	def __init__(self):
		pass
	
	def prepare_features(self, df: pd.DataFrame):
		df['Month_Order'] = df['Order_Date'].dt.month
		df['Day_Order'] = df['Order_Date'].dt.day
		df['Days_Shipping'] = (df['Ship_Date'] - df['Order_Date']).dt.days

		df['Max_Sales_Month'] = df.groupby(
                    'Month_Order')['Sales'].transform('max')
		df['Min_Sales_Month'] = df.groupby(
                    'Month_Order')['Sales'].transform('min')
		df['Mean_Sales_Month'] = df.groupby(
                    'Month_Order')['Sales'].transform('mean')

		#Product & Sub-Category Sales
		df['product_sales'] = df.groupby(
                    'Product_ID')['Sales'].transform('sum')
		df['sub_category_sales'] = df.groupby(
                    'Sub-Category')['Sales'].transform('sum')

		exclude_features = ['Order_ID', 'Customer_Name','Product_Name', 'Order_Date', 'Ship_Date', 'Country/Region']

		df_train = df.drop(columns=exclude_features)
		return df_train

	def preprocess(
		self, 
		df: pd.DataFrame, 
		fit_pipe: bool = True,
	) -> np.array:
		
		ohe_cols = ['Category', 'Sub-Category', 'City', 'State', 'Region']
		ordinal_cols = ['Ship_Mode', 'Segment']

		pipe = Pipeline(steps=[
			('encoder', Encoder(one_hot_cols=ohe_cols, ordinal_enc_cols=ordinal_cols)),
			('scaler', MinMaxScaler()),
		])

		if fit_pipe:
			preprocessed = pipe.fit_transform(df)
			return preprocessed
		else:
			return pipe.transform(df)

		
	
