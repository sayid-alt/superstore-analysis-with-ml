import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, MinMaxScaler


class Encoder(BaseEstimator, TransformerMixin):
	def __init__(self):
		self.one_hot_cols = ['Category', 'Sub-Category', 'City', 'State', 'Region']
		self.label_enc_cols = ['Ship_Mode', 'Segment', 'Product_ID', 'Customer_ID']
		self.fitted_one_hot_columns = None
		self.non_one_hot_output_cols = None
		self.final_output_columns_order = None
		# Use OrdinalEncoder for label_enc_cols to handle unseen labels
		self.ordinal_encoders = {col: OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1) for col in self.label_enc_cols}

	def fit(self, X, y=None):
		# Fit OrdinalEncoders on training data
		for col in self.label_enc_cols:
			self.ordinal_encoders[col].fit(X[[col]]) # OrdinalEncoder expects 2D array

		# Get columns that will not be one-hot encoded (numerical and label-encoded)
		# Ensure they are ordered as they appear in the input X
		self.non_one_hot_output_cols = [col for col in X.columns if col not in self.one_hot_cols]

		# Learn all possible columns for one-hot encoding from the training data
		temp_one_hot_df = pd.get_dummies(X[self.one_hot_cols], dtype='float')
		self.fitted_one_hot_columns = temp_one_hot_df.columns.tolist()

		# Define the complete final column order for the output DataFrame
		self.final_output_columns_order = self.non_one_hot_output_cols + self.fitted_one_hot_columns
		return self

	def transform(self, X) -> pd.DataFrame:
		X_processed = X.copy()

		# Apply Ordinal Encoding using fitted encoders
		for col in self.label_enc_cols:
			# OrdinalEncoder returns 2D, take 1st column
			X_processed[col] = self.ordinal_encoders[col].transform(X_processed[[col]])[:,0]

		# Separate numerical/label-encoded columns and ensure their order
		numerical_and_label_df = X_processed.drop(columns=self.one_hot_cols)

		# Reindex to ensure order. This also handles if X_processed has more columns than expected
		numerical_and_label_df = numerical_and_label_df[self.non_one_hot_output_cols]

		# Generate one-hot encoded features for the current X
		one_hot_df = pd.get_dummies(X_processed[self.one_hot_cols], dtype='float')

		# Align one-hot encoded columns with the columns learned during fit
		one_hot_df = one_hot_df.reindex(columns=self.fitted_one_hot_columns, fill_value=0.0).reset_index(drop=True)

		# Ensure the order of one-hot encoded columns is consistent with fit
		one_hot_df = one_hot_df[self.fitted_one_hot_columns]

		# Concatenate the two parts: numerical/label-encoded and one-hot encoded
		# Use reset_index(drop=True) to avoid issues with misaligned indices after dropping columns
		final_df = pd.concat([numerical_and_label_df.reset_index(drop=True), one_hot_df.reset_index(drop=True)], axis=1)

		# Finally, ensure the entire DataFrame has the exact same columns and order as defined during fit
		final_df = final_df[self.final_output_columns_order]

		return final_df

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
		include_decomposition: bool = False,
	) -> np.array:

		pipe = Pipeline(steps=[
			('encoder', Encoder()),
			('scaler', MinMaxScaler()),
		])

		if fit_pipe:
			preprocessed = pipe.fit_transform(df)
			return preprocessed
		else:
			return pipe.transform(df)

		
	
