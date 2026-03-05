import pathlib
import pickle
import numpy as np
import pandas as pd
from superstore_analysis.pipelines.feature_eng_pipeline import FeatureEng
from superstore_analysis.processor import DataProcessor
from superstore_analysis.processor import Encoder
from .training_pipeline import OrderPredictivePipeline
from sklearn.pipeline import Pipeline

script_dir = pathlib.Path(__file__).parent.resolve()

class ClusterInfer:
	def __init__(self, est = None):
		DEFAULT_MODEL_PATH = script_dir.parents[2] / "models/cluster_model.pkl"
		DEFAULT_SCALER_PATH = script_dir.parents[2] / "models/cluster_scaler.pkl"
		DEFAULT_DECOMPOSER_PATH = script_dir.parents[2] / \
			"models/cluster_decomposer.pkl"

		# load default model
		with open(DEFAULT_MODEL_PATH, 'rb') as f:
			default_model = pickle.load(f)

		self._est = est if est else default_model
		
		with open(DEFAULT_SCALER_PATH, 'rb') as f:
			self.scaler = pickle.load(f)

		with open(DEFAULT_DECOMPOSER_PATH, 'rb') as f:
			self.decomposer = pickle.load(f)
	
	@property
	def get_model(self):
		return self._est

	def infer(self, X) -> pd.DataFrame:
		X_preproc = self._preprocess(X)
		if hasattr(self._est, 'predict'):
			clusters = self._est.predict(X_preproc)
		else:
			clusters = self._est.fit_predict(X_preproc)
		X[['pca_1', 'pca_2']] = X_preproc
		X['clusters'] = clusters

		final_csv_save_path = script_dir.parents[2] / "data/Clustered.csv"
		X.to_csv(final_csv_save_path)
		return X

	def _preprocess(self, X) -> np.array:
		feat_eng = FeatureEng()
		X_eng = feat_eng.fit_transform(X)

		ohe_cols = ['Category', 'Sub-Category', 'City', 'State', 'Region']
		ordinal_cols = ['Ship_Mode', 'Segment']
		
		preproc_pipe = Pipeline([
			('encoder', Encoder(one_hot_cols=ohe_cols, ordinal_enc_cols=ordinal_cols)),
			('scaler', self.scaler),
			('decomposer', self.decomposer),
		])

		X_preproc = preproc_pipe.fit_transform(X_eng)
		return X_preproc


class ClassifyInfer:
	def __init__(self, est = None, pipe = None):
		DEFAULT_MODEL_PATH = script_dir.parents[2] / 'models/classify_model.pkl'
		DEFAULT_PIPE_PATH = script_dir.parents[2] / 'models/classify_pipe.pkl'
		with open(DEFAULT_MODEL_PATH, 'rb') as f:
			default_model = pickle.load(f)
		
		self._est = est if est else default_model

		with open(DEFAULT_PIPE_PATH, 'rb') as f:
			default_pipe = pickle.load(f)
		
		self._pipe = pipe if pipe else default_pipe

		self.proba_prediction = None
	
	@property
	def get_proba_prediction(self):
		return self.proba_prediction
	
	def infer(self, X):
		# X_eng = FeatureEng().fit_transform(X)
		X_preproc = self._pipe.transform(X)
		
		prediction = self._est.predict(X_preproc)

		self.proba_prediction = self._est.predict_proba(X_preproc)

		return prediction

class OrderPredictionInfer:
	def __init__(self, est = None, pipe = None):
		DEFAULT_MODEL_PATH = script_dir.parents[2] / 'models/regression_model.pkl'
		DEFAULT_PIPE_PATH = script_dir.parents[2] / 'models/regression_pipe.pkl'
		
		with open(DEFAULT_MODEL_PATH, 'rb') as f:
			default_model = pickle.load(f)
		
		with open(DEFAULT_PIPE_PATH, 'rb') as f:
			default_pipe = pickle.load(f)

		self._est = est if est else default_model
		self._pipe = pipe if pipe else default_pipe
	
	def infer(self, X, n_next: int = 30):
		X_preproc = self._pipe.fit_transform(X)
		X_preproc = X_preproc.drop(columns=['Target_Next_Diff'])
		data_infer = X_preproc.copy()
		for step in range(n_next):
			original_target = data_infer.loc[:, 'Order_Count']
			X_infer = data_infer.drop(columns=['Order_Count'])  # drop original target

			# predict difference
			diff_pred = self._est.predict(X_infer.iloc[-1, :].to_frame().T)

			# Define final prediction
			order_pred = diff_pred[0] + int(original_target[-1])

			# store new prediction
			new_date = data_infer.index[-1] + pd.Timedelta(days=1)
			print(f"prediction on: {new_date} --> {order_pred}")
			doy = 365 if new_date.dayofyear >= 365 else new_date.dayofyear

			new_row_dict = {col: 0 for col in data_infer.columns}
			new_row_dict['Order_Count'] = order_pred
			new_row_dict['Order_Diff'] = int(diff_pred[0])

			value_refs = data_infer.groupby(by='Day_of_Year').agg(
				Sales_Avg=('Sales_Avg', 'mean'),
				Discount_Avg=('Discount_Avg', 'mean'),
				Days_Shipping_Avg=('Days_Shipping_Avg', 'mean'),
				Postal_Code=('Postal_Code', 'max'),
				Order_Diff=('Order_Diff', 'mean')
			)

			new_row_dict['Sales_Avg'] = value_refs.loc[(doy, 'Sales_Avg')]
			new_row_dict['Discount_Avg'] = value_refs.loc[(doy, 'Discount_Avg')]
			new_row_dict['Days_Shipping_Avg'] = value_refs.loc[(doy, 'Days_Shipping_Avg')]
			new_row_dict['Postal_Code'] = value_refs.loc[(doy, 'Postal_Code')]
			new_row_dict['Order_Diff'] = (
				value_refs.loc[(doy, 'Order_Diff')] + diff_pred[0])/2

			new_row_df = pd.DataFrame([new_row_dict], index=[new_date])

			data_infer = pd.concat([data_infer, new_row_df])
			data_infer = self._pipe.named_steps['feature_creation']._create_features(data_infer)  # create features for new data
		
		return data_infer
