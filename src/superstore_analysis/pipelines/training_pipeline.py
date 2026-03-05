import numpy as np
import dill as pickle
import os
import shutil
import pathlib
import pandas as pd

from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from typing import Literal
from sklearn.cluster import KMeans, MeanShift, DBSCAN
from sklearn.metrics import make_scorer, silhouette_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from typing import List
from loguru import logger
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

from superstore_analysis.pipelines.feature_eng_pipeline import FeatureEng
from superstore_analysis.processor import Encoder

script_dir = pathlib.Path(__file__).parent.resolve()


class ClusterPipeline:
	def __init__(self):
		pass

	def preprocessing(self, X):

		ohe_cols = ['Category', 'Sub-Category', 'City', 'State', 'Region']
		ordinal_cols = ['Ship_Mode', 'Segment']

		preproc_pipe = Pipeline([
			('encoder', Encoder(one_hot_cols=ohe_cols, ordinal_enc_cols=ordinal_cols)),
			('scaler', MinMaxScaler()),
			('decomposer', PCA(n_componenX=2, random_state=42))
		])
		
		X_transformed = preproc_pipe.fit_transform(X)
		
		# Create saving dir
		model_save_dir = script_dir.parenX[2] / "models"
		os.makedirs(model_save_dir, exist_ok=True)

		# save scaler file
		scaler_save_path = script_dir.parenX[2] / 'models/cluster_scaler.pkl'
		scaler_obj = preproc_pipe.named_steps['scaler']
		with open(scaler_save_path, 'wb') as f:
			pickle.dump(scaler_obj, f)
		
		# save decomposer file
		decomposer_save_path =  script_dir.parenX[2] / 'models/cluster_decomposer.pkl'
		decomposer_obj = preproc_pipe.named_steps['decomposer']
		with open(decomposer_save_path, 'wb') as f:
			pickle.dump(decomposer_obj, f)

		return X_transformed
	
	def train(self, X):
		# Kmeans model fit
		kmeans_cv = GridSearchCV(
			estimator=KMeans(),
			param_grid={
				'n_clusters': [i for i in range(3, 16)]
			},
			scoring=self._calc_score,
		)
		kmeans_cv.fit(X, y=None)

		# Kmeans model fit
		dbscan_cv = GridSearchCV(
                    estimator=DBSCAN(),
                    param_grid={
                        'eps': np.arange(0.1, 1, 0.05),
                        'min_samples': [i for i in range(2, 10, 2)]
                    },
                    scoring=self._calc_score,
                    verbose=1,
                    cv=2
                )


		dbscan_cv.fit(X)

		mean_shift_cv = GridSearchCV(
                    estimator=MeanShift(),
                    param_grid={
                        'bandwidth': np.arange(0.1, 0.6),
                    },
                    cv=2,
                    scoring=self._calc_score,
                    verbose=2,
                )
		mean_shift_cv.fit(X)

		self.models = {
			'kmeans': kmeans_cv.best_estimator_,
			'mean_shift': mean_shift_cv.best_estimator_,
			'dbscan': dbscan_cv.best_estimator_,
		}

		return self.models
	
	def predict(self, X):
		best_model_name = list(self.eval_resulX.items())[0][0]
		best_model = self.models[best_model_name]

		# saving best model
		model_save_dir = script_dir.parenX[2] / "models"
		
		# create a new model file
		os.makedirs(model_save_dir, exist_ok=True)
		model_path = os.path.join(model_save_dir, 'cluster_model.pkl')
		with open(model_path, 'wb') as f:
			pickle.dump(best_model, f)

		# prediction
		labels = best_model.fit_predict(X)
		return labels
	
	def eval_models(self, X):
		self.eval_resulX = {}
		for name, model in self.models.items():
			self.eval_resulX[name] = self._calc_score(model, X)

		# sort from best score
		self.eval_resulX = dict(sorted(self.eval_resulX.items(), key=lambda item: item[1], reverse=True))

		return self.eval_resulX

	def _calc_score(self, estimator, X, y_true=None):
		labels = estimator.fit_predict(X)
		if len(set(labels)) > 1:
			return silhouette_score(X, labels)
		else:
			return -1


class ClassificationPipeline:
	def __init__(self, est=None):
		self._est = est if est else LogisticRegression(
			penalty='elasticnet', solver='saga', l1_ratio=0.2, max_iter=10)

	@property
	def get_estimator_(self):
		return self._est

	def preprocessing(self, X: pd.DataFrame, target: str,  features: List[str] = None, test_size: int = 0.2):
		
		if features == None:
			features = [
				'Sales', 
				'Category',
				'Ship_Mode', 
				'Segment',
				'Region',
			]
		
		y = X.loc[:, target]
		X = X.loc[:, features]
		X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=test_size)

		ohe_cols = ['Category', 'Region']
		ordinal_enc_cols = ['Ship_Mode', 'Segment']

		preproc_pipe = Pipeline([
			('encoder', Encoder(one_hot_cols=ohe_cols, ordinal_enc_cols=ordinal_enc_cols)),
			('scaler', MinMaxScaler()),
		])

		X_train_prep = preproc_pipe.fit_transform(X_train)
		X_test_prep = preproc_pipe.transform(X_test)

		# save pipeline object into pickle file
		pipe_save_dir = script_dir.parents[2] / 'models'
		pipe_save_path = script_dir.parents[2] / 'models/classify_pipe.pkl'
		with open(pipe_save_path, 'wb') as f:
			pickle.dump(preproc_pipe, f)

		return (X_train_prep, y_train), (X_test_prep, y_test)
	
	def train(self, X, y):
		self._est.fit(X, y)

		# create model dir for saving
		model_save_dir = script_dir.parents[2] / 'models'
		os.makedirs(model_save_dir, exist_ok=True)

		# save model
		model_save_path = script_dir.parents[2] / 'models/classify_model.pkl'
		with open(model_save_path, 'wb') as f:
			pickle.dump(self._est, f)

		return self._est
	
	def evaluate(self, train_set, test_set):
		X_train, y_train = train_set
		X_test, y_test = test_set

		y_train_pred = self._est.predict(X_train)
		train_acc = accuracy_score(y_train, y_train_pred)

		y_test_pred = self._est.predict(X_test)

		test_acc = accuracy_score(y_test, y_test_pred)
		print(f"Training Accuracy: {train_acc}")
		print(f"Test Accuracy: {test_acc}")
		


class OrderPredictivePipeline:
	def __init__(self, estimator: BaseEstimator = None, param_grid: dict = None):
		default_estimator = RandomForestRegressor()
		default_param_grid = {
			'n_estimators': [200, 250, 300, 350, 450],
			'min_samples_split': [150, 200, 300, 400]
		}
		self._estimator = estimator if estimator else default_estimator
		self._param_grid = param_grid if param_grid else default_param_grid
		
		self.estimator = None
		self.target_col = None
		self.original_target_col = None

	def preprocessing(
		self, 
		X: pd.DataFrame, 
		features: List[str] = None, 
		test_size: int = 0.2
	):
		class FeatureCreator(TransformerMixin, BaseEstimator):
			def fit(self, X, y=None):
				return self
			
			def transform(self,X):
				X_date_sorted = X.sort_values(by='Order_Date')
				X_selection = X_date_sorted.groupby(by='Order_Date').agg(
					Order_Count=('Order_ID', 'count'),
					# Quantity=('Quantity', 'sum'),
					Sales_Avg=('Sales', 'mean'),
					Discount_Avg=('Discount', 'mean'),
					Days_Shipping_Avg=('Days_Shipping', 'mean'),
					Postal_Code=('Postal_Code', 'max')
				)

				logger.info("Creating features...")

				X_prep = self._create_features(X_selection)
				X_prep = X_prep.dropna()

				X_prep['Order_Diff'] = X_prep['Order_Count'].diff(1)
				X_prep['Target_Next_Diff'] = X_prep['Order_Diff'].shift(-1)
				X_prep = X_prep.dropna()

				self.prepared_data = X_prep

				return X_prep
			
			def _create_features(self, X):
				X['Day'] = X.index.day
				X['Month'] = X.index.month
				X['Day_of_Week'] = X.index.dayofweek
				X['Day_of_Year'] = X.index.dayofyear
				X['Week_of_Year'] = X.index.isocalendar().week
				X['Order_Rolling_Sum_5'] = X['Order_Count'].rolling(window=6).sum()
				X['Order_Rolling_Mean_5'] = X['Order_Count'].rolling(window=6).mean()
				X['Order_Rolling_std_5'] = X['Order_Count'].rolling(window=6).std()
				X['Is_Weekend'] = [1 if (dayofweek == 5) or (
					dayofweek == 6) else 0 for dayofweek in X['Day_of_Week'].values]

				return X


		class Splitter(TransformerMixin, BaseEstimator):
			def __init__(self, target_col: str = 'Target_Next_Diff', original_target_col: str = 'Order_Count'):
				self._target = target_col
				self._original_target = original_target_col

			def fit(self, X, y=None):
				return self
			
			def transform(self, X):
				# splitting data into train and test sets
				logger.info("Splitting data into train and test sets...")

				frac_size = 0.5
				test_size = int(len(X) * frac_size)

				# split data into train and test sets
				train_reg = X.iloc[:-test_size, :]
				test_reg = X.iloc[-test_size:, :]

				# shuffle train set
				train_shuffled_reg = train_reg.sample(frac=1)

				# split into X and y
				X_train_reg, y_train_reg = train_shuffled_reg.drop(
					columns=self._target),  train_shuffled_reg.loc[:, self._target].values
				X_test_reg, y_test_reg = test_reg.drop(
					columns=self._target),  test_reg.loc[:, self._target].values
				
				train_set = (X_train_reg, y_train_reg)
				test_set = (X_test_reg, y_test_reg)

				return train_set, test_set
		
		# pipeline data transformation
		pipeline = Pipeline([
			('feature_engineering', FeatureEng(exclude_unused=False)),
			('feature_creation', FeatureCreator()),
		], verbose=True)
		prepared_data = pipeline.fit_transform(X)
		

		# store necessary attributes for later use
		splitter = Splitter()
		self.train_set, self.test_set = splitter.fit_transform(prepared_data)
		self.target_col = splitter._target
		self.original_target_col = splitter._original_target

		# save pipeline object into pickle file
		pipe_save_dir = script_dir.parents[2] / 'models'
		os.makedirs(pipe_save_dir, exist_ok=True)
		pipe_save_path = os.path.join(pipe_save_dir, 'regression_pipe.pkl')
		with open(pipe_save_path, 'wb') as f:
			pickle.dump(pipeline, f)

		return prepared_data
	
	@property
	def get_prepared_data_(self):
		if self.prepared_data is not None:
			return self.prepared_data
		else:
			logger.warning("No prepared data found. Please run the preprocessing method first.")
			return None

	def train(self, train_set=None, test_set=None):
		if train_set is not None and test_set is not None:
			self.train_set = train_set
			self.test_set = test_set

			logger.info("Using provided train and test sets for training.")
		else:
			X_train, y_train = self.train_set
			X_test, y_test = self.test_set

		grid_search = GridSearchCV(
                    self._estimator,
                    param_grid=self._param_grid,
                    scoring='neg_root_mean_squared_error',
                    cv=3,
                    n_jobs=2,
                    verbose=3)

		self.estimator = grid_search.fit(X_train, y_train)
		return self.estimator

	@property
	def print_cv_score_history_(self):
		if self.estimator is None:
			print("No CV results found. Please run the train method first.")
			return None
		cv_results = self.estimator.cv_results_
		add_data = {
                    'mean_test_score': cv_results['mean_test_score'] * -1,
					'rank_test_score': cv_results['rank_test_score']
			}
		results_df = pd.DataFrame(cv_results['params'])
		results_df = pd.concat([results_df, pd.DataFrame(add_data)], axis=1)
		return results_df
	
	def evaluate(self):
		X_train, X_test = self.train_set[0], self.test_set[0]
		y_train, y_test = self.train_set[1], self.test_set[1]

		# define original target before differencing
		train_original = X_train.loc[:, self.original_target_col].values
		test_original = X_test.loc[:, self.original_target_col].values

		# Drop original data to predict differencing
		X_train = X_train.drop(columns=[self.original_target_col])
		X_test = X_test.drop(columns=[self.original_target_col])

		best_estimator = self.estimator.best_estimator_
		best_estimator.fit(X_train, y_train)

		# saving best model
		model_save_dir = script_dir.parents[2] / 'models'
		model_save_path = os.path.join(model_save_dir, 'regression_model.pkl')
		os.makedirs(model_save_dir, exist_ok=True)
		with open(model_save_path, 'wb') as f:
			pickle.dump(best_estimator, f)

		# predict differencing
		y_train_pred = best_estimator.predict(X_train)
		y_test_pred = best_estimator.predict(X_test)

		# predict final value
		y_train_pred = train_original + y_train_pred.astype(int)
		y_test_pred = test_original + y_test_pred.astype(int)

		# get mae score
		mae = mean_absolute_error(train_original, y_train_pred)
		mae_test = mean_absolute_error(test_original, y_test_pred)

		mape = mean_absolute_percentage_error(train_original, y_train_pred)
		mape_test = mean_absolute_percentage_error(test_original, y_test_pred)


		return (y_train_pred, y_test_pred), {
			'model': best_estimator,
			'model_name': best_estimator.__class__.__name__,
			'mae': mae,
			'mae_test': mae_test,
			'mape': mape,
			'mape_test': mape_test,
		}

	


	
