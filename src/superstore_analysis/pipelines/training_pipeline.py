from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from superstore_analysis.processor import Encoder
from typing import Literal
from sklearn.cluster import KMeans, MeanShift, DBSCAN
from sklearn.metrics import make_scorer, silhouette_score
import numpy as np

class ClusteringPipeline(TransformerMixin, BaseEstimator):
	def __init__(self):
		pass

	def preprocessing(self, X, kind: Literal['fit', 'transform', 'both'] = 'both'):
		if kind not in ['fit', 'transform', 'both']:
			raise ValueError(f"Name {kind} is not in ['fit', 'transform', 'both']")

		preproc_pipe = Pipeline([
			('encoder', Encoder()),
			('scaler', MinMaxScaler()),
			('decomposition', PCA(n_components=2, random_state=1))
		])

		if kind == 'fit':
			return preproc_pipe.fit(X)
		
		elif kind == 'transform':
			return preproc_pipe.transform(X)
		
		else:
			return preproc_pipe.fit_transform(X)
	
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
                        'eps': np.arange(0.1, 0.5, 0.03),
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
		best_model_name = list(self.eval_results.items())[0][0]
		best_model = self.models[best_model_name]

		labels = best_model.predict(X)
		return labels
	
	def eval_models(self, X):
		self.eval_results = {}
		for name, model in self.models.items():
			self.eval_results[name] = self._calc_score(model, X)

		# sort from best score
		self.eval_results = dict(sorted(self.eval_results.items(), key=lambda item: item[1], reverse=True))
		return self.eval_results

	def _calc_score(self, estimator, X, y_true=None):
		labels = estimator.fit_predict(X)
		if len(set(labels)) > 1:
			return silhouette_score(X, labels)
		else:
			return -1


