import numpy as np
from .libfm import fm_train, fm_predict
from sklearn import base

class FactorizationMachineClassifier(base.BaseEstimator):

	def __init__(self, n_iter=5, method="sgd", dim=10):
		self.n_iter = n_iter
		self.method = method
		self.task = "c"
		self.dim = dim

	def fit(self, X, y, cat_columns, num_columns=None):
		self.cat_columns = cat_columns
		self.num_columns = num_columns
		params = fm_train(X, y, cat_columns, num_columns, self.n_iter, self.dim, self.method, self.task) 
		self.coef_ = params[0]
		self.params = params
		return self

	def predict(self, X):
		coef, v_file, bias, m_sum, m_sum_sqr = self.params
		preds = fm_predict(X, self.method, coef, v_file, bias, m_sum, m_sum_sqr,
						   self.dim, self.task, self.cat_columns, self.num_columns)
		return preds

class FactorizationMachineRegressor(base.BaseEstimator):

	def __init__(self, n_iter=5, method="sgd", dim=10):
		self.n_iter = n_iter
		self.method = method
		self.task = "r"
		self.dim = dim

	def fit(self, X, y, cat_columns, num_columns=None):
		self.cat_columns = cat_columns
		self.num_columns = num_columns
		params = fm_train(X, y, cat_columns, num_columns, self.n_iter, self.dim, self.method, self.task) 
		self.coef_ = params[0]
		self.params = params
		return self

	def predict(self, X):
		coef, v_file, bias, m_sum, m_sum_sqr = self.params
		preds = fm_predict(X, self.method, coef, v_file, bias, m_sum, m_sum_sqr,
						   self.dim, self.task, self.cat_columns, self.num_columns)
		return preds