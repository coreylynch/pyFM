import numpy as np
from .libfm import fm_train, fm_predict
from sklearn import base

def _prepare_fit(y):
	y_i = np.ones(y.shape, dtype=np.float64, order="C")
	y_i[y != 1.0] = -1.0
	return y_i

class FactorizationMachineClassifier(base.BaseEstimator):

	def __init__(self, n_iter=5, method="sgd", dim=10, learn_rate=0.01,
				 param_init_stdev=0.1, reg=0.0):
		self.n_iter = n_iter
		self.method = method
		self.task = "c"
		self.dim = dim
		self.learn_rate = learn_rate
		self.param_init_stdev = param_init_stdev
		self.reg = reg

	def fit(self, X, y, cat_columns, num_columns=None):
		self.cat_columns = cat_columns
		self.num_columns = num_columns
		y_i = _prepare_fit(y)
		params = fm_train(X, y_i, cat_columns, num_columns, self.n_iter,
			              self.dim, self.method, self.task, self.learn_rate,
			              self.param_init_stdev, self.reg) 
		self.coef_ = params[0]
		self.params = params
		return self

	def predict(self, X):
		coef, v_file, bias, m_sum, m_sum_sqr = self.params
		preds = fm_predict(X, self.method, coef, v_file, bias, m_sum,
						   m_sum_sqr, self.dim, self.task, self.cat_columns,
						   self.num_columns)
		return preds

class FactorizationMachineRegressor(base.BaseEstimator):

	def __init__(self, n_iter=5, method="sgd", dim=10, learn_rate=0.01,
				 param_init_stdev=0.1, reg=0.0):
		self.n_iter = n_iter
		self.method = method
		self.task = "r"
		self.dim = dim
		self.learn_rate = learn_rate
		self.param_init_stdev = param_init_stdev
		self.reg = reg

	def fit(self, X, y, cat_columns, num_columns=None):
		self.cat_columns = cat_columns
		self.num_columns = num_columns
		y_i = _prepare_fit(y)
		params = fm_train(X, y_i, cat_columns, num_columns, self.n_iter,
			              self.dim, self.method, self.task, self.learn_rate,
			              self.param_init_stdev, self.reg) 
		self.coef_ = params[0]
		self.params = params
		return self

	def predict(self, X):
		coef, v_file, bias, m_sum, m_sum_sqr = self.params
		preds = fm_predict(X, self.method, coef, v_file, bias, m_sum,
						   m_sum_sqr, self.dim, self.task, self.cat_columns,
						   self.num_columns)
		return preds