import sys, tempfile
import numpy as np
import _libfm
from transform import dump_categorical_df_to_svm_light
import os

def fm_train(X, y, cat_columns, num_columns=None, n_iter=5,
			 dim=10,  method="sgd", task="c", learn_rate=0.01, 
			 param_init_stdev=0.1, reg=0.0):
	if num_columns is not None:
		with tempfile.NamedTemporaryFile(delete=False) as f:
			dump_categorical_df_to_svm_light(X, y, f.name, cat_columns,
											 num_columns)
	else:
		with tempfile.NamedTemporaryFile(delete=False) as f:
			dump_categorical_df_to_svm_light(X, y, f.name, cat_columns)
	coef, v_file, bias, m_sum, m_sum_sqr = _libfm.train(f.name, method, task,
													    dim, learn_rate,
													    param_init_stdev,
													    reg)
	os.unlink(f.name)
	return coef, v_file, bias, m_sum, m_sum_sqr

def fm_predict(X, method, coef, v_file, bias, m_sum, m_sum_sqr,
			   dim, task, cat_columns, num_columns=None):
	# rewrite transform.dump_categorical_df_to_svm_light as a class
	#mock y for now
	y = np.ones(len(X))
	if num_columns is not None:
		with tempfile.NamedTemporaryFile(delete=False) as f:
			dump_categorical_df_to_svm_light(X, y, f.name, cat_columns,
											 num_columns)
	else:
		with tempfile.NamedTemporaryFile(delete=False) as f:
			dump_categorical_df_to_svm_light(X, y, f.name, cat_columns)	
	prediction = _libfm.predict(f.name, method, coef, v_file, bias, m_sum, m_sum_sqr, dim, task)
	os.unlink(f.name)
	return prediction