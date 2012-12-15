import sys, tempfile
import numpy as np
import _libfm
from minifm.transform import dump_categorical_df_to_svm_light

def fm_train(X, y, n_iter, cat_columns, num_columns=None, n_iter=5,
			 dim=10,  method="sgd", task="c"):
	if num_columns is not None:
		with tempfile.NamedTemporaryFile() as f:
			dump_categorical_df_to_svm_light(X, y, f.name, cat_columns,
											 num_columns)
	else:
		with tempfile.NamedTemporaryFile() as f:
			dump_categorical_df_to_svm_light(X, y, f.name, cat_columns)

	coef, v_file, bias, m_sum, m_sum_sqr = _libfm.train(f.name, method, task, dim)
	return coef, v_file, bias, m_sum, m_sum_sqr

def fm_predict(data, method, coef, v_file, bias, m_sum, m_sum_sqr, dim, task,
			   cat_columns, num_columns=None):
	# rewrite transform.dump_categorical_df_to_svm_light as a class so
	# we can easily put train and test sets in the same sparse space.
	if num_columns is not None:
		with tempfile.NamedTemporaryFile() as f:
			dump_categorical_df_to_svm_light(X, y, f.name, cat_columns,
											 num_columns)
	else:
		with tempfile.NamedTemporaryFile() as f:
			dump_categorical_df_to_svm_light(X, y, f.name, cat_columns)	
	prediction = _libfm.predict(f.name, method, coef, v_file, bias, m_sum, m_sum_sqr, dim, task)
	return prediction