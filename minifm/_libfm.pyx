# distutils: language = c++
# distutils: sources = pylibfm/src/reallib.h

import tempfile
from cython.operator cimport dereference as deref
cimport cython
from libcpp cimport bool, int

from libcpp.string cimport string

cimport numpy as np
import numpy as np

ctypedef np.float64_t DOUBLE
ctypedef np.int32_t INTEGER

cdef extern from "src/reallib.h":
	cdef cppclass Data:
		Data(int, bool, bool)
		void load(string)
		void debug()
		int num_feature
		int num_cases
		DOUBLE max_target
		DOUBLE min_target

	
	cdef cppclass fm_model:
		int num_attribute
		bool k0, k1
		int num_factor
		DOUBLE w0
		DOUBLE reg0
		DOUBLE regw 
		DOUBLE regv
		DOUBLE init_stdev
		DOUBLE init_mean
		void init()
		void debug()
		DVectorDouble w
		DMatrixDouble v
		DVectorDouble m_sum
		DVectorDouble m_sum_sqr
		

	cdef cppclass fm_learn_sgd_element_new:
		int num_iter
		fm_model *fm
		DOUBLE learn_rate
		DOUBLE min_target
		DOUBLE max_target
		int task
		void init()
		void learn(Data& train)
		DOUBLE predict_case(Data& data)
		void predict(Data& test, DVectorDouble& pred)


	cdef cppclass DVector[T]:
		int dim
		T* value
		DVector()
		DVector(int)
		void setSize(int)

	cdef cppclass DVectorDouble(DVector[DOUBLE]):
		void init_normal(DOUBLE, DOUBLE)


	cdef cppclass DMatrix[T]:
		T* value
		DMatrix()
		DMatrix(int, int)
		void save(string, bool)
		void load(string)
		void setSize(int, int)


	cdef cppclass DMatrixDouble(DMatrix[DOUBLE]):
		void init(DOUBLE, DOUBLE)


def train(train_data, method, task, dim):
	
	# Load training data
	
	# Mock train data
	cache_size = 100000000
	has_x = True
	has_xt = True
	verbose = False

	cdef Data *train = new Data(cache_size, has_x, has_xt)
	deref(train).load(train_data)
	if verbose is True:
		deref(train).debug()

	# Setup the factorization machine

	# Mock factorization machine data
	param_init_stdev = 0.1
	use_bias = True
	use_one_way = True

	cdef fm_model *fm = new fm_model()
	deref(fm).num_attribute = deref(train).num_feature
	deref(fm).init_stdev = param_init_stdev

	# k0,k1,k2': k0=use bias, k1=use 1-way interactions,
	# k2=dim of 2-way interactions
	deref(fm).k0 = use_bias
	deref(fm).k1 = use_one_way
	deref(fm).num_factor = dim
	deref(fm).init()

	method = "sgd"
	task = "c"
	num_iter = 100

	cdef fm_learn_sgd_element_new *fml
	# Setup the learning method
	if method == "sgd":
		fml = new fm_learn_sgd_element_new()
		deref(fml).num_iter = num_iter
		deref(fml).fm = fm
		deref(fml).max_target = deref(train).max_target
		deref(fml).min_target = deref(train).min_target
		if task == "r":
			deref(fml).task = 0
		else:
			deref(fml).task = 1		
		deref(fml).init()

		# No regularization for now
		fm.reg0 = 0.0
		fm.regw = 0.0
		fm.regv = 0.0

		# mock learn rate
		deref(fml).learn_rate = 0.01

		# learn
		deref(fml).learn(deref(train))

	# Set up out vars 
	cdef int n_features = deref(train).num_feature
	cdef np.ndarray[ndim=1, dtype=np.float64_t] coef = np.empty(n_features)
	cdef np.ndarray[ndim=1, dtype=np.float64_t] m_sum = np.empty(dim)
	cdef np.ndarray[ndim=1, dtype=np.float64_t] m_sum_sqr = np.empty(dim)

	for i in range(n_features):
		coef[i] = deref(fm).w.value[i]

	cdef DOUBLE bias = deref(fm).w0

	for i in range(dim):
		m_sum[i] = deref(fm).m_sum.value[i]
		m_sum_sqr[i] = deref(fm).m_sum_sqr.value[i]


	with tempfile.NamedTemporaryFile(delete=False) as f:
		deref(fm).v.save(f.name, False)		

	variance_file = f.name

	return coef, variance_file, bias, m_sum, m_sum_sqr

def predict(test_data, method, np.ndarray coef, variance_file, bias, m_sum,
			m_sum_sqr, dim, task):

	cache_size = 100000000
	has_x = True
	has_xt = True

	# Load test data
	cdef Data *test = new Data(cache_size, has_x, has_xt)
	deref(test).load(test_data)	

	# Load learned weights, m_sum, and m_sum_sqr into a new FM for prediction 
	use_bias = True
	use_one_way = True

	cdef fm_model *fm = new fm_model()
	deref(fm).num_attribute = deref(test).num_feature
	deref(fm).k0 = use_bias
	deref(fm).k1 = use_one_way
	deref(fm).num_factor = dim

	# from fm_model::init()
	deref(fm).w0 = bias
	deref(fm).w.setSize(deref(fm).num_attribute)
	deref(fm).v.setSize(dim, deref(fm).num_attribute)
	deref(fm).m_sum.setSize(dim)
	deref(fm).m_sum_sqr.setSize(dim)

	deref(fm).v.load(variance_file)
	
	for i in range(dim):
		deref(fm).m_sum.value[i] = m_sum[i]
		deref(fm).m_sum_sqr.value[i] = m_sum_sqr[i]

	for i in range(deref(test).num_feature):
		deref(fm).w.value[i] = coef[i]
	
	# set up learner to do prediction

	fml = new fm_learn_sgd_element_new()
	deref(fm).num_attribute = deref(test).num_feature
	deref(fml).fm = fm
	if task == "r":
		deref(fml).task = 0
	else:
		deref(fml).task = 1		

	# Predict
	cdef DVectorDouble pred
	pred.setSize(deref(test).num_cases)
	print(deref(test).num_cases)

	
	deref(fml).predict(deref(test), pred)

	cdef np.ndarray[ndim=1, dtype=np.float64_t] outs = np.empty(deref(test).num_cases)

	
	for i in range(deref(test).num_cases):
		outs[i] = pred.value[i]
	return outs
