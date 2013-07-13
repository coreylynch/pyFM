# encoding: utf-8
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
#
# Author: Corey Lynch <coreylynch9@gmail.com>
#
# License: BSD Style.

import numpy as np
import sys
from time import time

from libc.math cimport exp, log, pow
cimport numpy as np
cimport cython

np.import_array()

ctypedef np.float64_t DOUBLE
ctypedef np.int32_t INTEGER

# MODEL CONSTANTS
DEF REGRESSION = 0
DEF CLASSIFICATION = 1
DEF OPTIMAL = 0
DEF INVERSE_SCALING = 1

cdef class FM_fast(object):
    """Factorization Machine fitted by minimizing a regularized empirical loss with adaptive SGD.
    
    Parameters
    ----------
    w : np.ndarray[DOUBLE, ndim=1, mode='c']
    v : ndarray[DOUBLE, ndim=2, mode='c'] 
    num_factors : int 
    num_attributes : int 
    n_iter : int 
    k0 : int 
    k1 : int 
    w0 : double 
    t : double 
    t0 : double 
    l : double 
    power_t : double 
    min_target : double 
    max_target : double 
    eta0 : double 
    learning_rate_schedule : int 
    shuffle_training : int 
    task : int 
    seed : int 
    verbose : int 
    """

    cdef double w0
    cdef np.ndarray w
    cdef np.ndarray v
    cdef int num_factors
    cdef int num_attributes
    cdef int n_iter
    cdef int k0
    cdef int k1

    cdef DOUBLE t
    cdef DOUBLE t0
    cdef DOUBLE l
    cdef DOUBLE power_t
    cdef DOUBLE min_target
    cdef DOUBLE max_target
    cdef np.ndarray sum
    cdef np.ndarray sum_sqr
    cdef int task
    cdef int learning_rate_schedule
    cdef double learning_rate
    cdef int shuffle_training
    cdef int seed
    cdef int verbose

    cdef DOUBLE reg_0
    cdef DOUBLE reg_w
    cdef np.ndarray reg_v

    cdef np.ndarray grad_w
    cdef np.ndarray grad_v

    cdef DOUBLE sumloss
    cdef int count

    def __init__(self,
                  np.ndarray[DOUBLE, ndim=1, mode='c'] w,
                  np.ndarray[DOUBLE, ndim=2, mode='c'] v,
                  int num_factors,
                  int num_attributes,
                  int n_iter,
                  int k0,
                  int k1,
                  double w0,
                  double t,
                  double t0,
                  double power_t,
                  double min_target,
                  double max_target,
                  double eta0,
                  int learning_rate_schedule,
                  int shuffle_training,
                  int task,
                  int seed,
                  int verbose):

        self.w0 = w0
        self.w = w
        self.v = v
        self.num_factors = num_factors
        self.num_attributes = num_attributes
        self.n_iter = n_iter
        self.k0 = k0
        self.k1 = k1
        self.t = 1
        self.t0 = t0
        self.learning_rate = eta0
        self.power_t = power_t
        self.min_target = min_target
        self.max_target = max_target
        self.sum = np.zeros(self.num_factors)
        self.sum_sqr = np.zeros(self.num_factors)
        self.task = task
        self.learning_rate_schedule = learning_rate_schedule
        self.shuffle_training = shuffle_training
        self.seed = seed
        self.verbose = verbose

        self.reg_0 = 0.0
        self.reg_w = 0.0
        self.reg_v = np.zeros(self.num_factors)

        self.sumloss = 0.0
        self.count = 0

        self.grad_w = np.zeros(self.num_attributes)
        self.grad_v = np.zeros((self.num_factors, self.num_attributes))

    cdef _predict_instance(self, DOUBLE * x_data_ptr, 
                           INTEGER * x_ind_ptr, 
                           int xnnz):
        
        # Helper variables
        cdef DOUBLE result = 0.0
        cdef int feature
        cdef unsigned int i = 0
        cdef unsigned int f = 0
        cdef DOUBLE d

        # map instance variables to local variables
        cdef DOUBLE w0 = self.w0
        cdef np.ndarray[DOUBLE, ndim=1, mode='c'] w = self.w
        cdef np.ndarray[DOUBLE, ndim=2, mode='c'] v = self.v
        cdef np.ndarray[DOUBLE, ndim=1, mode='c'] sum_ = np.zeros(self.num_factors)
        cdef np.ndarray[DOUBLE, ndim=1, mode='c'] sum_sqr_ = np.zeros(self.num_factors)

        if self.k0 > 0:
            result += w0
        if self.k1 > 0:
            for i in range(xnnz):
                feature = x_ind_ptr[i]
                result += w[feature] * x_data_ptr[i]
        for f in range(self.num_factors):
            sum_[f] = 0.0
            sum_sqr_[f] = 0.0
            for i in range(xnnz):
                feature = x_ind_ptr[i]
                d = v[f, feature] * x_data_ptr[i]
                sum_[f] += d
                sum_sqr_[f] += d*d
            result += 0.5 * (sum_[f] * sum_[f] - sum_sqr_[f])

        # pass sum to sgd_theta
        self.sum = sum_
        return self._scale_prediction(result)

    cdef _predict_scaled(self, DOUBLE * x_data_ptr, 
                           INTEGER * x_ind_ptr, 
                           int xnnz):
        cdef DOUBLE result = 0.0
        cdef unsigned int i = 0
        cdef unsigned int f = 0
        cdef DOUBLE d
        cdef DOUBLE w_dash = 0.0
        cdef DOUBLE v_dash = 0.0

        # map instance variables to local variables
        cdef DOUBLE w0 = self.w0
        cdef np.ndarray[DOUBLE, ndim=1, mode='c'] w = self.w
        cdef np.ndarray[DOUBLE, ndim=2, mode='c'] v = self.v
        cdef np.ndarray[DOUBLE, ndim=1, mode='c'] grad_w = self.grad_w
        cdef np.ndarray[DOUBLE, ndim=2, mode='c'] grad_v = self.grad_v
        cdef np.ndarray[DOUBLE, ndim=1, mode='c'] sum_ = np.zeros(self.num_factors)
        cdef np.ndarray[DOUBLE, ndim=1, mode='c'] sum_sqr_ = np.zeros(self.num_factors)
        cdef DOUBLE learning_rate = self.learning_rate
        cdef DOUBLE reg_w = self.reg_w
        cdef np.ndarray[DOUBLE, ndim=1, mode='c'] reg_v = self.reg_v


        if self.k0 > 0:
            result += w0
        if self.k1 > 0:
            for i in xrange(xnnz):
                feature = x_ind_ptr[i]
                assert(feature < self.num_attributes)
                w_dash = w[feature] - learning_rate * (grad_w[feature] + 2 * reg_w * w[feature])
                result += w_dash * x_data_ptr[i]
        for f in xrange(self.num_factors):
            sum_[f] = 0.0
            sum_sqr_[f] = 0.0
            for i in xrange(xnnz):
                feature = x_ind_ptr[i]
                v_dash = v[f,feature] - learning_rate * (grad_v[f,feature] + 2 * reg_v[f] * v[f,feature])
                d = v_dash * x_data_ptr[i]
                sum_[f] += d
                sum_sqr_[f] += d*d
            result += 0.5 * (sum_[f]*sum_[f] - sum_sqr_[f])

        self.sum = sum_
        return result

    cdef _scale_prediction(self, DOUBLE p):

        if self.task == REGRESSION:
            p = min(self.max_target, p)
            p = max(self.min_target, p)
        elif self.task == CLASSIFICATION:
            p = 1.0 / (1.0 + exp(-p))
        return p
    
    def _predict(self, CSRDataset dataset):
        
        # Helper access variables
        cdef unsigned int i = 0
        cdef Py_ssize_t n_samples = dataset.n_samples
        cdef DOUBLE * x_data_ptr = NULL
        cdef INTEGER * x_ind_ptr = NULL
        cdef int xnnz
        cdef DOUBLE sample_weight = 1.0
        cdef DOUBLE y_placeholder
        cdef DOUBLE p = 0.0
    
        cdef np.ndarray[DOUBLE, ndim=1, mode='c'] return_preds = np.zeros(n_samples)
    
        for i in range(n_samples):
            dataset.next(& x_data_ptr, & x_ind_ptr, & xnnz, & y_placeholder,
                         & sample_weight)
            p = self._scale_prediction(self._predict_instance(x_data_ptr, x_ind_ptr, xnnz))
            return_preds[i] = p
        return return_preds
    
    cdef _sgd_theta_step(self, DOUBLE * x_data_ptr, 
                        INTEGER * x_ind_ptr, 
                        int xnnz,
                        DOUBLE y):
    
        cdef DOUBLE mult = 0.0
        cdef DOUBLE p
        cdef int feature
        cdef unsigned int i = 0
        cdef unsigned int f = 0
        cdef DOUBLE d
        cdef DOUBLE grad_0

        cdef DOUBLE w0 = self.w0
        cdef np.ndarray[DOUBLE, ndim=1, mode='c'] w = self.w
        cdef np.ndarray[DOUBLE, ndim=2, mode='c'] v = self.v
        cdef np.ndarray[DOUBLE, ndim=1, mode='c'] grad_w = self.grad_w
        cdef np.ndarray[DOUBLE, ndim=2, mode='c'] grad_v = self.grad_v
        cdef np.ndarray[DOUBLE, ndim=1, mode='c'] sum_ = np.zeros(self.num_factors)
        cdef DOUBLE learning_rate = self.learning_rate
        cdef DOUBLE reg_0 = self.reg_0
        cdef DOUBLE reg_w = self.reg_w
        cdef np.ndarray[DOUBLE, ndim=1, mode='c'] reg_v = self.reg_v

        p = self._predict_instance(x_data_ptr, x_ind_ptr, xnnz)

        if self.task == REGRESSION:
            p = min(self.max_target, p)
            p = max(self.min_target, p)
            mult = 2 * (p - y);
        else:
            mult = y * ( (1.0 / (1.0+exp(-y*p))) - 1.0)
        
        # Set learning schedule
        if self.learning_rate_schedule == OPTIMAL:
            self.learning_rate = 1.0 / (self.t + self.t0)

        elif self.learning_rate_schedule == INVERSE_SCALING:
            self.learning_rate = self.learning_rate / pow(self.t, self.power_t)
    
        if self.verbose > 0:
            self.sumloss += _squared_loss(p,y) if self.task == REGRESSION else _log_loss(p,y)
    
        # Update global bias
        if self.k0 > 0:
            grad_0 = mult
            w0 -= learning_rate * (grad_0 + 2 * reg_0 * w0)

        # Update feature biases
        if self.k1 > 0:
            for i in range(xnnz):
                feature = x_ind_ptr[i]
                grad_w[feature] = mult * x_data_ptr[i]
                w[feature] -= learning_rate * (grad_w[feature] 
                                   + 2 * reg_w * w[feature])

        # Update feature factor vectors

        for f in range(self.num_factors):
            for i in range(xnnz):
                feature = x_ind_ptr[i]
                grad_v[f,feature] = mult * (x_data_ptr[i] * (sum_[f] - v[f,feature] * x_data_ptr[i]))
                v[f,feature] -= learning_rate * (grad_v[f,feature] + 2 * reg_v[f] * v[f,feature])
    
        # Pass updated vars to other functions
        self.learning_rate = learning_rate
        self.w0 = w0
        self.w = w
        self.v = v
        self.grad_w = grad_w
        self.grad_v = grad_v

        self.t += 1
        self.count += 1

    cdef _sgd_lambda_step(self, DOUBLE * validation_x_data_ptr, 
                        INTEGER * validation_x_ind_ptr, 
                        int validation_xnnz,
                        DOUBLE validation_y):

        cdef DOUBLE sum_f
        cdef DOUBLE sum_f_dash
        cdef DOUBLE sum_f_dash_f
        cdef DOUBLE p
        cdef DOUBLE grad_loss
        cdef int feature
        cdef unsigned int i
        cdef unsigned int f
        cdef DOUBLE lambda_w_grad = 0.0
        cdef DOUBLE lambda_v_grad = 0.0
        cdef DOUBLE v_dash = 0.0

        cdef np.ndarray[DOUBLE, ndim=1, mode='c'] w = self.w
        cdef np.ndarray[DOUBLE, ndim=2, mode='c'] v = self.v
        cdef np.ndarray[DOUBLE, ndim=1, mode='c'] grad_w = self.grad_w
        cdef np.ndarray[DOUBLE, ndim=2, mode='c'] grad_v = self.grad_v
        cdef np.ndarray[DOUBLE, ndim=1, mode='c'] sum_ = np.zeros(self.num_factors)
        cdef DOUBLE learning_rate = self.learning_rate
        cdef DOUBLE reg_0 = self.reg_0
        cdef DOUBLE reg_w = self.reg_w
        cdef np.ndarray[DOUBLE, ndim=1, mode='c'] reg_v = self.reg_v

        p = self._predict_scaled(validation_x_data_ptr, validation_x_ind_ptr, validation_xnnz)
        if self.task == REGRESSION:
            p = min(self.max_target, p)
            p = max(self.min_target, p)
            grad_loss = 2 * (p - validation_y)
        else:
            grad_loss = validation_y * ( (1.0 / (1.0 + exp(-validation_y*p))) - 1.0)

        if self.k1 > 0:
            lambda_w_grad = 0.0
            for i in xrange(validation_xnnz):
                feature = validation_x_ind_ptr[i]
                lambda_w_grad += validation_x_data_ptr[i] * w[feature]
            lambda_w_grad = -2 * learning_rate * lambda_w_grad
            reg_w -= learning_rate * grad_loss * lambda_w_grad
            reg_w = max(0.0, reg_w)
        
        for f in xrange(self.num_factors):
            sum_f = 0.0
            sum_f_dash = 0.0
            sum_f_dash_f = 0.0

            for i in xrange(validation_xnnz):
                feature = validation_x_ind_ptr[i]
                v_dash = v[f,feature] - learning_rate * (grad_v[f,feature] + 2 * reg_v[f] * v[f,feature])
                sum_f_dash += v_dash * validation_x_data_ptr[i]
                sum_f += v[f,feature] * validation_x_data_ptr[i]
                sum_f_dash_f += v_dash * validation_x_data_ptr[i] * v[f,feature] * validation_x_data_ptr[i]
            lambda_v_grad = -2 * learning_rate * (sum_f_dash * sum_f - sum_f_dash_f)
            reg_v[f] -= learning_rate * grad_loss * lambda_v_grad
            reg_v[f] = max(0.0, reg_v[f])

        # Pass updated vars to other functions
        self.reg_w = reg_w
        self.reg_v = reg_v

    def fit(self, CSRDataset dataset, CSRDataset validation_dataset):
    
        # get the data information into easy vars
        cdef Py_ssize_t n_samples = dataset.n_samples
        cdef Py_ssize_t n_validation_samples = validation_dataset.n_samples
        
        cdef DOUBLE * x_data_ptr = NULL
        cdef INTEGER * x_ind_ptr = NULL
    
        cdef DOUBLE * validation_x_data_ptr = NULL
        cdef INTEGER * validation_x_ind_ptr = NULL
    
        # helper variables
        cdef int xnnz
        cdef DOUBLE y = 0.0
        cdef DOUBLE validation_y = 0.0
        cdef int validation_xnnz
        cdef unsigned int count = 0
        cdef unsigned int epoch = 0
        cdef unsigned int i = 0
    
        cdef DOUBLE sample_weight = 1.0
        cdef DOUBLE validation_sample_weight = 1.0

        for epoch in range(self.n_iter):
    
            if self.verbose > 0:
                print("-- Epoch %d" % (epoch + 1))
            self.count = 0
            self.sumloss = 0
            if self.shuffle_training:
                dataset.shuffle(self.seed)
    
            for i in range(n_samples):
                dataset.next( & x_data_ptr, & x_ind_ptr, & xnnz, & y,
                             & sample_weight)
    
                self._sgd_theta_step(x_data_ptr, x_ind_ptr, xnnz, y)
    
                if epoch > 0:
                    validation_dataset.next( & validation_x_data_ptr, & validation_x_ind_ptr,
                                             & validation_xnnz, & validation_y, 
                                             & validation_sample_weight)
                    self._sgd_lambda_step(validation_x_data_ptr, validation_x_ind_ptr,
                                          validation_xnnz, validation_y)
            if self.verbose > 0:
                error_type = "RMSE" if self.task == REGRESSION else "log loss"
                print "Training %s: %.5f" % (error_type, (self.sumloss / self.count))

cdef inline double max(double a, double b):
    return a if a >= b else b

cdef inline double min(double a, double b):
    return a if a <= b else b

cdef _log_loss(DOUBLE p, DOUBLE y):
    cdef DOUBLE z

    z = p * y
    # approximately equal and saves the computation of the log
    if z > 18:
        return exp(-z)
    if z < -18:
        return -z
    return log(1.0 + exp(-z))

cdef _squared_loss(DOUBLE p, DOUBLE y):
    return 0.5 * (p - y) * (p - y)

cdef class CSRDataset:
    """An sklearn ``SequentialDataset`` backed by a scipy sparse CSR matrix. This is an ugly hack for the moment until I find the best way to link to sklearn. """

    cdef Py_ssize_t n_samples
    cdef int current_index
    cdef int stride
    cdef DOUBLE *X_data_ptr
    cdef INTEGER *X_indptr_ptr
    cdef INTEGER *X_indices_ptr
    cdef DOUBLE *Y_data_ptr
    cdef np.ndarray feature_indices
    cdef INTEGER *feature_indices_ptr
    cdef np.ndarray index
    cdef INTEGER *index_data_ptr
    cdef DOUBLE *sample_weight_data

    def __cinit__(self, np.ndarray[DOUBLE, ndim=1, mode='c'] X_data,
                  np.ndarray[INTEGER, ndim=1, mode='c'] X_indptr,
                  np.ndarray[INTEGER, ndim=1, mode='c'] X_indices,
                  np.ndarray[DOUBLE, ndim=1, mode='c'] Y,
                  np.ndarray[DOUBLE, ndim=1, mode='c'] sample_weight):
        """Dataset backed by a scipy sparse CSR matrix.

        The feature indices of ``x`` are given by x_ind_ptr[0:nnz].
        The corresponding feature values are given by
        x_data_ptr[0:nnz].

        Parameters
        ----------
        X_data : ndarray, dtype=np.float64, ndim=1, mode='c'
            The data array of the CSR matrix; a one-dimensional c-continuous
            numpy array of dtype np.float64.
        X_indptr : ndarray, dtype=np.int32, ndim=1, mode='c'
            The index pointer array of the CSR matrix; a one-dimensional
            c-continuous numpy array of dtype np.int32.
        X_indices : ndarray, dtype=np.int32, ndim=1, mode='c'
            The column indices array of the CSR matrix; a one-dimensional
            c-continuous numpy array of dtype np.int32.
        Y : ndarray, dtype=np.float64, ndim=1, mode='c'
            The target values; a one-dimensional c-continuous numpy array of
            dtype np.float64.
        sample_weights : ndarray, dtype=np.float64, ndim=1, mode='c'
            The weight of each sample; a one-dimensional c-continuous numpy
            array of dtype np.float64.
        """
        self.n_samples = Y.shape[0]
        self.current_index = -1
        self.X_data_ptr = <DOUBLE *>X_data.data
        self.X_indptr_ptr = <INTEGER *>X_indptr.data
        self.X_indices_ptr = <INTEGER *>X_indices.data
        self.Y_data_ptr = <DOUBLE *>Y.data
        self.sample_weight_data = <DOUBLE *> sample_weight.data
        # Use index array for fast shuffling
        cdef np.ndarray[INTEGER, ndim=1,
                        mode='c'] index = np.arange(0, self.n_samples,
                                                    dtype=np.int32)
        self.index = index
        self.index_data_ptr = <INTEGER *> index.data

    cdef void next(self, DOUBLE **x_data_ptr, INTEGER **x_ind_ptr,
                   int *nnz, DOUBLE *y, DOUBLE *sample_weight):
        cdef int current_index = self.current_index
        if current_index >= (self.n_samples - 1):
            current_index = -1

        current_index += 1
        cdef int sample_idx = self.index_data_ptr[current_index]
        cdef int offset = self.X_indptr_ptr[sample_idx]
        y[0] = self.Y_data_ptr[sample_idx]
        x_data_ptr[0] = self.X_data_ptr + offset
        x_ind_ptr[0] = self.X_indices_ptr + offset
        nnz[0] = self.X_indptr_ptr[sample_idx + 1] - offset
        sample_weight[0] = self.sample_weight_data[sample_idx]

        self.current_index = current_index

    cdef void shuffle(self, seed):
        np.random.RandomState(seed).shuffle(self.index)