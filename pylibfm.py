import numpy as np
from sklearn import cross_validation
from sklearn.feature_extraction import DictVectorizer

class FM:
    def __init__(self, learn_rate=0.01, num_factors=10, num_iter=1, param_regular = (0,0,0.1), k0=True, k1=True, task='classification', verbose=False):
        self.num_factors = num_factors
        self.num_iter = num_iter
        self.learn_rate = learn_rate
        self.learn_rates = np.zeros(3)
        self.sum = np.zeros(self.num_factors)
        self.sum_sqr = np.zeros(self.num_factors)
        self.num_attr_groups = 1

        self.k0 = k0
        self.k1 = k1

        self.task = task
        # Regularization Parameters
        self.reg_0 = param_regular[0]
        self.reg_w = param_regular[1]
        self.reg_v = param_regular[2]
        
        # local parameters in the lambda_update step
        self.lambda_w_grad = 0.0
        self.lambda_v_grad = 0.0
        self.sum_f = 0.0
        self.sum_f_dash_f = 0.0
        self.verbose = verbose

    def fit(self, X, y):
        #vec = DictVectorizer()
        #sparse_one_hot = vec.fit_transform(X)

        # use sklearn to create validation
        X_train, validation, train_labels, validation_labels = cross_validation.train_test_split(
            X, y, test_size=0.1)

        self.num_attribute = X_train.shape[1]

        # Gradients
        self.grad_w = np.zeros(self.num_attribute)
        self.grad_v = np.zeros((self.num_factors, self.num_attribute))

        self.w0 = 0.0
        self.w = np.zeros(self.num_attribute)
        self.v = 0.01*np.random.random_sample((self.num_factors, self.num_attribute))

        val_idx = 0
        for epoch in xrange(self.num_iter):
            if self.verbose == True:
                print "Epoch #%d" % epoch
            for idx in xrange(X_train.shape[0]):
                self.sgd_theta_step(X_train[idx], train_labels[idx])
                if epoch > 0: # make no lambda steps in first iteration
                    if val_idx > (validation.shape[0]-1):
                        val_idx = 0
                    self.sgd_lambda_step(validation[val_idx], validation_labels[val_idx])
                    val_idx += 1

    def sgd_theta_step(self, instance, label):
        p = self.predict(instance)
        if self.task == "regression":
            mult = 2 * (p - label);
        else:
            mult = label * ( (1.0 / (1.0+np.exp(-label*p))) - 1.0)
        
        if self.k0:
            grad_0 = mult
            self.w0 -= self.learn_rate * (grad_0 + 2 * self.reg_0 * self.w0)

        if self.k1:
            for i in xrange(instance.size):
                feature = instance.indices[i]
                self.grad_w[feature] = mult * instance.data[i]
                self.w[feature] -= self.learn_rate * (self.grad_w[feature] + 2 * self.reg_w * self.w[feature])
        for f in xrange(self.num_factors):
            for i in xrange(instance.size):
                feature = instance.indices[i]
                self.grad_v[f,feature] = mult * (instance.data[i] * (self.sum[f] - self.v[f,feature] * instance.data[i]))
                self.v[f,feature] -= self.learn_rate * (self.grad_v[f,feature] + 2 * self.reg_v * self.v[f,feature])
     
    def sgd_lambda_step(self, validation_instance, validation_label):
        p = self.predict_scaled(validation_instance)
        if self.task == "regression":
            grad_loss = 2 * (p - validation_label)
        else:
            grad_loss = validation_label * ( (1.0 / (1.0+np.exp(-validation_label*p))) - 1.0)

        if self.k1:
            self.lambda_w_grad = 0.0
            for i in xrange(validation_instance.size):
                feature = validation_instance.indices[i]
                self.lambda_w_grad += validation_instance.data[i] * self.w[feature]
            self.lambda_w_grad = -2 * self.learn_rate * self.lambda_w_grad
            self.reg_w -= self.learn_rate * grad_loss * self.lambda_w_grad
            self.reg_w = max(0.0, self.reg_w)
        
        for f in xrange(self.num_factors):
            sum_f_dash = 0.0
            self.sum_f = 0.0
            self.sum_f_dash_f = 0.0
            for i in xrange(validation_instance.size):
                feature = validation_instance.indices[i]
                v_dash = self.v[f,feature] - self.learn_rate * (self.grad_v[f,feature] + 2 * self.reg_v * self.v[f,feature])
                sum_f_dash += v_dash * validation_instance.data[i]
                self.sum_f += self.v[f,feature] * validation_instance.data[i]
                self.sum_f_dash_f += v_dash * validation_instance.data[i] * self.v[f,feature] * validation_instance.data[i]
            self.lambda_v_grad = -2 * self.learn_rate * (sum_f_dash * self.sum_f - self.sum_f_dash_f)
            self.reg_v -= self.learn_rate * grad_loss * self.lambda_v_grad
            self.reg_v = max(0.0, self.reg_v)

    def predict(self, instance):
        result = 0.0
        if self.k0:
            result += self.w0
        if self.k1:
            for i in xrange(instance.size):
                feature = instance.indices[i]
                assert(feature < self.num_attribute)
                result += self.w[feature] * instance.data[i]
        for f in xrange(self.num_factors):
            self.sum[f] = 0.0
            self.sum_sqr[f] = 0.0
            for i in xrange(instance.size):
                feature = instance.indices[i]
                d = self.v[f,feature] * instance.data[i]
                self.sum[f] += d
                self.sum_sqr[f] += d*d
            result += 0.5 * (self.sum[f]*self.sum[f] - self.sum_sqr[f])
        return result

    def predict_scaled(self, instance):
        result = 0.0
        if self.k0:
            result += self.w0
        if self.k1:
            for i in xrange(instance.size):
                feature = instance.indices[i]
                assert(feature < self.num_attribute)
                w_dash = self.w[feature] - self.learn_rate * (self.grad_w[feature] + 2 * self.reg_w * self.w[feature])
                result += w_dash * instance.data[i]
        for f in xrange(self.num_factors):
            self.sum[f] = 0.0
            self.sum_sqr[f] = 0.0
            for i in xrange(instance.size):
                feature = instance.indices[i]
                v_dash = self.v[f,feature] - self.learn_rate * (self.grad_v[f,feature] + 2 * self.reg_v * self.v[f,feature])
                d = v_dash * instance.data[i]
                self.sum[f] += d
                self.sum_sqr += d*d
            result += 0.5 * (self.sum[f]*self.sum[f] - self.sum_sqr[f])
        return result