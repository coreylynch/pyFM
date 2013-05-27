# Factorization Machines in Python

This is a pure python implementation of Factorization Machines [1]. This uses stochastic gradient descent with adaptive regularization as a learning method, which adapts the regularization automatically while training the model parameters. See [2] for details. From libfm.org: "Factorization machines (FM) are a generic approach that allows to mimic most factorization models by feature engineering. This way, factorization machines combine the generality of feature engineering with the superiority of factorization models in estimating interactions between categorical variables of large domain."

[1] Steffen Rendle (2012): Factorization Machines with libFM, in ACM Trans. Intell. Syst. Technol., 3(3), May.
[2] Steffen Rendle: Learning recommender systems with adaptive regularization. WSDM 2012: 133-142

## Dependencies
* numpy
* sklearn

(Cython implementation in the works)