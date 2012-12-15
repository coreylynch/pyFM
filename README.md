# Factorization Machines in Python

Work in Progress

Cython bindings to Steffen Rendle’s [LibFM](http://www.libfm.org/), a c++ library implementing factorization machines for collaborative filtering and recommendation.

Factorization machines (FM) are a generic approach that allows to mimic most factorization models by feature engineering. This way, factorization machines combine the generality of feature engineering with the superiority of factorization models in estimating interactions between categorical variables of large domain.

compat.py implements an sklearn base.BaseEstimator to allow this library to be incorporated into a broader workflow in python.