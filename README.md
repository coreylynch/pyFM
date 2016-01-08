# Factorization Machines in Python

This is a python implementation of Factorization Machines [1]. This uses stochastic gradient descent with adaptive regularization as a learning method, which adapts the regularization automatically while training the model parameters. See [2] for details. From libfm.org: "Factorization machines (FM) are a generic approach that allows to mimic most factorization models by feature engineering. This way, factorization machines combine the generality of feature engineering with the superiority of factorization models in estimating interactions between categorical variables of large domain."

[1] Steffen Rendle (2012): Factorization Machines with libFM, in ACM Trans. Intell. Syst. Technol., 3(3), May.
[2] Steffen Rendle: Learning recommender systems with adaptive regularization. WSDM 2012: 133-142

## Installation
```
python setup.py build_ext --inplace
```

## Dependencies
* numpy
* sklearn

## Training Representation
The easiest way to use this class is to represent your training data as lists of standard Python dict objects, where the dict elements map each instance's categorical and real valued variables to its values. Then use a [sklearn DictVectorizer](http://scikit-learn.org/dev/modules/generated/sklearn.feature_extraction.DictVectorizer.html#sklearn.feature_extraction.DictVectorizer) to convert them to a design matrix with a one-of-K or “one-hot” coding.

Here's a toy example
```python
import pylibfm
from sklearn.feature_extraction import DictVectorizer
import numpy as np
train = [
	{"user": "1", "item": "5", "age": 19},
	{"user": "2", "item": "43", "age": 33},
	{"user": "3", "item": "20", "age": 55},
	{"user": "4", "item": "10", "age": 20},
]
v = DictVectorizer()
X = v.fit_transform(train)
print X.toarray()
[[ 19.   0.   0.   0.   1.   1.   0.   0.   0.]
 [ 33.   0.   0.   1.   0.   0.   1.   0.   0.]
 [ 55.   0.   1.   0.   0.   0.   0.   1.   0.]
 [ 20.   1.   0.   0.   0.   0.   0.   0.   1.]]
y = np.repeat(1.0,X.shape[0])
fm = pylibfm.FM()
fm.fit(X,y)
fm.predict(v.transform({"user": "1", "item": "10", "age": 24}))
```

## Getting Started
Here's an example on some real  movie ratings data. 

First get the smallest movielens ratings dataset from http://www.grouplens.org/system/files/ml-100k.zip.
ml-100k contains the files u.item (list of movie ids and titles) and u.data (list of user_id, movie_id, rating, timestamp).
```python
import numpy as np
from sklearn.feature_extraction import DictVectorizer
import pylibfm

# Read in data
def loadData(filename,path="ml-100k/"):
    data = []
    y = []
    users=set()
    items=set()
    with open(path+filename) as f:
        for line in f:
            (user,movieid,rating,ts)=line.split('\t')
            data.append({ "user_id": str(user), "movie_id": str(movieid)})
            y.append(float(rating))
            users.add(user)
            items.add(movieid)

    return (data, np.array(y), users, items)

(train_data, y_train, train_users, train_items) = loadData("ua.base")
(test_data, y_test, test_users, test_items) = loadData("ua.test")
v = DictVectorizer()
X_train = v.fit_transform(train_data)
X_test = v.transform(test_data)

# Build and train a Factorization Machine
fm = pylibfm.FM(num_factors=10, num_iter=100, verbose=True, task="regression", initial_learning_rate=0.001, learning_rate_schedule="optimal")

fm.fit(X_train,y_train)
Creating validation dataset of 0.01 of training for adaptive regularization
-- Epoch 1
Training MSE: 0.59477
-- Epoch 2
Training MSE: 0.51841
-- Epoch 3
Training MSE: 0.49125
-- Epoch 4
Training MSE: 0.47589
-- Epoch 5
Training MSE: 0.46571
-- Epoch 6
Training MSE: 0.45852
-- Epoch 7
Training MSE: 0.45322
-- Epoch 8
Training MSE: 0.44908
-- Epoch 9
Training MSE: 0.44557
-- Epoch 10
Training MSE: 0.44278
...
-- Epoch 98
Training MSE: 0.41863
-- Epoch 99
Training MSE: 0.41865
-- Epoch 100
Training MSE: 0.41874

# Evaluate
preds = fm.predict(X_test)
from sklearn.metrics import mean_squared_error
print "FM MSE: %.4f" % mean_squared_error(y_test,preds)
FM MSE: 0.9227

```
## Classification example
```python
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.cross_validation import train_test_split
import pylibfm

from sklearn.datasets import make_classification

X, y = make_classification(n_samples=1000,n_features=100, n_clusters_per_class=1)
data = [ {v: k for k, v in dict(zip(i, range(len(i)))).items()}  for i in X]

X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.1, random_state=42)

v = DictVectorizer()
X_train = v.fit_transform(X_train)
X_test = v.transform(X_test)

fm = pylibfm.FM(num_factors=50, num_iter=10, verbose=True, task="classification", initial_learning_rate=0.0001, learning_rate_schedule="optimal")

fm.fit(X_train,y_train)

Creating validation dataset of 0.01 of training for adaptive regularization
-- Epoch 1
Training log loss: 1.91885
-- Epoch 2
Training log loss: 1.62022
-- Epoch 3
Training log loss: 1.36736
-- Epoch 4
Training log loss: 1.15562
-- Epoch 5
Training log loss: 0.97961
-- Epoch 6
Training log loss: 0.83356
-- Epoch 7
Training log loss: 0.71208
-- Epoch 8
Training log loss: 0.61108
-- Epoch 9
Training log loss: 0.52705
-- Epoch 10
Training log loss: 0.45685

# Evaluate
from sklearn.metrics import log_loss
print "Validation log loss: %.4f" % log_loss(y_test,fm.predict(X_test))
Validation log loss: 1.5025

```
