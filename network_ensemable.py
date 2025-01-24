# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



from skopt.space import Integer
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.base import BaseEstimator, ClassifierMixin
from skopt import BayesSearchCV
import joblib
from data import getTrainNum, getTestNum
from network import CustomMLPClassifier

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


params = {
    'hidden_layer_1': Integer(3, 30),  # Size of the first hidden layer
    'hidden_layer_2': Integer(0, 30),  # Size of the second hidden layer (0 means no layer)
    'hidden_layer_3': Integer(0, 30),  # Size of the third hidden layer
    'hidden_layer_4': Integer(0, 30),  # Size of the third hidden layer
    'activation': ['tanh', 'relu', 'sigmoid'],
    'alpha': (1e-10, 1, 'log-uniform'),
    'l1': (1e-10,1, 'log-uniform'),
    'learning_rate_init': (0.00001,0.1, 'log-uniform'),
    'dropout_rate': (0,0.35),
    'input_size': Integer(5,10)
}

x, y = getTrainNum()
x_val = x[-50:]
#x = x[:-50]
y_val = y[-50:]
#y = y[:-50]
x_test = getTestNum()

class BaggedMLPClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self,n_estimators=50, input_size=10,**kwards):
        self.n_estimators = n_estimators
        self.input_size = input_size
        self.inner = CustomMLPClassifier(**kwards)
        self.kwards = kwards

        self.model = BaggingClassifier(estimator=self.inner, n_estimators=n_estimators, max_features=input_size)
    
    def fit(self,X,y):
        self.model.fit(X,y)
    
    def predict(self, X):
        return self.model.predict(X)
    
    def score(self, X, y):
        return self.model.score(X,y)
    
    def get_params(self, deep=True):
        # Return all parameters, including the inner estimator's parameters
        params = {
            "n_estimators": self.n_estimators,
            "input_size": self.input_size,
            **self.kwards,
        }
        if deep:
            params.update(self.model.get_params(deep=deep))
        return params
    
    def set_params(self, **params):
        # Set parameters for both the BaggingClassifier and the inner estimator
        self.n_estimators = params.pop("n_estimators", self.n_estimators)
        self.input_size = params.pop("input_size", self.input_size)
        self.kwards.update(params)  # Update additional parameters for the inner estimator
        self.inner = CustomMLPClassifier(**self.kwards)
        self.model = BaggingClassifier(estimator=self.inner, n_estimators=self.n_estimators)
        return self


#temp = joblib.load('C:\\Users\\Danie\\Desktop\\work\\kaggle\\titanic\\network_bayes_search.pkl')
#temp = CustomMLPClassifier(**temp.best_params_)

cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=1)
model = BayesSearchCV(estimator=BaggedMLPClassifier(), search_spaces=params, n_jobs=-1, n_points=3, cv=cv, n_iter=50, verbose=3)
model.fit(x,y)

print("Best parameters found:", model.best_params_)
joblib.dump(model, 'C:\\Users\\Danie\\Desktop\\work\\kaggle\\titanic\\network_ensemable_search.pkl')

# 0.94
print(model.score(x_val,y_val))

#output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
#output.to_csv('C:\\Users\\Danie\\Desktop\\work\\kaggle\\titanic\\submission.csv', index=False)
print("Your submission was successfully saved!")