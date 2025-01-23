# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import RepeatedStratifiedKFold
from skopt import BayesSearchCV
from skopt.space import Integer
import joblib
from data import getTrainNum, getTestNum

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


x, y = getTrainNum()
x_val = x[-50:]
x = x[:-50]
y_val = y[-50:]
y = y[:-50]
x_test = getTestNum()
test_data = pd.read_csv('C:\\Users\\Danie\\Desktop\\work\\kaggle\\titanic\\test.csv')

params = {
    'hidden_layer_1': Integer(3, 30),  # Size of the first hidden layer
    'hidden_layer_2': Integer(0, 30),  # Size of the second hidden layer (0 means no layer)
    'hidden_layer_3': Integer(0, 30),  # Size of the third hidden layer
    'hidden_layer_4': Integer(0, 30),  # Size of the third hidden layer
    'activation': ['tanh', 'relu', 'elu', 'sigmoid'],
    'alpha': (0, 0.1, 'log-uniform'),
    'l1': (0,0.1, 'log-uniform'),
    'learning_rate_init': (0.00001,0.1, 'log-uniform'),
    'dropout_rate': (0,0.35, 'log-uniform'),
    'sort_layers': [False, True],
}

class CustomMLPClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, hidden_layer_1=10, hidden_layer_2=0, hidden_layer_3=0, hidden_layer_4=0, 
                 activation='relu', alpha=0, learning_rate_init=0.001, sort_layers=True, l1=0,
                 dropout_rate=0.1):
        # Build the tuple of hidden_layer_sizes dynamically
        self.hidden_layer_1 = hidden_layer_1
        self.hidden_layer_2 = hidden_layer_2
        self.hidden_layer_3 = hidden_layer_3
        self.hidden_layer_4 = hidden_layer_4
        self.activation = activation
        self.alpha = alpha
        self.learning_rate_init = learning_rate_init
        self.dropout_rate = dropout_rate
        self.l1 = l1
        self.sort_layers = sort_layers
        self._create_model()
    
    def _create_model(self):
        # Build the tuple of hidden_layer_sizes dynamically
        layers = []
        hidden_layers = [self.hidden_layer_1, self.hidden_layer_2, self.hidden_layer_3, self.hidden_layer_4]
        input_size = 10
        if (self.sort_layers):
            hidden_layers.sort(reverse=True)
        for size in hidden_layers:
            if size > 0:
                layers.append(nn.Linear(input_size, size))
                if (self.activation == 'relu'):
                    layers.append(nn.ReLU())
                elif (self.activation == 'tanh'):
                    layers.append(nn.Tanh())
                else:
                    layers.append(nn.ELU())
                layers.append(nn.Dropout(p=self.dropout_rate))
                layers.append(nn.BatchNorm1d(size))
                input_size = size
        layers.append(nn.Linear(input_size, 1))  # Output layer for binary classification
        layers.append(nn.Sigmoid())
        self.model = nn.Sequential(*layers)
        self._initialize_weights()
    
    def _initialize_weights(self):
        for layer in self.model:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def fit(self, X, y):
        X_tensor = torch.tensor(X.values, dtype=torch.float32)
        y_tensor = torch.tensor(y.values, dtype=torch.float32)
        criterion = nn.BCELoss()  # Binary Cross Entropy Loss
        #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate_init, weight_decay=self.alpha)
        for _ in range(200):  # Example epochs, adjust as needed
            optimizer.zero_grad()
            outputs = self.model(X_tensor).squeeze()
            loss = criterion(outputs, y_tensor)
            l1_penalty = sum(p.abs().sum() for p in self.model.parameters()) * self.l1
            loss += l1_penalty
            loss.backward()
            optimizer.step()
        return self

    def predict(self, X):
        self.model.eval()
        if torch.is_tensor(X):
            X_tensor = X
        else:
            X_tensor = torch.tensor(X.values, dtype=torch.float32)
        with torch.no_grad():
            outputs = self.model(X_tensor).squeeze()
        return (outputs.numpy() > 0.5).astype(int)

    def score(self, X, y):
        X_tensor = torch.tensor(X.values, dtype=torch.float32)
        y_tensor = torch.tensor(y.values, dtype=torch.float32)
        y_pred = self.predict(X_tensor)
        return torch.mean((y_pred == y_tensor).float()).numpy()
    
    def get_params(self, deep=True):
        # Return parameters for cloning and grid search
        return {
            'hidden_layer_1': self.hidden_layer_1,
            'hidden_layer_2': self.hidden_layer_2,
            'hidden_layer_3': self.hidden_layer_3,
            'hidden_layer_4': self.hidden_layer_4,
            'activation':self.activation,
            'alpha':self.alpha,
            'learning_rate_init':self.learning_rate_init,
            'dropout_rate':self.dropout_rate,
            'sort_layers': self.sort_layers,
            'l1': self.l1,
        }
    
    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        self._create_model()
        return self


cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=1)
model = BayesSearchCV(estimator=CustomMLPClassifier(), search_spaces=params, n_jobs=-1, cv=cv, n_iter=50, verbose=3)
model.fit(x, y)
print("Best parameters found:", model.best_params_)
joblib.dump(model, 'C:\\Users\\Danie\\Desktop\\work\\kaggle\\titanic\\network_bayes_search.pkl')
# 0.08
print(np.mean(model.predict(x_val) != y_val))

predictions = model.predict(x_test)
output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv('C:\\Users\\Danie\\Desktop\\work\\kaggle\\titanic\\submission.csv', index=False)
print("Your submission was successfully saved!")