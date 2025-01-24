# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from skopt import BayesSearchCV
from skopt.space import Integer
import joblib
from data import getTrainNum, getTestNum
from rotation_forest_main import RotationForest

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


x, y = getTrainNum()
x_test = getTestNum()
test_data = pd.read_csv('C:\\Users\\Danie\\Desktop\\work\\kaggle\\titanic\\test.csv')
x_val = x[-50:]
x = x[:-50]
y_val = y[-50:]
y = y[:-50]

# n_features=3, sample_prop=0.5, bootstrap=False
params = {
    'n_features': Integer(3, 7),
    'sample_prop': (0.1,0.9),
    'bootstrap': [True,False],
}

# cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=1)

# model = BayesSearchCV(estimator=RotationForest(), search_spaces=params, n_jobs=-1, cv=cv, n_iter=50, verbose=3, n_points=3)
model = RotationForest()
model.fit(x, y)


print(model.score(x_val,y_val))
predictions = model.predict(x_test)
output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv('C:\\Users\\Danie\\Desktop\\work\\kaggle\\titanic\\submission.csv', index=False)
print("Your submission was successfully saved!")