# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from skopt import BayesSearchCV
from skopt.space import Integer
import joblib
from data import getTrainNum, getTestNum
from sklearn.ensemble import BaggingClassifier


# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


x, y = getTrainNum()
x_test = getTestNum()
test_data = pd.read_csv('C:\\Users\\Danie\\Desktop\\work\\kaggle\\titanic\\test.csv')
x_val = x[-50:]
x = x[:-50]
y_val = y[-50:]
y = y[:-50]

params = {
    'n_restarts_optimizer': Integer(0, 500),
    'warm_start': [True, False],
}

sub = GaussianProcessClassifier(n_restarts_optimizer=500,warm_start=True)
model = BaggingClassifier(estimator=sub,n_estimators=50,max_features=8,verbose=3)
model.fit(x, y)
joblib.dump(model, 'C:\\Users\\Danie\\Desktop\\work\\kaggle\\titanic\\gaussian_ensemable_bayes_search.pkl')

predictions = model.predict(x_test)
output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv('C:\\Users\\Danie\\Desktop\\work\\kaggle\\titanic\\submission.csv', index=False)
print("Your submission was successfully saved!")

print(model.score(x_val,y_val))