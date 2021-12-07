from google.colab import files
import io
import pandas as pd

uploaded = files.upload()
data = io.BytesIO(uploaded['crypto_lookback30.csv'])  
raw_dataset = pd.read_csv(data)
dataset1 = raw_dataset.copy()

from sklearn import datasets,metrics,svm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
 
dataset = dataset1.copy()
dataset = dataset.dropna()
dataset.drop(columns=["Date"], inplace=True)

# split data sequentially so that each split contains data from roughly the same time period
train_dataset, intermediate_test_valid_dataset = train_test_split(dataset, test_size=0.4, shuffle=False,random_state=9)
valid_dataset, test_dataset = train_test_split(intermediate_test_valid_dataset, test_size=0.5, shuffle=False,random_state=9)
intermediate_test_features = test_dataset.sample(frac=1,random_state=1)

# shuffle with a consistent random state and then copy to desired variable
intermediate_train_features = train_dataset.sample(frac=1,random_state=5)
intermediate_valid_features = valid_dataset.sample(frac=1,random_state=5)

train_features = intermediate_train_features.copy()
valid_features = intermediate_valid_features.copy()
test_features =   intermediate_test_features.copy()


clf = GradientBoostingRegressor()
clf.fit(train_features.iloc[:, 0:-1], train_features.y)
ypredict = clf.predict((valid_features.iloc[:, 0:-1]))
#print(reg.score(test_dataset.iloc[:, 0:-1],test_dataset.iloc[:, -2:-1],sample_weight=None ))

''' 
#auto hyper parameter tuning 
parameters =  {'n_estimators': [25,50,100,150,200] , 'learning_rate':[0.1, 0.25,0.5,1], 'max_depth':[1,3,6,8,10]}
regr = GradientBoostingRegressor()
clf = GridSearchCV( estimator= regr, param_grid= parameters, scoring='neg_mean_absolute_error' ,  cv=10, n_jobs=-1)
clf.fit(train_features.iloc[:, 0:-1], train_features.y)
clf.best_params_  '''
"{'learning_rate': 0.25, 'max_depth': 6, 'n_estimators': 100}"
print("Mean squared error: %.2f" % mean_squared_error(valid_features.y, ypredict))
# The coefficient of determination: 1 is perfect prediction
print("mean_absolute_error: %.2f" % mean_absolute_error(valid_features.y, ypredict))

plt.scatter(valid_features.y, ypredict)
plt.xlabel('True Values [y]')
plt.ylabel('Predictions [y]')
lims = [0, max(valid_features.y)]
plt.xlim(lims)
plt.ylim(lims)
plt.plot(lims, lims)
plt.show() 