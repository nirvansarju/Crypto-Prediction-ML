#support vector regression
from sklearn import svm
from sklearn.model_selection import GridSearchCV,TimeSeriesSplit,train_test_split,PredefinedSplit
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

from google.colab import files
import io
import pandas as pd

uploaded = files.upload()
data = io.BytesIO(uploaded['crypto_lookback30.csv'])  
raw_dataset = pd.read_csv(data)
dataset1 = raw_dataset.copy()

dataset = dataset1.copy()
dataset = dataset.dropna()
dataset.drop(columns=["Date"], inplace=True)


# split data sequentially so that each split contains data from roughly the same time period
train_dataset, intermediate_test_valid_dataset = train_test_split(dataset, test_size=0.4, shuffle=False )
valid_dataset, test_dataset = train_test_split(intermediate_test_valid_dataset, test_size=0.5, shuffle=False)

# shuffle with a consistent random state and then copy to desired variable
intermediate_train_features = train_dataset.sample(frac=1,random_state=1)
intermediate_valid_features = valid_dataset.sample(frac=1,random_state=1)
intermediate_test_features = test_dataset.sample(frac=1,random_state=1)

train_features = intermediate_train_features.copy()
valid_features = intermediate_valid_features.copy()
test_features =   intermediate_test_features.copy()

''' 
#deferent kernels 
for colorr, kernell in [  ('red', 'poly'), ('blue', 'rbf'), ('green','sigmoid')]:
    regr = svm.SVR(kernel=kernell, degree=3)
    regr.fit(train_features.iloc[:, 0:-1], train_features.y )
    ypredict = regr.predict(valid_features.iloc[:, 0:-1])
    plt.scatter(valid_features.y, ypredict,  c=colorr, s=15, label= kernell)
    print("kernel: " +  kernell )
    
    # The mean squared error
    print("Mean squared error: %.2f" % mean_squared_error(valid_features.y, ypredict))
    # The coefficient of determination: 1 is perfect prediction
    print("mean_absolute_error: %.2f" % mean_absolute_error(valid_features.y, ypredict))

plt.legend()
plt.xlabel('True Values [y]')
plt.ylabel('Predictions [y]')
lims = [0, max(valid_features.y)]
plt.xlim(lims)
plt.ylim(lims)
plt.plot(lims, lims)
plt.show()

#different degrees for poly kernel
for degreee in [2,3,4,9]:
    regr = svm.SVR(kernel='poly', degree= degreee)
    regr.fit(train_features.iloc[:, 0:-1], train_features.y)
    ypredict = regr.predict(valid_features.iloc[:, 0:-1])
    # The mean squared error
    print("degree: " +  str(degreee) )

    print("Mean squared error: %.2f" % mean_squared_error(valid_features.y, ypredict))
    # The coefficient of determination: 1 is perfect prediction
    print("Mean_absolute_error: %.2f" % mean_absolute_error(valid_features.y, ypredict))

    plt.scatter(valid_features.y, ypredict, s=15, label= 'poly degree: ' + str(degreee))

plt.legend()
plt.xlabel('True Values [y]')
plt.ylabel('Predictions [y]')
lims = [0, max(valid_features.y)]
plt.xlim(lims)
plt.ylim(lims)
plt.plot(lims, lims)
plt.show()  
   '''
''' #auto hyper parameter tuning 

split = TimeSeriesSplit(n_splits=5).split(train_features.iloc[:, 0:-1])
parameters =  {'kernel': ['poly'] , 'degree':[2,3,4], 'epsilon':[ 0.01,0.1,1,10], 'C':[0.1,0.5,1,10]}
regr = svm.SVR()
clf = GridSearchCV( estimator= regr, param_grid= parameters, scoring='neg_mean_absolute_error' ,  cv=split, n_jobs=-1)
clf.fit(train_features.iloc[:, 0:-1], train_features.y)
clf.best_params_ 
   '''

 #{'C': 200000, 'degree': 2, 'epsilon': 10, 'kernel': 'poly'}
regr = svm.SVR(C= 10, degree=3, epsilon= 0.01, kernel= 'poly')
regr.fit(train_features.iloc[:, 0:-1], train_features.y)
ypredict = regr.predict(test_features.iloc[:, 0:-1])
# The mean squared error

print("Mean squared error: %.2f" % mean_squared_error(test_features.y, ypredict))
# The coefficient of determination: 1 is perfect prediction
print("Mean_absolute_error: %.2f" % mean_absolute_error(test_features.y, ypredict))


plt.scatter(test_features.y, ypredict, s=15, )

plt.legend()
plt.xlabel('True Values [y]')
plt.ylabel('Predictions [y]')
lims = [0, max(test_features.y)]
plt.xlim(lims)
plt.ylim(lims)
plt.plot(lims, lims)
plt.show()  