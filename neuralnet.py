import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler


# Make NumPy printouts easier to read.
np.set_printoptions(precision=3, suppress=True)

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Dense
from keras import optimizers

column_names = ["Date","x_1","x_2","x_3","x_4","x_5","x_6","x_7","x_8","x_9","x_10","x_11","x_12","x_13","x_14","x_15",
"x_16","x_17","x_18","x_19","x_20","x_21","x_22","x_23","x_24","x_25","x_26","x_27","x_28","x_29","x_30","x_31","x_32",
"x_33","x_34","x_35","x_36","x_37","x_38","x_39","x_40","x_41","x_42","x_43","x_44","x_45","x_46","x_47","x_48","x_49",
"x_50","x_51","x_52","x_53","x_54","x_55","x_56","x_57","x_58","x_59","x_60","x_61","x_62","x_63","x_64","x_65","x_66",
"x_67","x_68","x_69","x_70","x_71","x_72","x_73","x_74","x_75","x_76","x_77","x_78","x_79","x_80","x_81","x_82","x_83",
"x_84","y"]

raw_dataset = pd.read_csv("crypto_lookback30.csv")
dataset = raw_dataset.copy()

# trim columns with nan
dataset.tail()
dataset.isna().sum()
dataset = dataset.dropna()

# drop the date row, not useful for prediction.
dataset.drop(columns=["Date"],inplace=True)

# create train, valid, test splits.

# shuffle and split data at the same time
train_dataset, intermediate_test_valid_dataset = train_test_split(dataset, test_size=0.4, shuffle=True)
valid_dataset, test_dataset = train_test_split(intermediate_test_valid_dataset, test_size=0.5, shuffle=True)

train_features = train_dataset.copy()
valid_features = valid_dataset.copy()
test_features = test_dataset.copy()


test_labels = test_features.pop('y')
train_labels = train_features.pop('y')
valid_labels = valid_features.pop('y')

# # split data sequentially so that each split contains data from roughly the same time period
# train_dataset, intermediate_test_valid_dataset = train_test_split(dataset, test_size=0.4, shuffle=False)
# valid_dataset, test_dataset = train_test_split(intermediate_test_valid_dataset, test_size=0.5, shuffle=False)

# intermediate_train_features = train_dataset.sample(frac=1,random_state=1)
# intermediate_valid_features = valid_dataset.sample(frac=1,random_state=1)
# intermediate_test_features = test_dataset.sample(frac=1,random_state=1)

# train_features = intermediate_train_features.copy()
# valid_features = intermediate_valid_features.copy()
# test_features = intermediate_test_features.copy()

# test_labels = test_features.pop('y')
# train_labels = train_features.pop('y')
# valid_labels = valid_features.pop('y')

# DNN  ==========================================================

normalizer = tf.keras.layers.Normalization(axis=-1)
normalizer.adapt(np.array(train_features))

# # go wider and then reduce -  learning rate may need to be tweaked
# def build_and_compile_model(norm):
#   model = keras.Sequential([norm])
#   model.add(Dense(units=128, activation='relu'))
#   model.add(Dense(units=256, activation='relu'))
#   model.add(Dense(units=512, activation='relu'))
#   model.add(Dense(units=1024, activation='relu')) # widest layer
#   model.add(Dense(units=1, activation='linear'))
  
#   model.compile(loss='mean_absolute_error', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
#   return model

# dnn_model = build_and_compile_model(normalizer)
# dnn_model.summary()

# history = dnn_model.fit(
#                         train_features,
#                         train_labels,
#                         verbose=1, 
#                         shuffle=False,
#                         epochs=100
#                         )
# # DNN  ==========================================================

# # LSTM ==========================================================
# # train_features = train_dataset.copy()
# # valid_features = valid_dataset.copy()

# # train_labels = train_features.pop('y')
# # valid_labels = valid_features.pop('y')

# # normalizer = tf.keras.layers.Normalization(axis=-1)
# # normalizer.adapt(np.array(train_features))

# # train_features = train_features.values.reshape(806,84,-1)
# # train_labels = train_labels.values.reshape(806,1,-1)

# # valid_features = valid_features.values.reshape(269,84,-1)
# # valid_labels = valid_labels.values.reshape(269,1,-1)

# # normalizer = tf.keras.layers.Normalization(axis=-1)
# # normalizer.adapt(np.array(train_features))

# # def build_and_compile_model(norm):
# #   model = keras.Sequential([norm])
# #   model.add(LSTM(units = 50, return_sequences = True, input_shape = (84,1)))
# #   model.add(Dropout(0.2)) 
# #   model.add(LSTM(units = 60, return_sequences = True))
# #   model.add(Dropout(0.2)) 
# #   model.add(LSTM(units = 80, return_sequences = True))
# #   model.add(Dropout(0.2)) 
# #   model.add(LSTM(units = 120))
# #   model.add(Dense(units = 1, activation='linear'))

# #   model.compile(loss='mean_absolute_error',
# #                 optimizer=tf.keras.optimizers.Adam(clipvalue=0.5, learning_rate=0.001))
# #   return model


# # dnn_model = build_and_compile_model(normalizer)
# # dnn_model.summary()

# # history = dnn_model.fit(
# #                         train_features,
# #                         train_labels,
# #                         verbose=1, 
# #                         batch_size=64,
# #                         shuffle=False,
# #                         epochs=10
# #                       )
# # LSTM ==========================================================


# valid_results = {}

# valid_results['dnn_model'] = dnn_model.evaluate(valid_features, valid_labels, verbose=1)

# pd.DataFrame(valid_results, index=['Mean absolute error [y]']).T
# print(pd.DataFrame(valid_results, index=['Mean absolute error [y]']).T)

# valid_predictions = dnn_model.predict(valid_features).flatten()


# a = plt.axes(aspect='equal')
# plt.scatter(valid_labels, valid_predictions)
# plt.xlabel('True Values [y]')
# plt.ylabel('Predictions [y]')
# lims = [0, 700]
# plt.xlim(lims)
# plt.ylim(lims)
# c = plt.plot(lims, lims)

# plot predictions against labels
# b = plt.axes()
# plt.figure(figsize=(8, 6), dpi=80)
# plt.plot(range(0, 269),valid_predictions,'r-',label='predictions')
# plt.plot(range(0, 269),valid_labels,'b-',label='labels')
# plt.xlabel('Date [y]')
# plt.ylabel('Price [y]')
# plt.xlim([0, 269])
# y_limit = np.amax(valid_predictions)
# plt.ylim([0, y_limit])
# d = plt.plot(lims, lims)


# rmse_score = mean_squared_error(valid_labels,valid_predictions, squared=False)
# print("RMSE: %.2f" % rmse_score)

# mse = tf.keras.losses.MeanSquaredError()
# mse_score = mse(valid_labels, valid_predictions).numpy()
# print("MSE: %.2f" % mse_score)

# print(valid_labels - valid_predictions)

#=======================TEST DATA EVALUATION - FOR AFTER PARAMETERS ARE TUNED===



def build_and_compile_model(norm):
  model = keras.Sequential([norm])
  model.add(Dense(units=128, activation='relu'))
  model.add(Dense(units=256, activation='relu'))
  model.add(Dense(units=512, activation='relu'))
  model.add(Dense(units=1024, activation='relu')) # widest layer
  model.add(Dense(units=1, activation='linear'))
  
  model.compile(loss='mean_absolute_error', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
  return model

dnn_model = build_and_compile_model(normalizer)
dnn_model.summary()

history = dnn_model.fit(
                        train_features,
                        train_labels,
                        verbose=1, 
                        shuffle=False,
                        epochs=200
                        )

test_results = {}

test_results['dnn_model'] = dnn_model.evaluate(test_features, test_labels, verbose=0)
pd.DataFrame(test_results, index=['Mean absolute error [y]']).T
print(pd.DataFrame(test_results, index=['Mean absolute error [y]']).T)

test_predictions = dnn_model.predict(test_features).flatten()

mse = tf.keras.losses.MeanSquaredError()
mse_score = mse(test_labels, test_predictions).numpy()

plt.scatter(test_labels, test_predictions)
plt.xlabel('True Values [x]')
plt.ylabel('Predictions [y]')
lims = [0, 5000]
plt.xlim(lims)
plt.ylim(lims)
plt.plot(lims, lims)


# plot predictions against labels
plt.figure(figsize=(8, 6), dpi=80)
plt.plot(range(0, 269),test_predictions,'r-',label='predictions')
plt.plot(range(0, 269),test_labels,'b-',label='labels')
plt.xlabel('Sample [x]')
plt.ylabel('Price [y]')
# plt.xlim([0, 269])
# y_limit = np.amax(np.maximum(test_predictions,test_labels))
# plt.ylim([0, y_limit])
# plt.plot(lims, lims)

plt.show()

# error = test_predictions - test_labels
# plt.hist(error, bins=25)
# plt.xlabel('Prediction Error [y]')
# _ = plt.ylabel('Count')

# mse = tf.keras.metrics.mean_absolute_error()
# mse_score = mse(test_labels, test_predictions).numpy()
# print("MSE: %.2f" % mse_score)
#=======================TEST DATA EVALUATION - FOR AFTER PARAMETERS ARE TUNED===

