## Myopia
# Necessary packages & libraries
import pandas as pd
import numpy as np
import tensorflow as tf
# Necessary functions
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import utils
from kerastuner.tuners import RandomSearch, Hyperband, BayesianOptimization
# Loading data
spreadSheet = 'Myopia_Data.xlsx'
data = pd.read_excel(spreadSheet)
# Creating datasets
train_proportion = 0.8 # 80% for training, 20% for test
valid_proportion = 0.2 # used as a valdation set later
train_size = int(train_proportion * len(data))
test_size  = len(data) - train_size
train_data, test_data = data.iloc[:train_size,:], data.iloc[train_size:,:]
# Splitting in y and x for simplicity
(x_train, y_train) = train_data.iloc[:, 3:].values, train_data.iloc[:, 2].values
(x_test, y_test)   = test_data.iloc[:, 3:].values, test_data.iloc[:, 2].values
x_train, x_test = x_train.astype('float32'), x_test.astype('float32')
# Data assesment for future unification (0 in x axis means that we consider every column separately)
mean = x_train.mean(axis=0) # mean value
std = x_train.std(axis=0)   # standard diviation
# Making the data unification (except the already unified ones)
x_train[:,0] -= mean[0]
x_train[:,0] /= std[0]
x_train[:,:-2] -= mean[:-2]
x_train[:,:-2] /= std[:-2]

x_test[:,0] -= mean[0]
x_test[:,0] /= std[0]
x_test[:,:-2] -= mean[:-2]
x_test[:,:-2] /= std[:-2]
# Categories (# 2 categories yes/no)
y_train, y_test  =  utils.to_categorical(y_train, 2), utils.to_categorical(y_test , 2)
classes = ['not having Myopia','having Myopia']
# Function of creating NN
def build_model(hp):
    model = Sequential()
    activation_choice = hp.Choice('activation', values=['relu', 'sigmoid', 'tanh', 'elu', 'selu'])
    model.add(Dense(units=hp.Int('units_input',
                                   min_value=x_test.shape[1],   # min neuron number - 15
                                   max_value=10*x_test.shape[1],# max neuron number - 150
                                   step=32),
                    input_dim=x_test.shape[1],
                    activation=activation_choice))
    # number of layers
    for i in range(hp.Int('num_layers', 2, 5)): # hidden layers from 1 to 4
        model.add(
            keras.layers.Dense(
                units=hp.Int("units_" + str(i), min_value=4, max_value=20, step=32),
                activation=activation_choice)
            )
    model.add(Dense(2, activation='softmax'))   # yes/no
    model.compile(
        optimizer=hp.Choice('optimizer', values=['adam','rmsprop','SGD']),
        loss='categorical_crossentropy',
        metrics=['accuracy'])
    return model
# Tuner creation
tuner = RandomSearch(
    build_model,                 # function of model building
    objective='val_accuracy',    # metrics, which we use for optimization
    max_trials=10,               # max trials for learning
    directory='test_directory',  # placa where trained NN saved
    overwrite=True               # this things is necessary! To remove "Reloading from existing project" after 2nd and other attempts
    )
# The best models (top 10)
tuner.search_space_summary()
tuner.search(x_train,
             y_train,
             batch_size=20,
             epochs=3,
             validation_split=valid_proportion,
             )
# Taking best n
n = 3
models = tuner.get_best_models(num_models=n)
# Using test date for the best three (from top to bottom)
for model in models:
  model.summary()
  model.evaluate(x_test, y_test)
  print()