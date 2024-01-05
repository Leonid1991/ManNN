## Myopia
# Necessary packages & libraries
import pandas as pd
import numpy as np
# Necessary functions
from tensorflow.keras import utils
from keras.models import Sequential
from keras.layers import Dense
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
(x_train, y_train) = train_data.iloc[:, 5:].values, train_data.iloc[:, 2].values
(x_test, y_test)   = test_data.iloc[:, 5:].values, test_data.iloc[:, 2].values
x_train, x_test = x_train.astype('float32'), x_test.astype('float32')
# Data assesment for future unification (0 in x axis means that we consider every column separately)
mean = x_train.mean(axis=0) # mean value
std = x_train.std(axis=0)   # standard diviation
# Making the data unification
x_train -= mean
x_train /= std
x_test -= mean
x_test /= std
# Categories (# 2 categories yes/no)
y_train, y_test  =  utils.to_categorical(y_train, 2), utils.to_categorical(y_test , 2)
classes = ['not having Myopia','having Myopia']
# Creation NN
model = Sequential() # classic, layer-by-layer model type
model.add(Dense(x_test.shape[1], activation='relu', input_shape=(x_train.shape[1],))) # 13 is due to the number of initial parameters columns
model.add(Dense(2, activation="softmax", kernel_initializer="normal"))               # yes/no
model.compile(loss="categorical_crossentropy", optimizer="SGD", metrics=["accuracy"])# some random optimization routine
# Training procedure
model.fit(x_train, y_train, batch_size=20, epochs=20, validation_split = valid_proportion, verbose=1)
# Model evaluation
scores = model.evaluate(x_test, y_test, verbose=0)
print("Accuracy of NN:", (scores[1]*100))
# Prediction (local/some cases) of # ith case
prediction = model.predict(x_test) # usage of the train parameters
i = -1
print("Considered case, i = ",i)
print("Predicted value",np.argmax(prediction[i]))
print("Actual value",np.argmax(y_test[i]))