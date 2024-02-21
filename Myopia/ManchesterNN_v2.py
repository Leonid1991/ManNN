## Myopia
# Necessary packages & libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# Necessary functions
from keras import utils
from keras.models import Sequential
from keras.layers import Dense
# Loading data
spreadSheet = 'Myopia_Data.xlsx'
data = pd.read_excel(spreadSheet)
# Creating datasets
nepoch = 50
train_proportion = 0.8 # 80% for training, 20% for test
valid_proportion = 0.2 # used as a valdation set later
train_size = int(train_proportion * len(data))
test_size  = len(data) - train_size
train_data, test_data = data.iloc[:train_size,:], data.iloc[train_size:,:]
# Splitting in y and x for simplicity
(x_train, y_train) = train_data.iloc[:, 3:].values, train_data.iloc[:, 2].values
(x_test, y_test)   = test_data.iloc[:, 3:].values, test_data.iloc[:, 2].values
x_train, x_test = x_train.astype('float'), x_test.astype('float')
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
# Creation NN
model = Sequential() # classic, layer-by-layer model type
model.add(Dense(x_test.shape[1], activation='relu', input_shape=(x_train.shape[1],))) # 15 is due to the number of initial parameters columns
model.add(Dense(2, activation="sigmoid", kernel_initializer="normal"))                # yes/no
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]) # some random optimization routine
# Training procedure
history = model.fit(x_train, y_train, batch_size=20, epochs=nepoch, validation_split = valid_proportion, verbose=1)
# Model evaluation
scores = model.evaluate(x_test, y_test, verbose=0)
print("Accuracy of NN:", (scores[1]*100))
# Accessing loss values and accuracy from the history object
train_loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(train_loss) + 1)
# Plotting the loss curve
plt.plot(epochs, train_loss, 'g', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()