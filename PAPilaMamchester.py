import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers import Dropout
from keras.preprocessing import image
from keras.layers import Conv2D, MaxPooling2D
import UutilsProv
from tensorflow.keras import utils
ROOT_DIR = '//maa1.cc.lut.fi/home/h18306/Desktop/New folder/'
y, eyeID, patID = UutilsProv.get_diagnosis(ROOT_DIR)
df_od, df_os = UutilsProv.read_clinical_data(ROOT_DIR)
print(y)
# datasets segmentation
train_proportion = 0.8 # 80% for training, 20% for test
valid_proportion = 0.2 # ussed as a valdation set later
train_size = int(train_proportion * len(y))
test_size  = len(y) - train_size
batch_size, nb_epoch  = 100, 10 
img_shape = (1934, 2576, 3) # passing a image shape
# NN (very simple, will be sub wit U-net)
model = Sequential()                                             
model.add(Conv2D(16, (3, 3), padding='same', input_shape=img_shape, activation='relu'))  
model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))                         
model.add(Dropout(0.25))                                         
model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(32, (3, 3), activation='relu'))                
model.add(MaxPooling2D(pool_size=(2, 2)))                       
model.add(Dropout(0.25))                                        
model.add(Flatten())                                            
model.add(Dense(200, activation='relu'))                        
model.add(Dropout(0.5))                                         
model.add(Dense(3, activation='softmax'))                      
# Info of the net
print(model.summary())

folder_path = ROOT_DIR + 'ExpertsSegmentations/ImagesWithContours/'
files = os.listdir(folder_path) 
# Filter the list of files to include only image files 
image_files = [f for f in files if f.endswith('.jpg')] 
# X data
x_train = []
for i in range(train_size):
    image = np.array(Image.open(os.path.join(folder_path, image_files[i])))
    x_train.append(image) 
x_train = np.array(x_train)  
x_test = []
for i in range(train_size,len(y)):
    image = np.array(Image.open(os.path.join(folder_path, image_files[i])))
    x_test.append(image)
x_test = np.array(x_test)
# Y data
y_train = y[:train_size]
y_test = y[train_size:len(y)]
classes=['no glaucoma', 'confirmed glaucoma', 'sufering glaucoma']
y_train, y_test = utils.to_categorical(y_train, 3), utils.to_categorical(y_test, 3) 
# Data normalization 
x_train, x_test = x_train.astype('float32'), x_test.astype('float32')
x_train = x_train / 255
x_test = x_test / 255

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# Training
myNN = model.fit(x_train, y_train, batch_size=batch_size, epochs=nb_epoch, validation_split=0.1, shuffle=True, verbose=2)
# assesment
scores = model.evaluate(x_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))
# result visualization
myNN_dict = myNN.history
acc_values = myNN_dict['accuracy']
val_acc_values = myNN_dict['val_accuracy']
epochs = range(1, len(acc_values) + 1)
plt.plot(epochs, acc_values, 'bo', label='Training acc')
plt.plot(epochs, val_acc_values, 'b', label='Validation acc')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
# Saving of the net
#model_json = model.to_json()
#json_file = open("cifar10_model.json", "w")
#json_file.write(model_json)
#json_file.close()
#model.save_weights("Fundus_model.h5")
# Check
i = -1
x = x_test[i]
x = np.expand_dims(x, axis=0)
prediction = np.argmax(model.predict(x))
print("Considered case, i = ", i)
print("Prognosis", classes[prediction])
print("Reality", classes[np.argmax(y_test[i])])
