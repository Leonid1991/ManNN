import os
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator

Dense = tf.keras.layers.Dense
Activation = tf.keras.layers.Activation
Dropout = tf.keras.layers.Dropout
Flatten = tf.keras.layers.Flatten
Conv2D = tf.keras.layers.Conv2D
MaxPooling2D = tf.keras.layers.MaxPooling2D
Sequential = tf.keras.models.Sequential

# foldes with data
train_dir, val_dir, test_dir = 'train', 'val', 'test'  # naming of the furture folders fro training 

img_shape = (2560,1920)                                                         # original size      
input_shape = (2560,1920, 3) # images' size
# nn data
epochs = 1
batch_size = 3
nb_train_samples = 56      # train samples' number  (both classes), number can be checkt at "train_generator"
nb_validation_samples = 12 # validation samples' number (both classes)
nb_test_samples = 12       # test samples' number (both classes)
## nn
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = datagen.flow_from_directory(train_dir, target_size=img_shape,
    batch_size=batch_size, class_mode='binary')
val_generator = datagen.flow_from_directory(val_dir, target_size=img_shape,
    batch_size=batch_size, class_mode='binary')
test_generator = datagen.flow_from_directory(test_dir, target_size=img_shape,
    batch_size=batch_size, class_mode='binary')




model.fit(train_generator, 
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=val_generator,
    validation_steps=nb_validation_samples // batch_size)

# scores = model.evaluate(test_generator, nb_test_samples // batch_size)
scores = model.evaluate(test_generator)

print("Accuracy on the test: %.2f%%" % (scores[1]*100))

