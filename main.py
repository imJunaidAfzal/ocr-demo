import keras
import pandas as pd
from keras.datasets import mnist
import numpy as np
from keras.models import Sequential
from keras.layers import InputLayer, Reshape, Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet import ResNet50
from tensorflow.keras import optimizers
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
# from imutils import build_montages
import matplotlib.pyplot as plt
import argparse
import cv2

# dataset = pd.read_csv('dataset/A_Z Handwritten Data.csv')


# Loading Dataset
((train_data, train_labels), (test_data, test_labels)) = mnist_datset = mnist.load_data()
train_data = np.array(train_data, dtype='float64')

train_data = np.expand_dims(train_data, axis=-1)
train_data /= 255.0

test_data = np.expand_dims(test_data, axis=-1)
print(train_data.shape)

test_data = np.array(test_data, dtype='float64')
test_data /= 255.0

le = LabelBinarizer()
train_labels = le.fit_transform(train_labels)
test_labels = le.fit_transform(test_labels)

# Image Data Generator

aug = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.05,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.15,
    horizontal_flip=False,
    fill_mode="nearest"
)

# Training OCR

EPOCHS = 4
INIT_LR = 1e-1
BS = 128

# my_opt = optimizers.SGD(learning_rate=INIT_LR, decay=INIT_LR / EPOCHS)
#
# my_model = keras.models.Sequential()
# my_model.add(InputLayer(input_shape=(28, 28, 1)))
# my_model.add(Dense())
# my_model.add(Reshape(None, 32, 32, 3))
#
# model = ResNet50(input_shape=(32, 32, 3), include_top=False)
# # model.build(input_shape=(None, 28, 28, 1))
# model.compile(loss='categorical_crossentropy',
#               optimizer=my_opt,
#               metrics=['accuracy']
#               )
# my_model.add(model)
#
# print("[INFO] training network...")
# H = my_model.fit(
#     x=train_data,
#     y=train_data,
#     batch_size=None,
#     epochs=EPOCHS,
#     verbose='auto',
#     validation_split=0.1,
# )
#
# my_model.save('models/ocr_v1', save_format='h5')



model_ = Sequential()
model_.add(Conv2D(300, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
model_.add(MaxPooling2D(pool_size=(2, 2)))

#model_.add(BatchNormalization())
#model_.add(Dropout(0.2))


model_.add(Conv2D(150, kernel_size=(3, 3), activation='relu') )
model_.add(MaxPooling2D(pool_size=(2, 2)))
#model_.add(BatchNormalization())
#model_.add(Dropout(0.2))

model_.add(Dropout(0.25))
model_.add(Flatten())
model_.add(Dense(128, activation='relu'))
model_.add(Dropout(0.5))
model_.add(Dense(10, activation='softmax'))


model_.compile(loss=keras.losses.categorical_crossentropy,
              optimizer='adam',
              metrics=['accuracy'])


history = model_.fit(train_data, train_labels,
                     batch_size=BS, 
                     epochs=EPOCHS,
                     verbose=1,
                     validation_data=(test_data, test_labels)
                     )

score = model_.evaluate(test_data, test_labels, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

model_.save('models/ocr_v1.h5')
