from keras.models import load_model
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('5.png')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

gray = np.array(gray, dtype='float64')
img_re = cv2.resize(gray, (28, 28))
img_re /= 225.0
# print(img_re.shape)
# cv2.imshow('img', img_re)
# cv2.waitKey(0)
model = load_model('models/mnist.h5')
# img_re = img_re.reshape(28*28)
# img_re = img_re.reshape(28*28)
# print(img_re.shape)
# input('--------------------------')
# img_re = np.expand_dims(img_re, axis=0)
# print(img_re.shape)
# input('--------------------------')
# img_re = np.array([img_re])
# print(img_re.shape)
# input('..............')
# res = model.predict(img_re)
# # print(res)
# result = []
# # print(res[0].shape)
#
# # imgplot = plt.imshow(img_re[0])
# # plt.show()
# # input('..............continue?..................')
#
# # print(res[0])
#
# for dd in res[0]:
#     if dd < 0.5:
#         result.append(0)
#     else:
#         result.append(1)
#
# print(res[0])
# print(result)
# for ind, dig in enumerate(result):
#     if dig == 1:
#         print(f'Number is: {ind}')
print(f'shape is {img_re.shape}')
img_re = img_re.reshape(28, 28, 1)
img_re = np.expand_dims(img_re, axis=0)
print(f'shape is {img_re.shape}')
res= model.predict(img_re)
result = []

for dd in res[0]:
    if dd < 0.5:
        result.append(0)
    else:
        result.append(1)

print(res[0])
# print(result)
ma = 0
num = -1
for ind, dig in enumerate(result):
    if ma < dig:
        # print(f'Number is: {ind}')
        ma = dig
        num = ind
# print(res)

print(f'number is {num}')

import keras
import tensorflow
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

num_classes = 10
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
y_train = keras.utils.np_utils.to_categorical(y_train, num_classes)
y_test = keras.utils.np_utils.to_categorical(y_test, num_classes)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255


score = model.evaluate(x_test, y_test, verbose=0)
print(x_test.shape)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

