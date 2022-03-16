from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.datasets import mnist


def create_model():
    model = Sequential()
    model.add(Dense(10, input_shape=(28*28,), activation='sigmoid'))
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy']
                  )
    return model


model = create_model()



# load dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(len(x_train), 28*28)
x_test = x_test.reshape(len(x_test), 28*28)
model.fit(
    x=x_train,
    y=y_train,
    epochs=10
)

result = model.evaluate(x=x_test, y=y_test)
print(f'accuracy is {result}')
model.save('models/simple_ann.h5')
