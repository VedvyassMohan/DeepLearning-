import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score, classification_report

# MNIST contains 28x28 grayscale handwritten digit images.
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
y_train_cat = to_categorical(y_train, 10)
model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(256, activation='relu'),
    Dropout(0.2),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train_cat, epochs=5, batch_size=128, validation_split=0.1)
loss, accuracy = model.evaluate(x_test, to_categorical(y_test, 10), verbose=0)
pred = np.argmax(model.predict(x_test, verbose=0), axis=1)
print('Test loss:', round(float(loss), 4))
print('Test accuracy:', round(float(accuracy), 4))
print(classification_report(y_test, pred, digits=3))
