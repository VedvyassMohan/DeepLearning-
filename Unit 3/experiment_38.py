import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
x_train = x_train[..., np.newaxis]; x_test = x_test[..., np.newaxis]
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    MaxPooling2D((2,2)), Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2,2)), Flatten(), Dense(128, activation='relu'),
    Dropout(0.3), Dense(10, activation='softmax')])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, to_categorical(y_train), epochs=5, batch_size=128, validation_split=0.1)
loss, acc = model.evaluate(x_test, to_categorical(y_test), verbose=0)
pred = np.argmax(model.predict(x_test, verbose=0), axis=1)
print('Test loss:', round(float(loss), 4), 'Test accuracy:', round(float(acc), 4))
print('Confusion matrix:\n', confusion_matrix(y_test, pred))
print(classification_report(y_test, pred, digits=3))
