import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import fashion_mnist

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train = x_train[..., None].astype('float32')/255.0
x_test = x_test[..., None].astype('float32')/255.0

for filters, kernel_size in [(16,3), (32,3), (32,5)]:
    model = models.Sequential([
        layers.Input(shape=(28,28,1)),
        layers.Conv2D(filters, kernel_size, activation='relu'),
        layers.MaxPooling2D(), layers.Flatten(),
        layers.Dense(64, activation='relu'), layers.Dense(10, activation='softmax')])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=3, batch_size=128, verbose=0)
    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    print(f'Filters={filters}, filter size={kernel_size}x{kernel_size}, test accuracy={acc:.4f}')
