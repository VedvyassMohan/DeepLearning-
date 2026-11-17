import tensorflow as tf
from tensorflow.keras import layers, models

# Arrange images in folders: dataset/train/cats, dataset/train/dogs,
# dataset/validation/cats, dataset/validation/dogs.
train_ds = tf.keras.utils.image_dataset_from_directory(
    'dataset/train', image_size=(128,128), batch_size=32, label_mode='binary')
val_ds = tf.keras.utils.image_dataset_from_directory(
    'dataset/validation', image_size=(128,128), batch_size=32, label_mode='binary')
model = models.Sequential([
    layers.Rescaling(1./255, input_shape=(128,128,3)),
    layers.Conv2D(32, 3, activation='relu'), layers.MaxPooling2D(),
    layers.Conv2D(64, 3, activation='relu'), layers.MaxPooling2D(),
    layers.Conv2D(128, 3, activation='relu'), layers.MaxPooling2D(),
    layers.Flatten(), layers.Dense(128, activation='relu'),
    layers.Dropout(0.4), layers.Dense(1, activation='sigmoid')])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(train_ds, validation_data=val_ds, epochs=10)
loss, accuracy = model.evaluate(val_ds)
print('Validation loss:', round(float(loss),4))
print('Validation accuracy:', round(float(accuracy),4))
