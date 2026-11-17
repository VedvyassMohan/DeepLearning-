import tensorflow as tf
from tensorflow.keras import layers, models

# Folder layout: leaves/train/<class_name>/ and leaves/validation/<class_name>/
train_ds=tf.keras.utils.image_dataset_from_directory('leaves/train',image_size=(224,224),batch_size=32)
val_ds=tf.keras.utils.image_dataset_from_directory('leaves/validation',image_size=(224,224),batch_size=32)
num_classes=len(train_ds.class_names)
base=tf.keras.applications.ResNet50(weights='imagenet',include_top=False,input_shape=(224,224,3))
base.trainable=False
inputs=layers.Input((224,224,3))
x=tf.keras.applications.resnet50.preprocess_input(inputs)
x=base(x,training=False); x=layers.GlobalAveragePooling2D()(x)
x=layers.Dropout(0.3)(x); outputs=layers.Dense(num_classes,activation='softmax')(x)
model=models.Model(inputs,outputs)
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.fit(train_ds,validation_data=val_ds,epochs=5)
loss,acc=model.evaluate(val_ds)
print('Plant disease validation accuracy:',round(float(acc),4))
