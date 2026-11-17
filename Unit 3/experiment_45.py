import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import fashion_mnist

(x_train,y_train),(x_test,y_test)=fashion_mnist.load_data()
x_train=x_train[...,None].astype('float32')/255.0
x_test=x_test[...,None].astype('float32')/255.0

def build(augment=False):
    seq=[]
    if augment:
        seq += [layers.RandomRotation(0.08), layers.RandomFlip('horizontal'), layers.RandomZoom(0.1)]
    seq += [layers.Conv2D(32,3,activation='relu',input_shape=(28,28,1)),
            layers.MaxPooling2D(),layers.Flatten(),layers.Dense(64,activation='relu'),
            layers.Dense(10,activation='softmax')]
    model=models.Sequential(seq)
    model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
    return model

for label,aug in [('Original training images',False),('With augmentation',True)]:
    model=build(aug); model.fit(x_train,y_train,epochs=5,batch_size=128,verbose=0)
    loss,acc=model.evaluate(x_test,y_test,verbose=0)
    print(label,'test accuracy:',round(float(acc),4))
