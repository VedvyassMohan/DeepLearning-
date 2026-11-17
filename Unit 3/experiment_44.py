import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import fashion_mnist

(x_train,y_train),(x_test,y_test)=fashion_mnist.load_data()
x_train=x_train[...,None].astype('float32')/255.0
x_test=x_test[...,None].astype('float32')/255.0

def make_model(use_bn):
    model=models.Sequential([layers.Input((28,28,1)), layers.Conv2D(32,3)])
    if use_bn: model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu')); model.add(layers.MaxPooling2D())
    model.add(layers.Flatten()); model.add(layers.Dense(64,activation='relu'))
    model.add(layers.Dense(10,activation='softmax'))
    model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
    return model

for label, use_bn in [('Without batch normalization',False),('With batch normalization',True)]:
    model=make_model(use_bn)
    history=model.fit(x_train,y_train,epochs=5,batch_size=128,validation_split=.1,verbose=0)
    loss,acc=model.evaluate(x_test,y_test,verbose=0)
    print(label,'final train accuracy:',round(history.history['accuracy'][-1],4),
          'test accuracy:',round(float(acc),4))
