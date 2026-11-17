from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

X,y=load_digits(return_X_y=True)
X=X/16.0
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.25,random_state=9,stratify=y)
model=MLPClassifier(hidden_layer_sizes=(128,64),max_iter=500,random_state=9).fit(Xtr,ytr)
print('Dataset: sklearn digits (8x8 handwritten digits; MNIST-style demonstration)')
print('Test accuracy:',round(accuracy_score(yte,model.predict(Xte)),4))
