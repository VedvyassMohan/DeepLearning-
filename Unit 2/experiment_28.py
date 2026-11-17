from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

X,y=make_moons(n_samples=500,noise=.25,random_state=6)
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.25,random_state=6)
for activation in ['relu','logistic','tanh']:
    m=MLPClassifier(hidden_layer_sizes=(20,10),activation=activation,max_iter=1500,random_state=6).fit(Xtr,ytr)
    print(activation,'accuracy=',round(accuracy_score(yte,m.predict(Xte)),4),'iterations=',m.n_iter_)
