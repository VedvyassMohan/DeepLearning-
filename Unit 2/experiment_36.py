from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import numpy as np

X,y=make_classification(n_samples=600,n_features=20,n_informative=6,random_state=11)
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.25,random_state=11)
for alpha in [0.0001,0.1]:
    m=MLPClassifier(hidden_layer_sizes=(30,15),alpha=alpha,max_iter=700,random_state=11).fit(Xtr,ytr)
    norm=np.sqrt(sum(np.sum(w*w) for w in m.coefs_))
    print('L2 alpha=',alpha,'weight norm=',round(float(norm),3),'train acc=',round(accuracy_score(ytr,m.predict(Xtr)),3),'test acc=',round(accuracy_score(yte,m.predict(Xte)),3))
