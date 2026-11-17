from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

X,y=make_classification(n_samples=600,n_features=12,n_informative=7,random_state=8)
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.25,random_state=8)
for solver in ['sgd','adam']:
    m=MLPClassifier(hidden_layer_sizes=(24,),solver=solver,learning_rate_init=.01,max_iter=500,random_state=8).fit(Xtr,ytr)
    print(solver,'iterations=',m.n_iter_,'loss=',round(m.loss_,4),'test accuracy=',round(accuracy_score(yte,m.predict(Xte)),4))
print('Note: scikit-learn MLPClassifier supports SGD and Adam; RMSprop is not provided by this estimator.')
