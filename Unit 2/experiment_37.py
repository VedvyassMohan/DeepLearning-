from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

X,y=make_classification(n_samples=700,n_features=25,n_informative=6,flip_y=.1,random_state=12)
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.3,random_state=12)
for label,alpha in [('Without regularization',0.000001),('With L2 regularization',0.1)]:
    model=MLPClassifier(hidden_layer_sizes=(50,25),alpha=alpha,max_iter=700,random_state=12).fit(Xtr,ytr)
    print(label)
    print('Training accuracy:',round(accuracy_score(ytr,model.predict(Xtr)),4))
    print('Testing accuracy:',round(accuracy_score(yte,model.predict(Xte)),4))
