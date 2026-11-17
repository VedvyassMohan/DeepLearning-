from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report

X,y=load_iris(return_X_y=True)
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.25,random_state=5,stratify=y)
model=MLPClassifier(hidden_layer_sizes=(16,8),max_iter=1500,random_state=5).fit(Xtr,ytr)
p=model.predict(Xte)
print('Accuracy:',round(accuracy_score(yte,p),4))
print(classification_report(yte,p,target_names=load_iris().target_names))
