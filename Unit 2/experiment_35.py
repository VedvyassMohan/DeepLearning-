from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

X,y=make_classification(n_samples=700,n_features=30,n_informative=5,flip_y=.12,random_state=10)
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.3,random_state=10)
# sklearn MLPClassifier has no dropout parameter; alpha supplies L2 regularization as a practical regularization comparison.
for label,alpha in [('Baseline (no L2)',0.000001),('Regularized (L2)',0.05)]:
    m=MLPClassifier(hidden_layer_sizes=(80,40),alpha=alpha,max_iter=700,random_state=10).fit(Xtr,ytr)
    print(label,'train=',round(accuracy_score(ytr,m.predict(Xtr)),3),'test=',round(accuracy_score(yte,m.predict(Xte)),3))
print('For actual dropout layers, use a framework such as TensorFlow/Keras or PyTorch.')
