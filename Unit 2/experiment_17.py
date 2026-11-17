from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

X,y = make_classification(n_samples=500, n_features=20, n_informative=5, random_state=7)
Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=0.3,random_state=7)
for depth in [None, 4]:
    model = DecisionTreeClassifier(max_depth=depth, random_state=7).fit(Xtr,ytr)
    print(f'max_depth={depth}: train={accuracy_score(ytr,model.predict(Xtr)):.3f}, test={accuracy_score(yte,model.predict(Xte)):.3f}')
print('A smaller tree can reduce overfitting by limiting model complexity.')
