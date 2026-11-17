from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np

X,y=load_breast_cancer(return_X_y=True)
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.25,random_state=4,stratify=y)
base=RandomForestClassifier(random_state=4)
params={'n_estimators':[50,100,150], 'max_depth':[3,5,None]}
grid=GridSearchCV(base,params,cv=3).fit(Xtr,ytr)
random=RandomizedSearchCV(base,params,n_iter=5,cv=3,random_state=4).fit(Xtr,ytr)
# Small sequential candidate search as a transparent Bayesian-optimization illustration
candidates=[(50,3),(100,3),(50,5),(100,5),(150,None)]
results=[]
for n,d in candidates:
    m=RandomForestClassifier(n_estimators=n,max_depth=d,random_state=4).fit(Xtr,ytr)
    results.append((accuracy_score(yte,m.predict(Xte)),n,d))
best=max(results)
print('Grid best:',grid.best_params_,'CV score:',round(grid.best_score_,3))
print('Random best:',random.best_params_,'CV score:',round(random.best_score_,3))
print('Sequential candidate best (illustrative):',best[1:],'test accuracy:',round(best[0],3))
