import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

rng=np.random.default_rng(4)
X=np.linspace(-3,3,100).reshape(-1,1); y=0.5*X[:,0]**2 + X[:,0] + rng.normal(0,1,100)
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.25,random_state=2)
for degree in [1,2,8]:
    model=make_pipeline(PolynomialFeatures(degree),LinearRegression()).fit(Xtr,ytr)
    print(f'degree {degree}: train MSE={mean_squared_error(ytr,model.predict(Xtr)):.3f}, test MSE={mean_squared_error(yte,model.predict(Xte)):.3f}')
