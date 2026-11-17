import numpy as np
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import mean_squared_error

rng=np.random.default_rng(3)
X=rng.normal(size=(120,5)); y=3*X[:,0]+1.5*X[:,1]+rng.normal(0,.5,120)
models={'Ordinary':LinearRegression(),'L1 (Lasso)':Lasso(alpha=.1),'L2 (Ridge)':Ridge(alpha=1.0)}
for name,m in models.items():
    m.fit(X,y); p=m.predict(X)
    print(name, 'coefficients=',np.round(m.coef_,3), 'MSE=',round(mean_squared_error(y,p),3))
