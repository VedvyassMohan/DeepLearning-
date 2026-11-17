import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error

X=np.array([[500],[700],[900],[1100],[1300],[1500],[1700],[1900]],float)
y=np.array([20,27,35,42,50,58,65,73],float)
model=MLPRegressor(hidden_layer_sizes=(8,),activation='relu',solver='lbfgs',max_iter=3000,random_state=2).fit(X,y)
p=model.predict(X)
print('Actual prices:',y)
print('Predicted prices:',np.round(p,2))
print('Mean squared error:',round(mean_squared_error(y,p),3))
