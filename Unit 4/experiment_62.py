import numpy as np
from sklearn.metrics import mean_absolute_error,mean_squared_error
actual=np.array([420,435,410,460,480,455,470],float)
pred=np.array([415,440,405,450,490,460,465],float)
mse=mean_squared_error(actual,pred)
print("Actual:",actual); print("Predicted:",pred)
print("MAE:",round(mean_absolute_error(actual,pred),3))
print("MSE:",round(mse,3)); print("RMSE:",round(mse**.5,3))
