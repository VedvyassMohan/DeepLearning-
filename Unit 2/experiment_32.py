import numpy as np

actual=np.array([1,0,1,0,0,1])
prob=np.array([.90,.20,.75,.10,.35,.60])
eps=1e-15
p=np.clip(prob,eps,1-eps)
loss=-np.mean(actual*np.log(p)+(1-actual)*np.log(1-p))
pred=(prob>=.5).astype(int)
print('Binary cross-entropy:',round(float(loss),4))
print('Predicted labels:',pred)
print('Accuracy:',round(float(np.mean(pred==actual))*100,2),'%')
