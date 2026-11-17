import numpy as np

X=np.array([1.,2.,3.,4.]); y=2*X+1
for lr in [0.001,0.03,0.2]:
    w,b=0.,0.; losses=[]
    for epoch in range(100):
        err=w*X+b-y
        w-=lr*np.mean(err*X); b-=lr*np.mean(err)
        losses.append(np.mean((w*X+b-y)**2))
    print(f'learning_rate={lr}: final MSE={losses[-1]:.6f}, first epoch MSE={losses[0]:.4f}')
