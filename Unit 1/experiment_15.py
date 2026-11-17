import numpy as np

X = np.array([1., 2., 3., 4., 5.]); y = 2*X + 1
lr, epochs = 0.02, 30

def train(mode, batch_size=None):
    w, b, updates = 0.0, 0.0, 0
    rng = np.random.default_rng(1)
    for _ in range(epochs):
        order = rng.permutation(len(X)) if mode != 'batch' else np.arange(len(X))
        size = len(X) if mode == 'batch' else (1 if mode == 'stochastic' else batch_size)
        for start in range(0, len(X), size):
            ids = order[start:start+size]
            error = (w*X[ids] + b) - y[ids]
            dw = np.mean(error*X[ids]); db = np.mean(error)
            w -= lr*dw; b -= lr*db; updates += 1
    loss = np.mean((w*X+b-y)**2)
    return w, b, loss, updates

for mode, size in [('batch', None), ('stochastic', 1), ('mini-batch', 2)]:
    w,b,loss,updates = train(mode, size)
    print(f'{mode}: w={w:.3f}, b={b:.3f}, MSE={loss:.5f}, updates={updates}')
