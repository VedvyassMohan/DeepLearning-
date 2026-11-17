import numpy as np

for dtype in [np.float16, np.float32, np.float64]:
    a=dtype(0.1); b=dtype(0.2); c=dtype(0.3)
    print(dtype.__name__, '0.1 + 0.2 =', a+b, '| difference from 0.3 =', float((a+b)-c))
print('Higher precision generally reduces representation error, but decimal fractions may not be exact in binary.')
