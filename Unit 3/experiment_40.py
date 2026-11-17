import numpy as np

feature_map = np.array([[1, 3, 2, 4],
                        [5, 6, 1, 2],
                        [2, 4, 8, 7],
                        [3, 1, 5, 9]], dtype=float)

def pool(matrix, mode):
    out = np.zeros((matrix.shape[0]//2, matrix.shape[1]//2))
    for i in range(0, matrix.shape[0], 2):
        for j in range(0, matrix.shape[1], 2):
            block = matrix[i:i+2, j:j+2]
            out[i//2,j//2] = np.max(block) if mode == 'max' else np.mean(block)
    return out

print('Original feature map:\n', feature_map)
print('Max pooling:\n', pool(feature_map, 'max'))
print('Average pooling:\n', pool(feature_map, 'average'))
