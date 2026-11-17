import numpy as np

image = np.array([[1, 2, 1, 0, 1],
                  [0, 1, 2, 1, 0],
                  [1, 2, 3, 2, 1],
                  [0, 1, 2, 1, 0],
                  [1, 0, 1, 2, 1]], dtype=float)
kernel = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]], dtype=float)

def convolution2d(img, filt):
    h, w = img.shape; kh, kw = filt.shape
    result = np.zeros((h-kh+1, w-kw+1))
    for i in range(result.shape[0]):
        for j in range(result.shape[1]):
            result[i,j] = np.sum(img[i:i+kh, j:j+kw] * filt)
    return result

feature_map = convolution2d(image, kernel)
print('Input image:\n', image)
print('Convolution feature map:\n', feature_map)
# 2x2 max pooling, stride 2
pooled = np.array([[np.max(feature_map[i:i+2, j:j+2])
                    for j in range(0, feature_map.shape[1]-1, 2)]
                   for i in range(0, feature_map.shape[0]-1, 2)])
print('Max pooled feature map:\n', pooled)
