from sklearn.preprocessing import OneHotEncoder
import numpy as np
from keras.utils import to_categorical
from numpy import array

M = 16
L = 2
input_shape = (1, M)
i = 0

data = np.zeros(M)
while(i<M):
    data[i] = i
    i += 1
# one hot encode
encoded = to_categorical(data)
print(encoded)