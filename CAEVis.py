import keras
import numpy as np

from keras.models import load_model
from keras.utils import to_categorical


# load trained model for visualization
model = load_model('trained_cae.h5')

M = 128

rawdata = np.zeros(M)
#rawdata = rawdata.astype(int)
i = 0
while(i<M):
    rawdata[i] = i
    i += 1

data = to_categorical(rawdata)
data = data.astype(int)
valdata = data

idx = data[0]

from matplotlib import pyplot as plt
#%matplotlib inline
plt.rcParams['figure.figsize'] = (18, 6)

plt.imshow(data[idx, ...])