import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import tensorflow

import keras

from keras.utils import plot_model
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Reshape
from keras.layers import Conv2D, Dropout, MaxPooling2D, UpSampling2D
from keras.layers import BatchNormalization, GaussianNoise
from keras.callbacks import History

from numpy import array

# definitions
M = 128
L = 5
numepochs = 25


# One hot encoding
rawdata = np.zeros(M)
i = 0
while(i<M):
    rawdata[i] = i
    i += 1

data = to_categorical(rawdata)
data = data.astype(int)
valdata = data

# Define Structure
#input_shape = (1, M)

model = Sequential()
model.add(Dense(M, activation='relu', input_dim=M))     # input layer, output size M
model.add(Dense(16*L*L, activation='relu'))
model.add(BatchNormalization())
model.add(Reshape((4*L, 4*L, 1)))
model.add(Conv2D(M, kernel_size=(3, 3), padding='same', activation='relu'))         # Conv_1, M 3x3 filters
model.add(BatchNormalization())
model.add(MaxPooling2D())
model.add(Conv2D(2*M, kernel_size=(3, 3), padding='same', activation='relu'))       # Conv_2, 2M 3x3 filters
model.add(BatchNormalization())
model.add(MaxPooling2D())
model.add(Conv2D(1, kernel_size=(3, 3), padding='same', activation='sigmoid'))      # Conv_3, 1 3x3 filters
# End Encoder layer
model.add(UpSampling2D(size=(4,4)))
model.add(GaussianNoise(stddev=0.1))        # add noise to channel simulation
# Start Decoder
model.add(Conv2D(2*M, kernel_size=(5, 5), padding='same', activation='relu'))       # Conv_4, 2M 5x5 filters
model.add(BatchNormalization())
model.add(MaxPooling2D())
model.add(Conv2D(2*M, kernel_size=(3, 3), padding='same', activation='relu'))       # Conv_5, 2M 3x3 filters
model.add(BatchNormalization())
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(M, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(M, activation='softmax'))                 # output layer, output size M





model.summary()
# model train
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = History()
model.fit(data, data, epochs=numepochs, validation_data=(valdata, valdata), batch_size=32, callbacks=[history])

model.save('trained_cae.h5')





import matplotlib.pyplot as plt

# Plot training & validation accuracy values
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
