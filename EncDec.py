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
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Activation, Flatten, Reshape, Input, Lambda
from keras.layers import Conv2D, Dropout, MaxPooling2D, UpSampling2D, ZeroPadding2D
from keras.layers import BatchNormalization, GaussianNoise
from keras.callbacks import History, EarlyStopping, ModelCheckpoint

from numpy import array

# definitions
M = 64
L = 5
numepochs = 100
delta = 1
batchsize = 32


# One hot encoding
rawdata = np.zeros(M)
i = 0
while(i<M):
    rawdata[i] = i
    i += 1

data = to_categorical(rawdata)
#from numpy import reshape
data = data.astype(int)
ydata = np.reshape(data, (data.shape[0], 1, 1, data.shape[1]))
yvaldata = ydata
xdata = np.reshape(data, (data.shape[0], 1, 1, data.shape[1]))
xvaldata = xdata
# Define Structure
#input_shape = data.shape


def Encoder():
    input_img = Input(shape=(1, 1, M))
    e1 = Dense(M, activation='relu')(input_img)     # input layer, output size M
    e2 = Dense(16*L*L, activation='relu')(e1)
    e3 = BatchNormalization()(e2)
    e4 = Reshape((4*L, 4*L, -1))(e3)
    e5 = Conv2D(M, kernel_size=(3, 3), padding='same', activation='relu')(e4)          # Conv_1, M 3x3 filters
    e6 = BatchNormalization()(e5)
    e7 = MaxPooling2D()(e6)
    e8 = Conv2D(2*M, kernel_size=(3, 3), padding='same', activation='relu')(e7)       # Conv_2, 2M 3x3 filters
    e9 = BatchNormalization()(e8)
    e10 = MaxPooling2D()(e9)
    e11 = Conv2D(1, kernel_size=(3, 3), padding='same')(e10)      # Conv_3, 1 3x3 filters 
    sigmoid = Lambda(lambda x: 1/(1 + 2.718281 ** (-x*delta)))(e11)
    return Model(input_img, sigmoid)
# End Encoder layer

def Decoder():
    input_img = Input(shape=(L, L, 1))
    d01 = UpSampling2D(size=(2,2))(input_img)
    d02 = ZeroPadding2D(padding=4)(d01)
    d03 = GaussianNoise(stddev=0.1)(d02)        # add noise to channel simulation
# Start Decoder
    d1 = Conv2D(2*M, kernel_size=(5, 5), padding='same', activation='relu')(d03)       # Conv_4, 2M 5x5 filters
    d2 = BatchNormalization()(d1)
    d3 = MaxPooling2D()(d2)
    d4 = Conv2D(2*M, kernel_size=(3, 3), padding='same', activation='relu')(d3)       # Conv_5, 2M 3x3 filters
    d5 = BatchNormalization()(d4)
    d6 = MaxPooling2D()(d5)
    d7 = Reshape((1, 1, -1))(d6)
    d8 = Dense(M, activation='relu')(d7)
    d9 = BatchNormalization()(d8)
    d10 = Dense(M, activation='softmax')(d9)                 # output layer, output size M
    return Model(input_img, d10)

x = Input(shape=(1, 1, M))
callbacks = [EarlyStopping(monitor='loss', patience=50),
             ModelCheckpoint(filepath='saved_weights.h5', monitor='loss', save_best_only=True, save_weights_only=True),
             History()]

adam = keras.optimizers.Adam(clipnorm=1., clipvalue=0.5, amsgrad=True)   # clipnorm is necessary to prevent gradients from blowing up (weights = NaN)

for delta in np.arange(1, 4):
    print('delta = ', delta)
    cae = Model(x, Decoder()(Encoder()(x)))

    # model train
    cae.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
    if delta != 1:
        cae.load_weights('saved_weights.h5')
    
    cae.fit(xdata, ydata, epochs=numepochs, validation_data=(xvaldata, yvaldata), batch_size=batchsize, callbacks=callbacks)


"""numepochs = 25
for delta in [10, 100]:

    encoder = cae.layers[1]
    predicted = encoder.predict(xdata)

    print('delta = ', delta)
    cae = Model(x, Decoder()(Encoder()(x)))

    # model train
    cae.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
    if delta != 1:
        cae.load_weights('saved_weights.h5')
    
    
    cae.fit(xdata, ydata, epochs=numepochs, validation_data=(xvaldata, yvaldata), batch_size=batchsize, callbacks=callbacks)
    cae
    cae.save_weights('saved_weights.h5')"""

cae.save('trained_cae.h5')

encoder = cae.layers[1]
decoder = cae.layers[2]
encoder.save_weights('encoder_weights.h5')
decoder.save_weights('decoder_weights.h5')
encoder.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
decoder.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
encoder.load_weights('encoder_weights.h5')
decoder.load_weights('decoder_weights.h5')
encoder.save('trained_encoder.h5')
decoder.save('trained_decoder.h5')




predicted = encoder.predict(xdata)

import matplotlib.pyplot as plt

# Plot training & validation accuracy values
plt.plot(callbacks[2].history['acc'])
plt.plot(callbacks[2].history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(callbacks[2].history['loss'])
plt.plot(callbacks[2].history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()


import vis
from vis.utils import utils
from vis.visualization import visualize_saliency