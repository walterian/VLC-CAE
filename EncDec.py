import keras
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.callbacks import EarlyStopping, History, ModelCheckpoint
from keras.layers import (Activation, BatchNormalization, Conv2D, Dense,
                          Dropout, Flatten, GaussianNoise, Input, Lambda,
                          MaxPooling2D, Reshape, UpSampling2D, ZeroPadding2D)
from keras.models import Model, Sequential, load_model
from keras.utils import plot_model, to_categorical
from numpy import array
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# definitions
M = 64
L = 3
numepochs = 100
batchsize = M
C = 1 # average signal strength is unity (not 100% accurate but hopefully good enough to test)
SNR = 20

# One hot encoding/data generation
rawdata = np.zeros(M)
i = 0
while(i<M):
    rawdata[i] = i
    i += 1
data = to_categorical(rawdata)
data = data.astype(int)
ydata = np.reshape(data, (data.shape[0], 1, 1, data.shape[1]))
yvaldata = ydata
xdata = np.reshape(data, (data.shape[0], 1, 1, data.shape[1]))
xvaldata = xdata

def Encoder():
    input_img = Input(shape=(1, 1, M))
    e1 = Dense(M, activation='relu')(input_img)     # input layer, output size M
    e01 = BatchNormalization()(e1)
    e2 = Dense(16*L*L, activation='relu')(e01)
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
def Channel():
    input_img = Input(shape=(L, L, 1))
    d01 = UpSampling2D(size=(4,4))(input_img)
    d02 = ZeroPadding2D(padding=int((28-4*L)/2))(d01)
    d03 = GaussianNoise(stddev=np.sqrt(C/(10**(SNR/10))))(d02)        # add noise to channel simulation
    return Model(input_img, d03)
def Decoder():
    # Start Decoder
    input_enc = Input(shape=(28, 28, 1))
    d1 = Conv2D(2*M, kernel_size=(5, 5), padding='same', activation='relu')(input_enc)       # Conv_4, 2M 5x5 filters
    d2 = BatchNormalization()(d1)
    d3 = MaxPooling2D()(d2)
    d4 = Conv2D(2*M, kernel_size=(3, 3), padding='same', activation='relu')(d3)       # Conv_5, 2M 3x3 filters
    d5 = BatchNormalization()(d4)
    d6 = MaxPooling2D()(d5)
    d7 = Reshape((1, 1, -1))(d6)
    d8 = Dense(M, activation='relu')(d7)
    d9 = BatchNormalization()(d8)
    d10 = Dense(M, activation='softmax')(d9)                 # output layer, output size M
    return Model(input_enc, d10)
def Encoder_Test():
    input_img = Input(shape=(1, 1, M))
    e1 = Dense(M, activation='relu')(input_img)     # input layer, output size M
    e01 = BatchNormalization()(e1)
    e2 = Dense(16*L*L, activation='relu')(e01)
    e3 = BatchNormalization()(e2)
    e4 = Reshape((4*L, 4*L, -1))(e3)
    e5 = Conv2D(M, kernel_size=(3, 3), padding='same', activation='relu')(e4)          # Conv_1, M 3x3 filters
    e6 = BatchNormalization()(e5)
    e7 = MaxPooling2D()(e6)
    e8 = Conv2D(2*M, kernel_size=(3, 3), padding='same', activation='relu')(e7)       # Conv_2, 2M 3x3 filters
    e9 = BatchNormalization()(e8)
    e10 = MaxPooling2D()(e9)
    e11 = Conv2D(1, kernel_size=(3, 3), padding='same')(e10)      # Conv_3, 1 3x3 filters 
    sigmoid = Lambda(lambda x: 1/(1 + 2.718281 ** (-x*1000)))(e11)
    return Model(input_img, sigmoid)

inp = Input(shape=(1, 1, M))
# Define callbacks to be monitored during training time
callbacks = [History(),
             ModelCheckpoint(filepath='saved_weights.h5', monitor='loss', save_best_only=True, save_weights_only=True),
             EarlyStopping(monitor='loss', patience=50)]
adam = keras.optimizers.Adam(clipnorm=1., clipvalue=0.1, amsgrad=True)   # clipnorm is necessary to prevent gradients from blowing up (weights = NaN)
for delta in np.arange(1, 5):
    print('delta = ', delta)
    cae = Model(inp, Decoder()(Channel()(Encoder()(inp))))
    # model training
    cae.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
    if delta != 1:
        cae.load_weights('saved_weights.h5')
    cae.fit(xdata, ydata, epochs=numepochs, validation_data=(xvaldata, yvaldata), batch_size=batchsize, callbacks=callbacks)
    plt.plot(callbacks[0].history['loss'], label=('delta: '+str(delta)))
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper left')
plt.savefig('TrainingGraphs/inputs_'+str(M)+'_L_'+str(L)+'_snr_'+str(SNR)+'.png')
plt.clf()

cae = Model(inp, Decoder()(Channel()(Encoder_Test()(inp))))
cae.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
cae.load_weights('saved_weights.h5')
cae.save('Trained/inputs_'+str(M)+'_L_'+str(L)+'_snr_'+str(SNR)+'.h5')
            
#predict = cae.predict(xdata)
#print(predict)

'''
# Plot training & validation accuracy values
plt.plot(callbacks[0].history['acc'])
plt.plot(callbacks[0].history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()'''