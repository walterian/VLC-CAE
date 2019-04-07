# This file contains an attempt at actually putting the network trained in EncDec.py to practice
import keras
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Model, load_model
import pandas as pd


SNR = 20
M = 64
# Generate random signal of length 32 with 64 possible values
sig = np.random.randint(0, 64, 100)

data = np.array(sig)
data = keras.utils.to_categorical(data, num_classes=64)
data = data.astype(int)
data = np.reshape(data, (data.shape[0], 1, 1, data.shape[1]))

model = load_model('Trained1/inputs_'+str(M)+'_del_4_snr_'+str(SNR)+'.h5')
x = keras.layers.Input(shape=(1,1,M))
encoder = Model(x, model.layers[2](model.layers[1](x)))
decoder = model.layers[3]
encoder.summary()
decoder.summary()
encoder.save_weights('encoder_weights.h5')
decoder.save_weights('decoder_weights.h5')
encoder.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
decoder.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
encoder.load_weights('encoder_weights.h5')
decoder.load_weights('decoder_weights.h5')

predicted = encoder.predict(data)
noise = np.random.normal(0, 1, predicted.size)
noisysig = np.reshape(noise, predicted.shape)+predicted
sig_hat = decoder.predict(noisysig)


SIG = (sig-32) * 5 / 32
SIG = np.zeros(991)
SIG[:] = np.nan
for n in np.arange(0, 991, 10):
    SIG[n] = sig[int(n/10)]

SIG_HAT = (sig-32) * 5 / 32
SIG_HAT = np.zeros(991)
SIG_HAT[:] = np.nan
for n in np.arange(0, 991, 10):
    SIG_HAT[n] = sig_hat[int(n/10)]

sigtoplot = pd.Series(SIG)
sigtoplot.set_axis(np.linspace(0.0, 9.9, num=991, endpoint=True), inplace=True)
sigtoplot = sigtoplot.interpolate(method='cubic')
sigtoplot.plot()
sigtoplot = pd.Series(SIG_HAT)
sigtoplot.set_axis(np.linspace(0.0, 9.9, num=991, endpoint=True), inplace=True)
sigtoplot = sigtoplot.interpolate(method='cubic')
sigtoplot.plot()
plt.title('Signal Comparison')
plt.ylabel('Signal Voltage')
plt.xlabel('Time (s)')
plt.legend(['Input', 'Output'], loc='upper left')
plt.show()
