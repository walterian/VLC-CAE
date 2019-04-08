# This file contains an attempt at actually putting the network trained in EncDec.py to practice
import keras
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Model, load_model
import pandas as pd
from matplotlib.widgets import Slider

def decode(onehot):
    return np.argmax(onehot)

SNR = 200
M = 64
delta = 4
C = 1
graph_pretty = True
# Generate random signal of length 32 with 64 possible values
siglength = 10
sig = np.random.randint(0, 64, siglength)

data = np.array(sig)
data = keras.utils.to_categorical(data, num_classes=64)
data = data.astype(int)
data = np.reshape(data, (data.shape[0], 1, 1, data.shape[1]))
# Load model and compile encoder and decoder portions
model = load_model('Trained/inputs_'+str(M)+'_del_'+str(delta)+'_snr_'+str(SNR)+'.h5')
x = keras.layers.Input(shape=(1,1,M))
encoder = Model(x, model.layers[2](model.layers[1](x)))
decoder = model.layers[3]
encoder.save_weights('encoder_weights.h5')
decoder.save_weights('decoder_weights.h5')
encoder.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
decoder.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
encoder.load_weights('encoder_weights.h5')
decoder.load_weights('decoder_weights.h5')
# Pass input through network and decode output (implement a LUT)
predicted = encoder.predict(data)
noise = np.random.normal(0, np.sqrt(C/(10**(SNR/10))), predicted.size)
noisysig = np.reshape(noise, predicted.shape)+predicted
encoded = decoder.predict(noisysig)
sig_hat = np.zeros(siglength)
for i in range(encoded.shape[0]):
    sig_hat[i] = decode(encoded[i])

# Check what kind of plot we want
if graph_pretty == False:
    SIG = sig
    SIG_HAT = sig_hat
    numpoints = siglength
else:
    SIG = np.zeros(siglength*10-9)
    SIG[:] = np.nan
    for n in np.arange(0, siglength*10-9, 10):
        SIG[n] = (sig[int(n/10)]-32) * 5 / 32
    SIG_HAT = np.zeros(siglength*10-9)
    SIG_HAT[:] = np.nan
    for n in np.arange(0, siglength*10-9, 10):
        SIG_HAT[n] = (sig_hat[int(n/10)]-32) * 5 / 32
    numpoints = siglength*10-9
# Plot both signals
sigtoplot = pd.Series(SIG)
sigtoplot.set_axis(np.linspace(0.0, 9.9, num=numpoints, endpoint=True), inplace=True)
sigtoplot = sigtoplot.interpolate(method='cubic')
sigtoplot.plot(linewidth=3)
sigtoplot = pd.Series(SIG_HAT)
sigtoplot.set_axis(np.linspace(0.0, 9.9, num=numpoints, endpoint=True), inplace=True)
sigtoplot = sigtoplot.interpolate(method='cubic')
sigtoplot.plot(linestyle='--')
plt.title('Signal Comparison')
plt.ylabel('Signal Voltage')
plt.xlabel('Time (s)')
plt.legend(['Input', 'Output'], loc='upper left')
plt.show()
