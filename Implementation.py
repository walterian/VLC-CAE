# This file contains an attempt at actually putting the network trained in EncDec.py to practice
import keras
import numpy as np
from keras.models import Model, load_model

model = load_model('trained_cae.h5')
encoder = load_model('trained_encoder.h5')
decoder = load_model('trained_decoder.h5')

""" So variance is related to the physical property of noise power
"""

# SNR is in db (20log10(S/N))
SNR = 20

Beta = np.sqrt(1/(2*10**(SNR/20)))

print(Beta)