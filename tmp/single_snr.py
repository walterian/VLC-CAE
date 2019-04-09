# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 10:34:21 2019

@author: SINE_Lab
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 10:25:14 2019

@author: SINE_Lab
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 09:56:32 2018

@author: SINE_Lab
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Nov 11 12:56:47 2018

@author: pachp
"""

# importing libs
import numpy as np
import tensorflow as tf
import keras
from keras.layers import Input, Dense, GaussianNoise,Lambda,Dropout
from keras.models import Model
from keras import regularizers
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam,SGD
from keras.layers import Dropout
from keras import backend as K
import scipy.io as spio
import pandas as pd
import matplotlib.pyplot as plt
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Reshape
from keras import regularizers
import random
# In[ ]:


# for reproducing reslut
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(3)


# In[ ]:


# defining parameters
# define (n,k) here for (n,k) autoencoder
# n = n_channel 
# k = log2(M)  ==> so for (7,4) autoencoder n_channel = 7 and M = 2^4 = 16 
M = 4
k = np.log2(M)
k = int(k)
n_channel = 2
R = k/n_channel
print ('M:',M,'k:',k,'n:',n_channel)



# In[ ]:


#generating data of size N
N = 7000
label = np.random.randint(M,size=N)


# In[ ]:


# creating one hot encoded vectors
data = []
for i in label:
    temp = np.zeros(M)
    temp[i] = 1
    data.append(temp)


# In[ ]:


# checking data shape
data = np.array(data)
print (data.shape)




# checking generated data with it's label
temp_check = [17,23,45,67,89,96,72,250,350]
for i in temp_check:
    print(label[i],data[i])




# defining autoencoder and it's layer
input_signal = Input(shape=(M,))
#input_signal = Flatten(input_shape=(28))
#input_signal = Dropout(0.2, input_shape=(M,)))

encoded = Dense(M, activation='relu')(input_signal)
encoded1 = Dense(n_channel, activation='linear')(encoded)
encoded2 =Lambda(lambda x: np.sqrt(n_channel)*K.l2_normalize(x,axis=1))(encoded1)

encoded3 = Lambda(lambda x: tf.to_float(tf.spectral.ifft(tf.cast(x, dtype=tf.complex64))))(encoded2)


EbNo_train =12 
e_channel = GaussianNoise(np.sqrt(1/(2*R*EbNo_train)))(encoded3)




decoded1 = Dense(M, activation='relu')(e_channel)
decoded2 = Dense(M, activation='softmax')(decoded1)

autoencoder = Model(input_signal, decoded2)

adam = Adam(lr=0.001)
hist = autoencoder.compile(optimizer=adam, loss='categorical_crossentropy')

# In[ ]:


# printing summary of layers and it's trainable parameters 
print (autoencoder.summary())



# traning auto encoder
history = autoencoder.fit(data, data,
                epochs=45,
                batch_size=32)

                           


# In[ ]:


# saving keras model
#from keras.models import load_model
# if you want to save model then remove below comment
# autoencoder.save('autoencoder_v_best.model')



# In[ ]:


# making encoder from full autoencoder
encoder = Model(input_signal, encoded3)


# In[ ]: retrieve the last layer of the autoencoder model
#decoder_layer = autoencoder.layers[-1]
## create the decoder model
#decoder = Model(encoded_input, decoder_layer(encoded_input))

# create a placeholder for an encoded  input)
encoded_input = Input(shape=(n_channel,))
###Decoder

deco1 = autoencoder.layers[-2](encoded_input)
deco2 = autoencoder.layers[-1](deco1)
#decoder_layer = autoencoder.layers[-1](encoded_input)
decoder = Model(encoded_input, deco2)


# In[ ]:


# generating data for checking BER
# if you're not using t-sne for visulation than set N to 70,000 for better result 
# for t-sne use less N like N = 1500
N = 3000
test_label = np.random.randint(M,size=N)
test_data = []

for i in test_label:
    temp = np.zeros(M)
    temp[i] = 1
    test_data.append(temp)
    
test_data = np.array(test_data)
print('test data', test_data)

# In[ ]:


# checking generated data
temp_test = 6
print (test_data[temp_test][test_label[temp_test]],test_label[temp_test])


# In[ ]:


 #for plotting learned consteallation diagram



eye_matrix = np.eye(M)
encoded_planisphere = encoder.predict(eye_matrix) 
plt.title('Constellation')
plt.xlim(-2, 2)
plt.ylim(-2, 2)
plt.plot(encoded_planisphere[:,0], encoded_planisphere[:,1], 'r.')
plt.grid(True)
plt.show()



# In[ ]:

#
## ploting constellation diagram
import matplotlib.pyplot as plt
#scatter_plot = scatter_plot.reshape(M,2,1)
#plt.scatter(scatter_plot[:,0],scatter_plot[:,1])
#plt.axis((-2.5,2.5,-2.5,2.5))
#plt.grid()
#plt.show()
#

# In[ ]:


def frange(x, y, jump):
  while x < y:
    yield x
    x += jump


# In[ ]:


# calculating BER
# this is optimized BER function so it can handle large number of N
# previous code has another for loop which was making it slow
#EbNodB_range = list(frange(20,30,0.5))
EbNodB_range = list(frange(0,40,1))


ber = [None]*len(EbNodB_range)
for n in range(0,len(EbNodB_range)):
    EbNo=10.0**(EbNodB_range[n]/10.0)
 
    noise_std = np.sqrt(1/(2*R*EbNo))####Set a value
    noise_mean = 0
    no_errors = 0
    nn = N
    subcarrier = 16
    noise = noise_std * np.random.randn(nn,n_channel)

    encoded_signal = encoder.predict(test_data) 

    
    final_signal = encoded_signal + noise
   

    pred_final_signal =  decoder.predict(final_signal)
    pred_output = np.argmax(
            pred_final_signal,axis=1)
    no_errors = (pred_output != test_label)
    no_errors =  no_errors.astype(int).sum()
    ber[n] = no_errors / nn 
    #print ('S')
    print ('SNR:',EbNodB_range[n],'BER:',ber[n])
    #print (ber[n])
#    # use below line for generating matlab like matrix which can be copy and paste for plotting ber graph in matlab
   # print(ber[n], " ",end='')
ber = np.array(ber)



# ploting ber curve
import matplotlib.pyplot as plt
from scipy import interpolate
plt.plot(EbNodB_range, ber, 'bo',label='Autoencoder(2,6)')
plt.yscale('log')
plt.xlabel('SNR Range')
plt.ylabel('Block Error Rate')
plt.grid()
plt.legend(loc='upper right',ncol = 1)
plt.title('BER calculated using Autoencoder')
