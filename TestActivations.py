import numpy as np
#import matplotlib.pyplot as plt
import matplotlib
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import tensorflow

import keras

from keras import models
from keras.utils import plot_model
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Reshape
from keras.layers import Conv2D, Dropout, MaxPooling2D, UpSampling2D
from keras.layers import BatchNormalization
from keras.callbacks import ModelCheckpoint

from numpy import array

# definitions
M = 16
L = 5
numepochs = 16


# One hot encoding
rawdata = np.zeros(M)
i = 0
while(i<M):
    rawdata[i] = i
    i += 1

data = to_categorical(rawdata)
valdata = data

# Define Structure
input_shape = (1, M)

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

# for visualizing activations


# train the data
history = model.fit(data, data, epochs=numepochs, batch_size=8)

model.save('trained_cae.h5')


# Extracts the outputs of the top 12 layers
layer_outputs = [layer.output for layer in model.layers[:12]]
# Creates a model that will return these outputs, given the model input
activation_model = models.Model(inputs=model.input, outputs=layer_outputs)
predictdata = valdata[0, :]
activations = activation_model.predict(valdata) # Returns a list of five Numpy arrays: one array per layer activation


first_layer_activation = activations[4]
plt.matshow(first_layer_activation[0, :, :, 4], cmap='viridis')

# Visualize Activations
layer_names = []
for layer in model.layers[:4]:
    layer_names.append(layer.name) # Names of the layers, so you can have them as part of your plot
    
images_per_row = 20
for layer_name, layer_activation in zip(layer_names, activations): # Displays the feature maps
    n_features = layer_activation.shape[-1] # Number of features in the feature map
    size = layer_activation.shape[1] #The feature map has shape (1, size, size, n_features).
    n_cols = n_features // images_per_row # Tiles the activation channels in this matrix
    display_grid = np.zeros((size * n_cols, images_per_row * size))
    for col in range(n_cols): # Tiles each filter into a big horizontal grid
        for row in range(images_per_row):
            channel_image = layer_activation[0,
                                             col * images_per_row + row]
            channel_image -= channel_image.mean() # Post-processes the feature to make it visually palatable
#            channel_image /= channel_image.std()
            channel_image *= 64
            channel_image += 128
            channel_image = np.clip(channel_image, 0, 255).astype('uint8')
            display_grid[col * size : (col + 1) * size, # Displays the grid
                         row * size : (row + 1) * size] = channel_image
    scale = 1. / size
    plt.figure(figsize=(scale * display_grid.shape[1],
                        scale * display_grid.shape[0]))
    plt.title(layer_name)
    plt.grid(False)
    plt.imshow(display_grid, aspect='auto', cmap='viridis')