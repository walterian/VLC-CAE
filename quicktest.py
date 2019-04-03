import keras
from keras.utils import plot_model
from keras.models import load_model
from keras.utils import to_categorical

from vis.visualization import visualize_saliency

# load trained model for visualization
model = load_model('trained_cae.h5')

model.summary()
visualize_saliency()