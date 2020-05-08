import tensorflow as tf
import numpy as np
import math
import timeit
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential,load_model, Model
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Input
from keras.constraints import max_norm
from keras import backend as K
from PIL import Image
import skimage


K.set_learning_phase(0) #set learning phase
path = '/scratch/user/narendra5/LER_machine_learning/'


base_model = load_model(path + 'models/' + 'Linenet_round_L2_epoch_4.h5')
layer_dict = dict([(layer.name, layer) for layer in base_model.layers])

activations = []

sigma = 1.6
alpha = 0.5
Xi = 30
width = 30
space = 30
noise = 5
shift = math.floor(-25 + (width + space/2 + Xi + alpha*10 + sigma*10)%16) 

#path = r"C:\\Users\\narendra\\Documents\\LER\\"
original_file = path + 'original_images2/oim_' + "{:.2g}".format(sigma*1e-09) + '_' + str(alpha) + '_' + "{0:.2g}".format(Xi*1e-09) + '_' + str(width) + '_' + str(space) + '_' + str(-shift) + '.tiff'
noisy_file = path + 'noisy_images2/nim_' + "{0:.2g}".format(sigma*1e-09) + '_' + str(alpha) + '_' + "{0:.2g}".format(Xi*1e-09) + '_' + str(width) + '_' + str(space) + '_' + str(-shift) + '_' + str(noise) + '.tiff'


im = np.array(Image.open(original_file))
imnoisy = np.array(Image.open(noisy_file))


for i in range(1,17):
    layer_name = 'conv2d_' + str(i)
    model = Model(inputs=base_model.input, \
              outputs= base_model.get_layer(layer_name).output)

    im = im/256
    imnoisy = (imnoisy)/256
    imnoisy = imnoisy.reshape(1,1024,64,1)
    impredict = model.predict(imnoisy)
    impredict = impredict.reshape(1024,64,64)

    for filter_index in range(64):
        activations.append(impredict[:,:,filter_index].mean())

np.savez(path + 'mean_activation_plots/activation_' + "{0:.2g}".format(sigma*1e-09) + '_' + str(alpha) + '_' + "{0:.2g}".format(Xi*1e-09) + '_' + str(width) + '_' + str(space) + '_' + str(-shift) + '_' + str(noise) + '.npz', activations = np.array(activations))
#plt.figure(figsize = (16,8))
#plt.plot(activations,'-o')
#plt.ylim(0,100)
#plt.savefig(path + 'mean_activation_plots/activation_' + "{0:.2g}".format(sigma*1e-09) + '_' + str(alpha) + '_' + "{0:.2g}".format(Xi*1e-09) + '_' + str(width) + '_' + str(space) + '_' + str(-shift) + '_' + str(noise) + '.png', dpi = 350)
