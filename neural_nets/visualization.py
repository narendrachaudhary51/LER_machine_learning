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
K.set_learning_phase(0) #set learning phase

path = '/scratch/user/narendra5/LER_machine_learning/'

base_model = load_model(path + 'models/' + 'Linenet_round_L2_epoch_4.h5')
layer_dict = dict([(layer.name, layer) for layer in base_model.layers])

sigmas = [1.4, 1.6, 1.8]
Xis = [10, 30, 40]
alphas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
widths = [20, 30]
noises = [2, 3, 4, 5, 10, 20, 30, 50, 100, 200]

testsize = len(sigmas)*len(Xis)*len(alphas)*len(widths)*len(noises)*2

for sigma in sigmas:
    Gram_sigma = np.zeros((len(alphas),len(Xis),len(widths),2,len(noises),16,64,64))
    for a, alpha in enumerate(alphas):
        for x, Xi in enumerate(Xis):
            for w, width in enumerate(widths):
                for s in range(2):
                    for n, noise in enumerate(noises):
                        space = math.floor(width*2**s)
                        shift = math.floor(-25 + (width + space/2 + Xi + alpha*10 + sigma*10)%16) 
                        
                        #original_file = path + 'original_images2/oim_' + "{:.2g}".format(sigma*1e-09) + '_' + str(alpha) + '_' + "{0:.2g}".format(Xi*1e-09) + '_' + str(width) + '_' + str(space) + '_' + str(-shift) + '.tiff'
                        noisy_file = path + 'noisy_images2/nim_' + "{0:.2g}".format(sigma*1e-09) + '_' + str(alpha) + '_' + "{0:.2g}".format(Xi*1e-09) + '_' + str(width) + '_' + str(space) + '_' + str(-shift) + '_' + str(noise) + '.tiff'
                    
                        #im = np.array(Image.open(original_file))
                        imnoisy = np.array(Image.open(noisy_file))

                        #im = im/256
                        imnoisy = (imnoisy)/256
                        imnoisy = imnoisy.reshape(1,1024,64,1)
                        for i in range(1,17):
                            layer_name = 'conv2d_' + str(i)
                            model = Model(inputs=base_model.input, \
                                          outputs= base_model.get_layer(layer_name).output)
                            impredict = model.predict(imnoisy)
                            impredict = impredict.reshape(1024,64,64)
                            features = impredict.reshape(1024*64, 64)

                            Gram = np.dot(features.T, features)
                            Gram_sigma[a,x,w,s,n,i-1,:,:] = Gram
    np.savez(path + 'Gram/LineNet1_Gram_matrices_' + str(sigma)+ '.npz', Gram_sigma = Gram_sigma)
