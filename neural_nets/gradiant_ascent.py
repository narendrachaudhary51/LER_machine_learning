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
#path = r"C:\\Users\\narendra\\Documents\\LER\\LER_machine_learning\\"

base_model = load_model(path + 'models/' + 'Linenet_round_L2_epoch_4.h5')
layer_dict = dict([(layer.name, layer) for layer in base_model.layers])

model = Model(inputs=base_model.input, \
              outputs= base_model.get_layer('conv2d_16').output)
#model.summary()

layer_dict = dict([(layer.name, layer) for layer in model.layers])

def gradient_ascent(layer_name, filter_index, step = 1, num_steps = 20):
    # build a loss function that maximizes the activation
    # of the nth filter of the layer considered
    layer_output = layer_dict[layer_name].output
    loss = K.mean(layer_output[:, :, :, filter_index])

    # compute the gradient of the input picture wrt this loss
    grads = K.gradients(loss, model.input)[0]

    # normalization trick: we normalize the gradient
    grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)

    # this function returns the loss and grads given the input picture
    iterate = K.function([model.input], [loss, grads])

    # we start from a gray image with some noise
    input_img_data = np.random.random((1, 1024, 64, 1))


    # run gradient ascent for 20 steps
    for i in range(num_steps):
        loss_value, grads_value = iterate([input_img_data])
        input_img_data += grads_value * step
    
    return input_img_data

for i in range(1,17):
    layer_name = 'conv2d_' + str(i)
    img_data = np.zeros((64,1024,64,1))

    for filter_index in range(64):
        img_data[filter_index] = gradient_ascent(layer_name, filter_index)
        print("Layer_" + layer_name + "_Filter_" + str(filter_index) + "_done")
    np.savez(path + 'Visualization/LineNet1_images/layer_' + layer_name + '_max_img_data.npz', img_data = img_data)
    
    #fig, ax = plt.subplots(nrows=8, ncols=8, figsize=(8,20), sharex=True,
    #                   sharey=True, subplot_kw={'adjustable': 'box-forced'})
    #fig.subplots_adjust(wspace= 0.05, hspace=0.05)

    #for i in range(8):
    #    for j in range(8):
    #        ax[i,j].imshow(img_data[i*8 +j,:,:,0],cmap = 'gray', aspect = 0.2)

    #plt.savefig(path + 'Visualization/LineNet1_images/layer_' + layer_name + '_max_img_data.png', dpi = 350)
    #plt.show()





