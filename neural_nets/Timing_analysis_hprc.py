import tensorflow as tf
import numpy as np
import math
import timeit, time
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential,load_model
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from PIL import Image
import skimage

from skimage import measure

#import os
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
#os.environ["CUDA_VISIBLE_DEVICES"] = ""

path = '/scratch/user/narendra5/'

LineNet1_model = load_model(path + 'LER_machine_learning/models/' + 'Linenet_round_L2_epoch_4.h5')
LineNet1_6layer_model = load_model(path + 'LER_machine_learning/models/' + 'Linenet_round_L2_6layer_epoch_4.h5')
LineNet1_11layer_model = load_model(path + 'LER_machine_learning/models/' + 'Linenet_round_L2_11layer_epoch_4.h5')
#SEMNet_model = load_model(path + 'LER_machine_learning/models/' + 'SEMNet_run2_epoch_4.h5')
#EDGENet_model = load_model(path + 'LER_machine_learning/models/' + 'EDGEnet2_round_L1_epoch_4.h5')
#LineNet2_model = load_model(path + 'LER_machine_learning/models/' + 'Linenet_image3_round_L2_epoch_2.h5')


sigma = 0.8
alpha = 0.3
Xi = 10
width = 20
space = 20
noise = 10
shift = math.floor(-25 + (width + space/2 + Xi + alpha*10 + sigma*10)%16) 

original_file = path + 'LER_machine_learning/original_images2/oim_' + "{:.2g}".format(sigma*1e-09) + '_' + str(alpha) + '_' + "{0:.2g}".format(Xi*1e-09) + '_' + str(width) + '_' + str(space) + '_' + str(-shift) + '.tiff'
noisy_file = path + 'LER_machine_learning/noisy_images2/nim_' + "{0:.2g}".format(sigma*1e-09) + '_' + str(alpha) + '_' + "{0:.2g}".format(Xi*1e-09) + '_' + str(width) + '_' + str(space) + '_' + str(-shift) + '_' + str(noise) + '.tiff'
linescan_file = path + 'LER_machine_learning/linescans/linescan_' + "{:.2g}".format(sigma*1e-09) + '_' + str(alpha) + '_' + "{0:.2g}".format(Xi*1e-09) + '_' + str(width) + '_' + str(space) + '.txt'

im = np.array(Image.open(original_file))
im = im/256

imnoisy = np.array(Image.open(noisy_file))
imnoisy = (imnoisy)/256
imnoisy = imnoisy.reshape(1,1024,64,1)


SETUP_CODE = """

from __main__ import imnoisy, LineNet1_model, LineNet1_6layer_model, LineNet1_11layer_model
"""
LineNet1_TEST_CODE = """
LineNet1_model.predict(imnoisy)
"""

LineNet1_6layer_TEST_CODE = """
LineNet1_6layer_model.predict(imnoisy)
"""

LineNet1_11layer_TEST_CODE = """
LineNet1_11layer_model.predict(imnoisy)
"""


#LineNet2_TEST_CODE = """
#LineNet2_model.predict(imnoisy)
#"""

#SEMNet_TEST_CODE = """
#SEMNet_model.predict(imnoisy)
#"""
#EDGENet_TEST_CODE = """
#EDGENet_model.predict(imnoisy)
#"""

N = 10
LineNet1_times = timeit.repeat(setup = SETUP_CODE, 
                          stmt = LineNet1_TEST_CODE,
			  repeat=3,  
                          number = N) 
print('LineNet1 Run time: {}'.format(np.asarray(LineNet1_times)/N)) 


LineNet1_6layer_times = timeit.repeat(setup = SETUP_CODE,
                          stmt = LineNet1_6layer_TEST_CODE,
                          repeat=3,
                          number = N)
print('LineNet1_6layer Run time: {}'.format(np.asarray(LineNet1_6layer_times)/N))


LineNet1_11layer_times = timeit.repeat(setup = SETUP_CODE,
                          stmt = LineNet1_11layer_TEST_CODE,
                          repeat=3,
                          number = N)
print('LineNet1_11layer Run time: {}'.format(np.asarray(LineNet1_11layer_times)/N))




#SEMNet_times = timeit.repeat(setup = SETUP_CODE,
#                          stmt = SEMNet_TEST_CODE,
#			  repeat=3,
#                          number = N)
#print('SEMNet Run time: {}'.format(np.asarray(SEMNet_times)/N))

#EDGENet_times = timeit.repeat(setup = SETUP_CODE,
#                          stmt = EDGENet_TEST_CODE,
#			  repeat=3,
#                          number = N)
#print('EDGENet Run time: {}'.format(np.asarray(EDGENet_times)/N))

#imnoisy = np.random.rand(1,256,256,1)
#LineNet2_times = timeit.repeat(setup = SETUP_CODE,
#                          stmt = LineNet2_TEST_CODE,
#			  repeat=3,
#                          number = N)
#print('LineNet2 Run time: {}'.format(np.asarray(LineNet2_times)/N))
