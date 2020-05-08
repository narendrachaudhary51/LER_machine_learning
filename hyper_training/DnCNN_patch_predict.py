import tensorflow as tf
import numpy as np
import math
import timeit, time
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential,load_model, Model
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Input, Lambda
from keras.constraints import max_norm

from PIL import Image
import matplotlib.pyplot as plt
from skimage import measure

path = '/scratch/user/narendra5/LER_machine_learning/'

model = load_model(path + 'models/' + 'DnCNN_patch_epoch_4.h5')

np.random.seed(21)
sigma = 1.6
alpha = 0.5
Xi = 30

width = 20
space = 40
noise = 100
shift = math.floor(-25 + (width + space/2 + Xi + alpha*10 + sigma*10)%16) 


original_file = path + 'original_images2/oim_' + "{:.2g}".format(sigma*1e-09) + '_' + str(alpha) + '_' + "{0:.2g}".format(Xi*1e-09) + '_' + str(width) + '_' + str(space) + '_' + str(-shift) + '.tiff'
noisy_file = path + 'noisy_images2/nim_' + "{0:.2g}".format(sigma*1e-09) + '_' + str(alpha) + '_' + "{0:.2g}".format(Xi*1e-09) + '_' + str(width) + '_' + str(space) + '_' + str(-shift) + '_' + str(noise) + '.tiff'


im = np.array(Image.open(original_file))
imnoisy = np.array(Image.open(noisy_file))


im = im/256
imnoisy = (imnoisy)/256
#imnoisy = imnoisy + np.random.normal(0, 0.05 * (256/255), imnoisy.shape)


imnoisy = imnoisy.reshape(1,1024,64,1)
#impredict = np.zeros((1,1024,64,1))
impredict = model.predict(imnoisy)

imnoisy = imnoisy.reshape(1024,64)
impredict = impredict.reshape(1024,64)
impredict = impredict.astype(float)
psnr_predict = measure.compare_psnr(im[0:40,0:40],(imnoisy - impredict)[0:40,0
:40])

print('PSNR = ',psnr_predict)
#np.savez('prediction_' + "{0:.2g}".format(sigma*1e-09) + '_' + str(alpha) + '_' + "{0:.2g}".format(Xi*1e-09) + '_' + str(width) + '_' + str(space) + '_' + str(-shift) + '_' + str(noise) + '.npz', impredict = impredict)

#plt.imshow(impredict,cmap = 'gray', aspect= 0.2)
#plt.imsave(impredict, 'DnCNN_full_prediction.png', cmap = 'gray')

