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

model_1 = load_model(path + 'models/' + 'DnCNN2_full_epoch_2.h5')
model_2 = load_model(path + 'models/' + 'DnCNN_patch64_epoch_3.h5')
model_3 = load_model(path + 'models/' + 'SEMNet_run2_epoch_4.h5')
np.random.seed(21)
sigma = 1.2
alpha = 0.7
Xi = 40

width = 30
space = 30
noise = 2
shift = math.floor(-25 + (width + space/2 + Xi + alpha*10 + sigma*10)%16) 


original_file = path + 'original_images2/oim_' + "{:.2g}".format(sigma*1e-09) + '_' + str(alpha) + '_' + "{0:.2g}".format(Xi*1e-09) + '_' + str(width) + '_' + str(space) + '_' + str(-shift) + '.tiff'
noisy_file = path + 'noisy_images2/nim_' + "{0:.2g}".format(sigma*1e-09) + '_' + str(alpha) + '_' + "{0:.2g}".format(Xi*1e-09) + '_' + str(width) + '_' + str(space) + '_' + str(-shift) + '_' + str(noise) + '.tiff'

#noisy_file = path + 'Example_images/nim_' + "{0:.2g}".format(sigma*1e-09) + '_' + str(alpha) + '_' + "{0:.2g}".format(Xi*1e-09) + '_' + str(width) + '_' + str(space) + '_' + str(-shift) + '_' + str(noise) + '.tiff'


im = np.array(Image.open(original_file))
imnoisy = np.array(Image.open(noisy_file))


im = im/256
imnoisy = (imnoisy)/256
imnoisy = imnoisy + np.random.normal(0, 0.1 * (256/255), imnoisy.shape)


imnoisy = imnoisy.reshape(1,1024,64,1)
#impredict = np.zeros((1,1024,64,1))
impredict_1 = model_1.predict(imnoisy)
impredict_2 = model_2.predict(imnoisy)
impredict_3 = model_3.predict(imnoisy)
imnoisy = imnoisy.reshape(1024,64)

impredict_1 = impredict_1.reshape(1024,64)
impredict_1 = impredict_1.astype(float)
impredict_2 = impredict_2.reshape(1024,64)
impredict_2 = impredict_2.astype(float)
impredict_3 = impredict_3.reshape(1024,64)
impredict_3 = impredict_3.astype(float)


mse_1 = ((im - impredict_1) ** 2).mean()
mse_2 = ((im - impredict_2) ** 2).mean()
mse_3 = ((im - impredict_3) ** 2).mean()
print(mse_1, mse_2, mse_3)

psnr_predict_1 = measure.compare_psnr(im,impredict_1)
psnr_predict_2 = measure.compare_psnr(im,impredict_2)
psnr_predict_3 = measure.compare_psnr(im,impredict_3)
print('DnCNN_fullimage PSNR = ',psnr_predict_1)
print('DnCNN_patch64 PSNR = ',psnr_predict_2)
print('PSNR = ',psnr_predict_3)


np.savez('DnCNN_fullimage_gauss01_prediction_' + "{0:.2g}".format(sigma*1e-09) + '_' + str(alpha) + '_' + "{0:.2g}".format(Xi*1e-09) + '_' + str(width) + '_' + str(space) + '_' + str(-shift) + '_' + str(noise) +'.npz', impredict = impredict_1)

np.savez('DnCNN_patch64_gauss01_prediction_' + "{0:.2g}".format(sigma*1e-09) + '_' + str(alpha) + '_' + "{0:.2g}".format(Xi*1e-09) + '_' + str(width) + '_' + str(space) + '_' + str(-shift) + '_' + str(noise) +'.npz', impredict = impredict_2)
