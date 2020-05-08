import tensorflow as tf
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
import numpy as np
import math
import timeit
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential,load_model, Model
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Input, Lambda, Subtract
from keras.constraints import max_norm
from PIL import Image
import keras.backend as K
#from multi_gpu import make_parallel
import time
from random import shuffle

start = time.time()


#getting the data

num_validation = 2880					# will be 2880 in full set
num_test = 8640

X_val = np.zeros((16*num_validation,64,64))
y_val = np.zeros((16*num_validation,64,64))


path = '/scratch/user/narendra5/LER_machine_learning/'
sigmas = [0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8]
alphas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]	

widths = [20, 30]
noises = [2, 3, 4, 5, 10, 20, 30, 50, 100, 200]
#noises = [2]
						

Xis = [20]
count = 0
for sigma in sigmas:
	for alpha in alphas:
		for Xi in Xis:
			for width in widths:
				for s in range(2):
					for noise in noises:
						space = math.floor(width*2**s)
						shift = math.floor(-25 + (width + space/2 + Xi + alpha*10 + sigma*10)%16) 
						
						original_file = path + 'original_images2/oim_' + "{:.2g}".format(sigma*1e-09) + '_' + str(alpha) + '_' + "{0:.2g}".format(Xi*1e-09) + '_' + str(width) + '_' + str(space) + '_' + str(-shift) + '.tiff'
						noisy_file = path + 'noisy_images2/nim_' + "{0:.2g}".format(sigma*1e-09) + '_' + str(alpha) + '_' + "{0:.2g}".format(Xi*1e-09) + '_' + str(width) + '_' + str(space) + '_' + str(-shift) + '_' + str(noise) + '.tiff'
						
						
						im = np.array(Image.open(original_file))
						imnoisy = np.array(Image.open(noisy_file))
						#imres = imnoisy - im 
						#X_val[count] = imnoisy
						#y_val[count] = imnoisy - im
						for i in range(16):
							X_val[count + i,:,:] = imnoisy[i*64:(i+1)*64]
							#X_val[count + i+1,:,:] = imnoisy[i*40:(i+1)*40,24:64] 
							y_val[count + i,:,:] = im[i*64:(i+1)*64]
							#y_val[count + i+1,:,:] = imres[i*40:(i+1)*40,24:64]
						
						count += 16
print('Validation_count: ',count)

X_val = X_val/256
y_val = y_val/256
X_val = np.reshape(X_val,(16*num_validation,64,64,1))
y_val = np.reshape(y_val,(16*num_validation,64,64,1))						

print('Validation data shape: ', X_val.shape)
print('Validation labels shape: ', y_val.shape)

	

#print('Train data shape: ', X_train.shape)
#print('Train labels shape: ', y_train.shape)

batch_size = 128
epochs = 4


#subtract_layer = Lambda(lambda inputs: inputs[0] - inputs[1],
#                        output_shape=lambda shapes: shapes[0])

def DnCNN(depth,filters=64,image_channels=1, use_bnorm=True):
    layer_count = 0
    inpt = Input(shape=(None,None,image_channels),name = 'input'+str(layer_count))
    # 1st layer, Conv+relu
    layer_count += 1
    x = Conv2D(filters=filters, kernel_size=(3,3), strides=(1,1),kernel_initializer='Orthogonal', padding='same',name = 'conv'+str(layer_count))(inpt)
    layer_count += 1
    x = Activation('relu',name = 'relu'+str(layer_count))(x)
    # depth-2 layers, Conv+BN+relu
    for i in range(depth-2):
        layer_count += 1
        x = Conv2D(filters=filters, kernel_size=(3,3), strides=(1,1),kernel_initializer='Orthogonal', padding='same',use_bias = False,name = 'conv'+str(layer_count))(x)
        if use_bnorm:
            layer_count += 1
            #x = BatchNormalization(axis=3, momentum=0.1,epsilon=0.0001, name = 'bn'+str(layer_count))(x) 
            x = BatchNormalization(axis=3, momentum=0.0,epsilon=0.0001, name = 'bn'+str(layer_count))(x)
        layer_count += 1
        x = Activation('relu',name = 'relu'+str(layer_count))(x)  
    # last layer, Conv
    layer_count += 1
    x = Conv2D(filters=image_channels, kernel_size=(3,3), strides=(1,1), kernel_initializer='Orthogonal',padding='same',use_bias = False,name = 'conv'+str(layer_count))(x)
    layer_count += 1
    x = Subtract(name = 'subtract' + str(layer_count))([inpt, x])   # input - noise
    #x = subtract_layer([inpt, x])
    model = Model(inputs=inpt, outputs=x)
    
    return model

model = DnCNN(depth=17)
model.summary()

adam = keras.optimizers.adam(lr=1e-5)

model.compile(loss = 'mean_squared_error',
              optimizer=adam)


# ----------------------------------load training data and train on it --------------------------------------- 

num_training = 9920
X_train = np.zeros((16*num_training,64,64,1))
y_train = np.zeros((16*num_training,64,64,1))

noises = [2, 3, 4, 5, 10, 20, 30, 50, 100, 200]

Xis = [6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40]
Xis.remove(10)			# remove 10, 20, 30 and 40 value (This value will be used to create validation and test set)
Xis.remove(20)
Xis.remove(30)	
Xis.remove(40)


for epoch in range(1,epochs+1):
	shuffle(alphas)
	for alpha in alphas:
		count = 0
		for sigma in sigmas:
			for noise in noises:
				for Xi in Xis:
					for width in widths:
						for s in range(2):
							space = math.floor(width*2**s)
							shift = math.floor(-25 + (width + space/2 + Xi + alpha*10 + sigma*10)%16) 
							
							original_file = path + 'original_images2/oim_' + "{:.2g}".format(sigma*1e-09) + '_' + str(alpha) + '_' + "{0:.2g}".format(Xi*1e-09) + '_' + str(width) + '_' + str(space) + '_' + str(-shift) + '.tiff'
							noisy_file = path + 'noisy_images2/nim_' + "{0:.2g}".format(sigma*1e-09) + '_' + str(alpha) + '_' + "{0:.2g}".format(Xi*1e-09) + '_' + str(width) + '_' + str(space) + '_' + str(-shift) + '_' + str(noise) + '.tiff'
							
							im = np.array(Image.open(original_file))
							imnoisy = np.array(Image.open(noisy_file))
							#imres = imnoisy - im

							im = im/256
							imnoisy = imnoisy/256
							#imres = imres/256
							#im = np.reshape(im,(1024,64,1))
							#imnoisy = np.reshape(imnoisy,(1024,64,1))
							#imres = np.reshape(imres,(1024,64,1))
							for i in range(16):
								X_train[count + i,:,:,0] = imnoisy[i*64:(i+1)*64]
								#X_train[count + i+1,:,:,:] = imnoisy[i*40:(i+1)*40,24:64]
								y_train[count + i,:,:,0] = im[i*64:(i+1)*64]
								#y_train[count + i+1,:,:,:] = imres[i*40:(i+1)*40,24:64]
							count += 16
		print("alpha set, Training count :",alpha,',',count)
		history = model.fit(X_train, y_train, batch_size=batch_size, epochs=1, shuffle=True)
	print('Running validation now for epoch ' + str(epoch))
	val_score = model.evaluate(X_val,y_val)
	print('Validation score:',val_score)
	model.save(path + 'models/' + 'DnCNN_patch64_lr5_epoch_'+ str(epoch) + '.h5')

#history = model.fit(X_train, y_train,
#              batch_size=batch_size,
#              epochs=epochs,
#              validation_data=(X_val, y_val),
#              shuffle=True)
			  

#model.save(path + 'models/' +'nnet_test_run_5.h5')
del model  # deletes the existing model


print("Execttion Time= ", time.time() - start)


#print(history.history['loss'])
#print(history.history['val_loss'])

# summarize history for accuracy
#plt.plot(history.history['acc'])
#plt.plot(history.history['val_acc'])
#plt.title('model accuracy')
#plt.ylabel('accuracy')
#plt.xlabel('epoch')
#plt.legend(['train', 'validation'], loc='upper left')
#plt.savefig('loss_plot.png')
# summarize history for loss
"""
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig(path + 'models/' + 'nnet_test_run_3.png')
"""

