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
from keras.models import Sequential,load_model
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from keras.constraints import max_norm
from PIL import Image
import keras.backend as K
from multi_gpu import make_parallel
import time

start = time.time()


#getting the data

num_validation = 288					# will be 2880 in full set
num_test = 8640

X_val = np.zeros((num_validation,1024,64))
y_val = np.zeros((num_validation,1024,64))


path = '/scratch/user/narendra5/LER_machine_learning/'
sigmas = [0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8]
alphas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]	

widths = [20, 30]
#noises = [2, 3, 4, 5, 10, 20, 30, 50, 100, 200]
noises = [2]
						

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
						
						original_file = path + 'original_images/oim_' + "{:.2g}".format(sigma*1e-09) + '_' + str(alpha) + '_' + "{0:.2g}".format(Xi*1e-09) + '_' + str(width) + '_' + str(space) + '_' + str(-shift) + '.tiff'
						noisy_file = path + 'noisy_images/nim_' + "{0:.2g}".format(sigma*1e-09) + '_' + str(alpha) + '_' + "{0:.2g}".format(Xi*1e-09) + '_' + str(width) + '_' + str(space) + '_' + str(-shift) + '_' + str(noise) + '.tiff'
						
						
						im = np.array(Image.open(original_file))
						imnoisy = np.array(Image.open(noisy_file))

						X_val[count] = imnoisy
						y_val[count] = im
						count += 1
print('Validation_count: ',count)

X_val = X_val/256
y_val = y_val/256
X_val = np.reshape(X_val,(num_validation,1024,64,1))
y_val = np.reshape(y_val,(num_validation,1024,64,1))						

print('Validation data shape: ', X_val.shape)
print('Validation labels shape: ', y_val.shape)

	

#print('Train data shape: ', X_train.shape)
#print('Train labels shape: ', y_train.shape)

batch_size = 8
epochs = 1

model = Sequential()
model.add(Conv2D(64, (3, 3), padding='same',
                 input_shape= (1024,64,1), activation = 'relu'))
model.add(BatchNormalization(axis=3))
model.add(Dropout(0.2))

model.add(Conv2D(64, (3, 3), padding='same',activation = 'relu'))
model.add(BatchNormalization(axis=3))
model.add(Dropout(0.2))

model.add(Conv2D(64, (3, 3), padding='same', activation = 'relu'))
model.add(BatchNormalization(axis=3))
model.add(Dropout(0.2))

model.add(Conv2D(64, (3, 3), padding='same', activation = 'relu'))
model.add(BatchNormalization(axis=3))
model.add(Dropout(0.2))

model.add(Conv2D(64, (3, 3), padding='same', activation = 'relu'))
model.add(BatchNormalization(axis=3))
model.add(Dropout(0.2))

model.add(Conv2D(64, (3, 3), padding='same', activation = 'relu'))
model.add(BatchNormalization(axis=3))
model.add(Dropout(0.2))

model.add(Conv2D(64, (3, 3), padding='same', activation = 'relu'))
model.add(BatchNormalization(axis=3))
model.add(Dropout(0.2))

model.add(Conv2D(64, (3, 3), padding='same', activation = 'relu'))
model.add(BatchNormalization(axis=3))
model.add(Dropout(0.2))

model.add(Conv2D(64, (3, 3), padding='same', activation = 'relu'))
model.add(BatchNormalization(axis=3))
model.add(Dropout(0.2))

model.add(Conv2D(64, (3, 3), padding='same', activation = 'relu'))
model.add(BatchNormalization(axis=3))
model.add(Dropout(0.2))

model.add(Conv2D(64, (3, 3), padding='same', activation = 'relu'))
model.add(BatchNormalization(axis=3))
model.add(Dropout(0.2))

model.add(Conv2D(64, (3, 3), padding='same', activation = 'relu'))
model.add(BatchNormalization(axis=3))
model.add(Dropout(0.2))

model.add(Conv2D(64, (3, 3), padding='same', activation = 'relu'))
model.add(BatchNormalization(axis=3))
model.add(Dropout(0.2))

model.add(Conv2D(64, (3, 3), padding='same', activation = 'relu'))
model.add(BatchNormalization(axis=3))
model.add(Dropout(0.2))

model.add(Conv2D(64, (3, 3), padding='same', activation = 'relu'))
model.add(BatchNormalization(axis=3))
model.add(Dropout(0.2))

model.add(Conv2D(64, (3, 3), padding='same', activation = 'relu'))
model.add(BatchNormalization(axis=3))
model.add(Dropout(0.2))


#model.add(Dropout(0.5))

model.add(Conv2D(1, (3, 3), padding='same'))


model.summary()

G = 1
if G > 1:
   model = make_parallel(model,G)


adam = keras.optimizers.adam(lr=1e-3)

model.compile(loss = 'mean_squared_error',
              optimizer=adam)


# ----------------------------------load training data and train on it --------------------------------------- 

num_training = 8928
X_train = np.zeros((num_training,1024,64,1))
y_train = np.zeros((num_training,1024,64,1))

noises = [2, 3, 4, 5, 10, 20, 30, 50, 100, 200]

Xis = [6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40]
Xis.remove(10)			# remove 10, 20, 30 and 40 value (This value will be used to create validation and test set)
Xis.remove(20)
Xis.remove(30)	
Xis.remove(40)

for noise in noises:
	count = 0
	#X_train = np.reshape(X_train,(num_training,1024,64))
	#y_train = np.reshape(y_train,(num_training,1024,64))
	for sigma in sigmas:
		for alpha in alphas:
			for Xi in Xis:
				for width in widths:
					for s in range(2):
						space = math.floor(width*2**s)
						shift = math.floor(-25 + (width + space/2 + Xi + alpha*10 + sigma*10)%16) 
						
						original_file = path + 'original_images/oim_' + "{:.2g}".format(sigma*1e-09) + '_' + str(alpha) + '_' + "{0:.2g}".format(Xi*1e-09) + '_' + str(width) + '_' + str(space) + '_' + str(-shift) + '.tiff'
						noisy_file = path + 'noisy_images/nim_' + "{0:.2g}".format(sigma*1e-09) + '_' + str(alpha) + '_' + "{0:.2g}".format(Xi*1e-09) + '_' + str(width) + '_' + str(space) + '_' + str(-shift) + '_' + str(noise) + '.tiff'
						
						im = np.array(Image.open(original_file))
						imnoisy = np.array(Image.open(noisy_file))

						im = im/256
						imnoisy = imnoisy/256
						im = np.reshape(im,(1024,64,1))
						imnoisy = np.reshape(imnoisy,(1024,64,1))
						X_train[count] = imnoisy
						y_train[count] = im
						count += 1
	
	print("noise set, Training count :",noise, count)
	#X_train = X_train/256
	#y_train = y_train/256
	#X_train = np.reshape(X_train,(num_training,1024,64,1))
	#y_train = np.reshape(y_train,(num_training,1024,64,1))
	history = model.fit(X_train, y_train, batch_size=batch_size, epochs=1, validation_data=(X_val, y_val), shuffle=True)


#history = model.fit(X_train, y_train,
#              batch_size=batch_size,
#              epochs=epochs,
#              validation_data=(X_val, y_val),
#              shuffle=True)
			  

model.save(path + 'models/' +'nnet_test_run_5.h5')
del model  # deletes the existing model


print("Execttion Time= ", time.time() - start)


print(history.history['loss'])
print(history.history['val_loss'])

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

