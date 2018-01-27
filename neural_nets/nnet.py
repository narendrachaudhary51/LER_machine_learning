#import tensorflow as tf
#from tensorflow.python.client import device_lib
#print(device_lib.list_local_devices())
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
import os.path

#getting the data
num_training = 10000
num_validation = 80

X_train = np.zeros((10080,1024,64))
y_train = np.zeros((10080,1024,64))

path = '/home/grads/n/narendra5/Desktop/Programs/LER_machine_learning/'
sigmas = [0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8]
alphas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
Xis = range(6,41)
widths = [20, 30]
#noises = [2, 3, 4, 5, 10, 20, 30, 50, 100, 200]
noises = [2]
count = 0

for sigma in sigmas:
	for alpha in alphas:
		for Xi in Xis:
			for width in widths:
				for s in range(2):
					for noise in noises:
						space = math.floor(width*2**s)
						shift = math.floor(-25 + (width + space/2 + Xi + alpha*10 + sigma*10)%16) 
						#print(shift,count)
						original_file = path + 'original_images/oim_' + "{:.2g}".format(sigma*1e-09) + '_' + str(alpha) + '_' + "{0:.2g}".format(Xi*1e-09) + '_' + str(width) + '_' + str(space) + '_' + str(-shift) + '.tiff'
						noisy_file = path + 'noisy_images/nim_' + "{0:.2g}".format(sigma*1e-09) + '_' + str(alpha) + '_' + "{0:.2g}".format(Xi*1e-09) + '_' + str(width) + '_' + str(space) + '_' + str(-shift) + '_' + str(noise) + '.tiff'
						
						#print(original_file)
						#if os.path.isfile(original_file):
						im = np.array(Image.open(original_file))
						imnoisy = np.array(Image.open(noisy_file))

						X_train[count] = imnoisy
						y_train[count] = im
						count += 1
						
print(count)

"""
for i in range(1,51):
	for j in range(1,51):
		imnoisy = np.array(Image.open('/scratch/user/narendra5/noisy_lines/nline_' +str(i) +'_'+str(j)+'.tiff'))
		im = np.array(Image.open('/scratch/user/narendra5/original_lines/line'+str(i)+'.tiff'))
		X_train[(i-1)*50+j-1] = imnoisy
		y_train[(i-1)*50+j-1] = im

"""

X_train = np.reshape(X_train,(10080,1024,64,1))
y_train = np.reshape(y_train,(10080,1024,64,1))

X_train = X_train/256
y_train = y_train/256

#print(X_train[0])

X_train = X_train.astype(np.float32)
y_train = y_train.astype(np.float32)
mask = range(num_training, num_training + num_validation)
X_val = X_train[mask]
y_val = y_train[mask]
mask = range(num_training)
X_train = X_train[mask]
y_train = y_train[mask]

assert not np.any(np.isnan(X_train))
assert not np.any(np.isnan(y_train))

print (X_train.dtype)
#mean_image = np.mean(X_train, axis=0)
#std_image = np.std(X_train, axis=0)


#X_train -= mean_image
#X_val -= mean_image

#X_train /= std_image
#X_val   /= std_image


print('Train data shape: ', X_train.shape)
print('Train labels shape: ', y_train.shape)
print('Validation data shape: ', X_val.shape)
print('Validation labels shape: ', y_val.shape)

batch_size = 1
epochs = 1

model = Sequential()
model.add(Conv2D(64, (3, 3), padding='same',
                 input_shape=X_train.shape[1:], activation = 'relu'))
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


adam = keras.optimizers.adam(lr=1e-3)

model.compile(loss = 'mean_squared_error',
              optimizer=adam)

history = model.fit(X_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(X_val, y_val),
              shuffle=True)
			  
model.save('nnet_test_run_1.h5')  # creates a HDF5 file 'my_model.h5'
del model  # deletes the existing model

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
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
#plt.savefig('dropout_loss_17_plot.png')
plt.show()

