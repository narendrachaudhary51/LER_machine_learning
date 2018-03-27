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
from random import shuffle

start = time.time()


#getting the data

num_validation = 2880					# will be 2880 in full set
num_test = 8640

X_val = np.zeros((num_validation,1024,64))
y_val = np.zeros((num_validation,1024,2))


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
						linescan_file = path + 'linescans/linescan_' + "{:.2g}".format(sigma*1e-09) + '_' + str(alpha) + '_' + "{0:.2g}".format(Xi*1e-09) + '_' + str(width) + '_' + str(space) + '.txt'
						
						noisy_file = path + 'noisy_images/nim_' + "{0:.2g}".format(sigma*1e-09) + '_' + str(alpha) + '_' + "{0:.2g}".format(Xi*1e-09) + '_' + str(width) + '_' + str(space) + '_' + str(-shift) + '_' + str(noise) + '.tiff'
						linescan = []
						with open(linescan_file,'r') as f:
							for i,line in enumerate(f):
								if i < 3000:
									a, b = line.split(',')
									linescan.append(float(b))
								else:
									break
						linescan = linescan[:2048]
						leftline = np.array(linescan[:1024])
						rightline = linescan[1024:]
						rightline.reverse()
						rightline = np.array(rightline)

						imnoisy = np.array(Image.open(noisy_file))
						leftline = leftline + shift
						rightline = rightline + shift
						
						X_val[count] = imnoisy
						y_val[count,:,0] = leftline.astype(int)/64
						y_val[count,:,1] = rightline.astype(int)/64
						count += 1
print('Validation_count: ',count)

X_val = X_val/256

X_val = np.reshape(X_val,(num_validation,1024,64,1))
y_val = np.reshape(y_val,(num_validation,1024,2,1))						

print('Validation data shape: ', X_val.shape)
print('Validation labels shape: ', y_val.shape)

	

batch_size = 32
epochs = 4

model = Sequential()
model.add(Conv2D(64, (3, 3), padding='same',input_shape= (1024,64,1), activation = 'relu'))
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

model.add(Conv2D(128, (3, 3), padding='same', strides=(1, 2),activation = 'relu'))
model.add(BatchNormalization(axis=3))
model.add(Dropout(0.2))

model.add(Conv2D(128, (3, 3), padding='same', activation = 'relu'))
model.add(BatchNormalization(axis=3))
model.add(Dropout(0.2))

model.add(Conv2D(128, (3, 3), padding='same', activation = 'relu'))
model.add(BatchNormalization(axis=3))
model.add(Dropout(0.2))

model.add(Conv2D(128, (3, 3), padding='same', activation = 'relu'))
model.add(BatchNormalization(axis=3))
model.add(Dropout(0.2))

model.add(Conv2D(256, (3, 3), padding='same', strides=(1, 2), activation = 'relu'))
model.add(BatchNormalization(axis=3))
model.add(Dropout(0.2))

model.add(Conv2D(256, (3, 3), padding='same', activation = 'relu'))
model.add(BatchNormalization(axis=3))
model.add(Dropout(0.2))

model.add(Conv2D(256, (3, 3), padding='same', strides=(1, 2), activation = 'relu'))
model.add(BatchNormalization(axis=3))
model.add(Dropout(0.2))

model.add(Conv2D(256, (3, 3), padding='same', activation = 'relu'))
model.add(BatchNormalization(axis=3))
model.add(Dropout(0.2))

model.add(Conv2D(512, (3, 3), padding='same', strides=(1, 2), activation = 'relu'))
model.add(BatchNormalization(axis=3))
model.add(Dropout(0.2))


model.add(Conv2D(512, (3, 3), padding='same', activation = 'relu'))
model.add(BatchNormalization(axis=3))
model.add(Dropout(0.2))

model.add(Conv2D(512, (3, 3), padding='same', strides=(1, 2), activation = 'relu'))
model.add(BatchNormalization(axis=3))
model.add(Dropout(0.2))

model.add(Conv2D(512, (3, 3), padding='same', activation = 'relu'))
model.add(BatchNormalization(axis=3))
model.add(Dropout(0.2))

model.add(Conv2D(1, (3, 3), padding='same'))


model.summary()

G = 1
if G > 1:
   model = make_parallel(model,G)


adam = keras.optimizers.adam(lr=1e-3)

model.compile(loss = 'mean_squared_error',
              optimizer=adam)


# ----------------------------------load training data and train on it --------------------------------------- 

num_training = 9920
X_train = np.zeros((num_training,1024,64,1))
y_train = np.zeros((num_training,1024,2,1))

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
							
							linescan_file = path + 'linescans/linescan_' + "{:.2g}".format(sigma*1e-09) + '_' + str(alpha) + '_' + "{0:.2g}".format(Xi*1e-09) + '_' + str(width) + '_' + str(space) + '.txt'
							noisy_file = path + 'noisy_images/nim_' + "{0:.2g}".format(sigma*1e-09) + '_' + str(alpha) + '_' + "{0:.2g}".format(Xi*1e-09) + '_' + str(width) + '_' + str(space) + '_' + str(-shift) + '_' + str(noise) + '.tiff'
							
							
							imnoisy = np.array(Image.open(noisy_file))
							imnoisy = imnoisy/256
							imnoisy = np.reshape(imnoisy,(1024,64,1))
							linescan = []
							#print(linescan_file)
							with open(linescan_file,'r') as f:
								for i,line in enumerate(f):
									if i < 3000:
										a, b = line.split(',')
										linescan.append(float(b))
									else:
										break
							linescan = linescan[:2048]
							leftline = np.array(linescan[:1024])
							rightline = linescan[1024:]
							rightline.reverse()
							rightline = np.array(rightline)
							
							leftline = leftline + shift
							rightline = rightline + shift
							leftline = np.reshape(leftline,(1024,1))
							rightline = np.reshape(rightline,(1024,1))

							X_train[count] = imnoisy
							y_train[count,:,0] = leftline.astype(int)/64
							y_train[count,:,1] = rightline.astype(int)/64
							count += 1
		print("alpha set, Training count :",alpha,',',count)
		history = model.fit(X_train, y_train, batch_size=batch_size, epochs=1, shuffle=True)
	print('Running validation now for epoch ' + str(epoch))
	val_score = model.evaluate(X_val,y_val)
	print('Validation score:',val_score)
	model.save(path + 'models/' + 'EDGEnet2_run_epoch_'+ str(epoch) + '.h5')

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

