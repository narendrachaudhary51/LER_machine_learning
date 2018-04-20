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

X_val = np.zeros((num_validation,1024,64,1))
y_val = np.zeros((num_validation,1024,64,2))


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
						
						original_file = path + 'original_images/oim_' + "{:.2g}".format(sigma*1e-09) + '_' + str(alpha) + '_' + "{0:.2g}".format(Xi*1e-09) + '_' + str(width) + '_' + str(space) + '_' + str(-shift) + '.tiff'
						noisy_file = path + 'noisy_images/nim_' + "{0:.2g}".format(sigma*1e-09) + '_' + str(alpha) + '_' + "{0:.2g}".format(Xi*1e-09) + '_' + str(width) + '_' + str(space) + '_' + str(-shift) + '_' + str(noise) + '.tiff'
						linescan_file = path + 'linescans/linescan_' + "{:.2g}".format(sigma*1e-09) + '_' + str(alpha) + '_' + "{0:.2g}".format(Xi*1e-09) + '_' + str(width) + '_' + str(space) + '.txt'
						
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
						leftline = leftline + shift
						rightline = rightline + shift
						
						edgeimage = np.zeros((1024,64))
						for i in range(1024):
							if leftline[i] < 0:
								leftline[i] = 0
							if rightline[i] > 63:
								rightline[i] = 63
							edgeimage[i, leftline[i].astype(int)] = 1
							edgeimage[i, rightline[i].astype(int)] = 1

						im = np.array(Image.open(original_file))
						imnoisy = np.array(Image.open(noisy_file))

						X_val[count,:,:,0] = imnoisy/256
						y_val[count,:,:,0] = im/256
						y_val[count,:,:,1] = edgeimage
						count += 1
print('Validation_count: ',count)						

print('Validation data shape: ', X_val.shape)
print('Validation labels shape: ', y_val.shape)

	

#print('Train data shape: ', X_train.shape)
#print('Train labels shape: ', y_train.shape)

batch_size = 8
epochs = 4

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

model.add(Conv2D(2, (3, 3), padding='same'))


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
y_train = np.zeros((num_training,1024,64,2))

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
							
							original_file = path + 'original_images/oim_' + "{:.2g}".format(sigma*1e-09) + '_' + str(alpha) + '_' + "{0:.2g}".format(Xi*1e-09) + '_' + str(width) + '_' + str(space) + '_' + str(-shift) + '.tiff'
							noisy_file = path + 'noisy_images/nim_' + "{0:.2g}".format(sigma*1e-09) + '_' + str(alpha) + '_' + "{0:.2g}".format(Xi*1e-09) + '_' + str(width) + '_' + str(space) + '_' + str(-shift) + '_' + str(noise) + '.tiff'
							linescan_file = path + 'linescans/linescan_' + "{:.2g}".format(sigma*1e-09) + '_' + str(alpha) + '_' + "{0:.2g}".format(Xi*1e-09) + '_' + str(width) + '_' + str(space) + '.txt'
						
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
							leftline = leftline + shift
							rightline = rightline + shift
						
							edgeimage = np.zeros((1024,64))
							for i in range(1024):
								if leftline[i] < 0:
									leftline[i] = 0
								if rightline[i] > 63:
									rightline[i] = 63
								edgeimage[i, leftline[i].astype(int)] = 1
								edgeimage[i, rightline[i].astype(int)] = 1

							im = np.array(Image.open(original_file))
							imnoisy = np.array(Image.open(noisy_file))
							
							im = im/256
							imnoisy = imnoisy/256
							#im = np.reshape(im,(1024,64,1))
							#imnoisy = np.reshape(imnoisy,(1024,64,1))
							X_train[count,:,:,0] = imnoisy
							y_train[count,:,:,0] = im
							y_train[count,:,:,1] = edgeimage
							count += 1
		print("alpha set, Training count :",alpha,',',count)
		history = model.fit(X_train, y_train, batch_size=batch_size, epochs=1, shuffle=True)
	print('Running validation now for epoch ' + str(epoch))
	val_score = model.evaluate(X_val,y_val)
	print('Validation score:',val_score)
	model.save(path + 'models/' + 'Linenet_run1_epoch_'+ str(epoch) + '.h5')


del model  # deletes the existing model


print("Execttion Time= ", time.time() - start)


