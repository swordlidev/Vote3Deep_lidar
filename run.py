import numpy as np
import os
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten, Activation
from keras.layers import Input
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D, Convolution3D, ZeroPadding3D
from keras.layers.pooling import MaxPooling3D
from keras.optimizers import SGD, Adam
from keras.models import Model
from keras.utils import np_utils
from sklearn.metrics import mean_squared_error
import glob
from keras import __version__ as keras_version

np.random.seed(420)

def read_data ():
	data = []
	y1 = []
	y2 = []
	car = 0
	cyc = 0
	ped = 0
	files = glob.glob ('grid_files/*.csv')
	for i in range (len(files)):
		files[i] = files [i].replace ('.csv', '')
		f = files[i]
		dt = np.loadtxt (f+'.csv', delimiter=',')
		x = np.zeros (10*10*10*6)
		x = x.reshape ((6,10,10,10))
		for j in dt:
			for k in range (6):
				x[k][int(j[0]-1)][int(j[01]-1)][int(j[02]-1)] = j[k+3]
		yd = x
		dt = np.loadtxt (f+'.txt', delimiter=' ', dtype=np.str)
		r = np.zeros (3)
		for j in dt:
			x = np.zeros (3)
			if j[0] == 'Car' and r[0] != 1:
				x[0] = 1
				r[0] = 1
				car += 1
				y1.append (x)
				x = j[11:]
				x = np.array(x, dtype='|S24')
				x = x.astype(np.float32)
				y2.append (x)
				data.append (yd)
			elif j[0] == 'Cyclist' and r[01] != 1:
				x[1] = 1
				r[1] = 1
				cyc += 1
				y1.append (x)
				x = j[11:]
				x = np.array(x, dtype='|S24')
				x = x.astype(np.float32)
				y2.append (x)
				data.append (yd)
			elif j[0] == 'Pedestrian' and r[02] != 1:
				x[2] = 1
				r[2] = 1
				ped += 1
				y1.append (x)
				x = j[11:]
				x = np.array(x, dtype='|S24')
				x = x.astype(np.float32)
				y2.append (x)
				data.append (yd)

	data, y1, y2 = np.array (data), np.array (y1), np.array (y2)
	arr = np.arange(len (data))
	np.random.shuffle(arr)
	TX = data[arr[0:len (arr)*0.6]]
	tX = data[arr[len (arr)*0.6:len (arr)*0.895]]
	vX = data[arr[len (arr)*0.895:]]
	Ty1 = y1[arr[0:len (arr)*0.6]]
	ty1 = y1[arr[len (arr)*0.6:len (arr)*0.895]]
	vy1 = y1[arr[len (arr)*0.895:]]
	Ty2 = y2[arr[0:len (arr)*0.6]]
	ty2 = y2[arr[len (arr)*0.6:len (arr)*0.895]]
	vy2 = y2[arr[len (arr)*0.895:]]
	print 'Cars = ', car, 'Peds = ', ped, 'Cyc = ', cyc
	return TX, Ty1, Ty2, tX, ty1, ty2, vX, vy1, vy2

def calc_err (a, b):
	s = np.zeros (len(a))
	for i in range (len (a)):
		s[i] = ((a[i] - b[i]) ** 2) 

def gen_model (input_shape=(6,10,10,10)):
	model = Sequential()
	inp = Input (input_shape)
	model.add(Convolution3D(8, 3, 3, 3, border_mode='same',
							input_shape=input_shape))
	X = Convolution3D(8, 3, 3, 3, border_mode='same')(inp)
	model.add(Activation('relu'))
	X = Activation('relu') (X)
	model.add(ZeroPadding3D())
	X = ZeroPadding3D()(X)
	model.add(Convolution3D(8, 3, 3, 3))
	X = Convolution3D(8, 3, 3, 3)(X)
	model.add(Activation('relu'))
	X = Activation('relu')(X)
	model.add(ZeroPadding3D())
	X = ZeroPadding3D()(X)

	model.add(Convolution3D(1, 3, 3, 3))
	X = Convolution3D(1, 3, 3, 3)(X)
	model.add(Activation('relu'))
	X = Activation('relu')(X)
	model.add(Flatten())
	X = Flatten()(X)
	model.add(Dense(1024))
	Z = X
	X = Dense(1024)(X)
	model.add(Activation('relu'))
	X = Activation('relu')(X)
	model.add(Dense(1024))
	X = Dense(1024)(X)
	model.add(Activation('relu'))
	X = Activation('relu')(X)
	model.add(Dense(3))
	X1 = Dense(3)(X)
	model.add(Activation('softmax'))
	X1 = Activation('softmax', name='Class Act')(X1)
	model.add(Dense(400))
	X2 = Dense(400)(X)
	model.add(Activation('relu'))
	X2 = Activation('relu')(X2)
	model.add(Dense(4))
	X2 = Dense(4)(X2)
	model.add(Activation('relu'))

	X2 = Activation('relu', name='Box Act')(X2)
	model = Model (inp, [X1, X2])
	convModel = Model (inp, Z)
	sgd = SGD(lr=2e-30, decay=1e-4, momentum=0.9, nesterov=True)
	
	adm = Adam (lr = 1e-4)
	model.compile(optimizer=sgd, loss=['binary_crossentropy', 'mse']
		#, metrics = ['accuracy']
	)
	convModel.compile(optimizer=sgd, loss='mse'
		#, metrics = ['accuracy']
	)

	return model, convModel

if __name__ == '__main__':
	print('Keras version: {}'.format(keras_version))
	model, cm = gen_model ()
	model.summary ()
	print 'Reading data'
	TX, Ty1, Ty2, tX, ty1, ty2, vX, vy1, vy2 = read_data ()
	print (TX.shape)
	print (Ty1.shape)
	print (Ty2.shape)
	print (tX.shape)
	print (ty1.shape)
	print (ty2.shape)
	print (vX.shape)
	print (vy1.shape)
	print (vy2.shape)
	model.fit(TX, [Ty1, Ty2], batch_size=16, nb_epoch=1, validation_data=(vX, [vy1, vy2]))
	model.save_weights('vote3deep.h5')
	y1, y2 = model.predict(tX, batch_size=32, verbose=0)
	y = cm.predict (TX[0:3])
	np.savetxt ('0.txt', y)
	er1 = mean_squared_error (ty1, y1)
	er2 = mean_squared_error (ty2, y2)
	print 'Total error 1 = ', er1
	print 'Total error 2 = ', er2