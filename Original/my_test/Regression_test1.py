#!/usr/bin/env python
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
import tensorflow as tf
import keras.backend as K

from ROOT import TH1F, TFile
from root_numpy import fill_hist
from root_numpy import root2array, tree2array, array2root
from root_numpy import testdata

import math
from subprocess import call
from os.path import isfile

from keras import callbacks
from keras.models import Sequential
from keras.layers import Input
from keras.layers.core import Dense, Activation, Dropout
from keras.layers import BatchNormalization
from keras.regularizers import l1, l2
from keras import initializers
from keras import layers
from keras.optimizers import SGD, Adam

def Camel1D(x,a):
	return math.log(0.5/(a*math.sqrt(math.pi)) * (math.exp(-(x-1./3)*(x-1/3.)/(a*a)) + math.exp(-(x-2./3)*(x-2/3.)/(a*a))))
	
def CamelND(x,a=0.1,n=1):
	sum1 = 0
	sum2 = 0
	denom = 1
	for i in range(n):
		sum1 += -(x[i]-1./3)*(x[i]-1/3.)/(a*a)
		sum2 += -(x[i]-2./3)*(x[i]-2/3.)/(a*a)
		denom *= (a*math.sqrt(math.pi))
	sum1 = math.exp(sum1)
	sum2 = math.exp(sum2)
	sum = math.log((0.5/denom) * (sum1+sum2))
	return sum
	
def tfCamelND(x,a=0.1,n=1):
	sum1 = 0
	sum2 = 0
	denom = 1
	for i in range(n):
		sum1 += -(x[i]-1./3)*(x[i]-1/3.)/(a*a)
		sum2 += -(x[i]-2./3)*(x[i]-2/3.)/(a*a)
		denom *= (a*math.sqrt(math.pi))
	sum1 = K.exp(sum1)
	sum2 = K.exp(sum2)
	sum = (0.5/denom) * (sum1+sum2)
	#print(sum)
	return sum

def sample_data(n_samples=1000, maxval=1, n=1):
	vectors = []
	for i in range(n_samples):
		#subvector = []
		#for j in range(0,n):
		val = np.random.random(n) *  maxval;
		#	subvector.append(val)
		#print (val)
		vectors.append(val)
	return np.array(vectors)
	
	
	

ndim=1
nEvents=50000
nEventsBatch=2000
#print (nEvents//nEventsBatch)
	
data_train = sample_data(nEvents,1,ndim)
data_test = sample_data(nEvents,1,ndim)
data_train_batch = data_train
#print (data_train)

vectors = []
for x in data_train:
	#print (x)
	#print (x, Camel1D(x,0.1)) 
	vectors.append(CamelND(x,0.1,ndim))
target_train = np.array(vectors)

vectors = []
for x in data_test:
	#print (x, Camel1D(x,0.1)) 
	vectors.append(CamelND(x,0.1,ndim))
target_test = np.array(vectors)

#### REGRESSION MODEL ####
modelRegress = Sequential()
modelRegress.add(Dense(64, kernel_initializer='truncated_normal', activation='relu', W_regularizer=l2(1e-5), input_dim=ndim))
modelRegress.add(Dense(64, kernel_initializer='truncated_normal', activation='relu', W_regularizer=l2(1e-5)))
#modelRegress.add(Dense(64, kernel_initializer='truncated_normal', activation='relu', W_regularizer=l2(1e-5)))
#modelRegress.add(Dense(64, kernel_initializer='truncated_normal', activation='relu', W_regularizer=l2(1e-5)))
#modelRegress.add(Dense(64, kernel_initializer='truncated_normal', activation='relu', W_regularizer=l2(1e-5)))
modelRegress.add(Dense(1, kernel_initializer='truncated_normal',  activation='linear')) #sigmoid
modelRegress.compile(loss='mean_squared_error', optimizer='adam')
modelRegress.summary()
modelRegress.fit(data_train, target_train, validation_data=(data_test, target_test), epochs=20, batch_size=500)
#modelRegress.save("model.h5")
#score = model.evaluate(data_test, target_test, batch_size=50)
predict_train = modelRegress.predict(data_train, batch_size=1)
predict_test = modelRegress.predict(data_test, batch_size=1) 



f = TFile("generative_tree_1D.root", "recreate")
hist_target_train = TH1F('TrainData','TrainData',100,0,1)
hist_target_test = TH1F('TestData','TestData',100,0,1)
hist_output_train = TH1F('OutputDataTrain','OutputDataTrain',100,0,1)
hist_output_test = TH1F('OutputDataTest','OutputDataTest',100,0,1)

#Plots: projection selon l'axe X (premiere variable)
fill_hist(hist_target_train, data_train[:,0], target_train)
fill_hist(hist_target_test, data_test[:,0], target_test)
fill_hist(hist_output_train, data_train[:,0], predict_train[:,0])
fill_hist(hist_output_test, data_test[:,0], predict_test[:,0])

f.Write()
f.Close()
