####  vistualize model 
#!/usr/bin/env python
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
import tensorflow as tf
import keras.backend as K

from ROOT import TH1F, TFile, TTree
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
from keras.utils import plot_model
from keras.utils.vis_utils import plot_model
####=================================================####
from A_sample_data import sample_data
from A_Camel import Camel1D
from A_Camel import CamelND
from A_Camel import tfCamelND
####=================================================####
ndim=1
nEvent=10000     #test
nEvents=50000    #train
	
data_train = sample_data(nEvents,1,ndim)
data_test = sample_data(nEvent,1,ndim)

vectors = []
for x in data_train:
	vectors.append(CamelND(x,0.1,ndim))
target_train = np.array(vectors)

vectors = []
for x in data_test:
	vectors.append(CamelND(x,0.1,ndim))
target_test = np.array(vectors)

#### REGRESSION MODEL ####
modelRegress = Sequential()
modelRegress.add(Dense(64, kernel_initializer='truncated_normal', activation='relu', W_regularizer=l2(1e-5), input_dim=ndim))
modelRegress.add(Dense(64, kernel_initializer='truncated_normal', activation='relu', W_regularizer=l2(1e-5)))
modelRegress.add(Dense(64, kernel_initializer='truncated_normal', activation='relu', W_regularizer=l2(1e-5)))
modelRegress.add(Dense(64, kernel_initializer='truncated_normal', activation='relu', W_regularizer=l2(1e-5)))
modelRegress.add(Dense(64, kernel_initializer='truncated_normal', activation='relu', W_regularizer=l2(1e-5)))
modelRegress.add(Dense(1, kernel_initializer='truncated_normal',  activation='linear')) #sigmoid
modelRegress.compile(loss='mean_squared_error', optimizer='adam')
modelRegress.summary()
modelRegress.fit(data_train, target_train, validation_data=(data_test, target_test), epochs=200, batch_size=500)
modelRegress.save("model.h5")
#score = model.evaluate(data_test, target_test, batch_size=50)
predict_train = modelRegress.predict(data_train, batch_size=1)
predict_test = modelRegress.predict(data_test, batch_size=1) 
plot_model(modelRegress, to_file='model_test1.eps', show_shapes=True, show_layer_names=True)

layer_0 = modelRegress.layers[0]
layer_1 = modelRegress.layers[1]
#layer_2 = modelRegress.layers[2]
#layer_3 = modelRegress.layers[3]
#layer_4 = modelRegress.layers[4]
#layer_5 = modelRegress.layers[5]

weight_0 = layer_0.get_weights()
weight_1 = layer_1.get_weights()
weight_0_m = weight_0[0]          # multiplying
weight_0_p = weight_0[1]          # adding
weight_1_m = weight_1[0]
weight_1_p = weight_1[1]
print(type(weight_0))
print(type(weight_1))
print(type(weight_0_m))
print(type(weight_0_p))
print(type(weight_1_m))
print(type(weight_1_p))
#print(weight_0)
#print(weight_1)

f = TFile("Camel.root", "recreate")
hist_target_train = TH1F('TrainData','TrainData',100,0,1)
hist_target_test = TH1F('TestData','TestData',100,0,1)
hist_output_train = TH1F('OutputDataTrain','OutputDataTrain',100,0,1)
hist_output_test = TH1F('OutputDataTest','OutputDataTest',100,0,1)

fill_hist(hist_target_train, data_train[:,0], target_train)
fill_hist(hist_target_test, data_test[:,0], target_test)
fill_hist(hist_output_train, data_train[:,0], predict_train[:,0])
fill_hist(hist_output_test, data_test[:,0], predict_test[:,0])


tree_train = TTree('tree_train','tree_train')
Ttrain = np.zeros(1, dtype=float)
Otrain = np.zeros(1, dtype=float)
Dtrain = np.zeros(1, dtype=float)
tree_train.Branch('target_train',Ttrain,'target_train/D')
tree_train.Branch('output_train',Otrain,'output_train/D')
tree_train.Branch('data_train',Dtrain,'data_train/D')
for ij1 in range(nEvents):
    Ttrain[0] = target_train[ij1]
    Otrain[0] = predict_train[ij1]
    Dtrain[0] = data_train[ij1,0]
    tree_train.Fill()

tree_test = TTree('tree_test','tree_test')
Ttest = np.zeros(1, dtype=float)
Otest = np.zeros(1, dtype=float)
Dtest = np.zeros(1, dtype=float)
tree_test.Branch('target_test',Ttest,'target_test/D')
tree_test.Branch('output_test',Otest,'output_test/D')
tree_test.Branch('data_test',Dtest,'data_test/D')
for ij2 in range(nEvent):
    Ttest[0] = target_test[ij2]
    Otest[0] = predict_test[ij2]
    Dtest[0] = data_test[ij2,0]
    tree_test.Fill()


f.Write()
f.Close()
