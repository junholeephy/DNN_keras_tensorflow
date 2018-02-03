##!/usr/bin/env python
import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import keras.backend as K
import math

from ROOT import TFile, TTree, TCut, TH1F
from subprocess import call
from os.path import isfile
from array import array

from root_numpy import fill_hist
from root_numpy import root2array, tree2array, array2root
from root_numpy import testdata

from keras import callbacks
from keras.models import Sequential
from keras.layers import Input
from keras.layers.core import Dense, Activation, Dropout
from keras.layers import BatchNormalization
from keras.regularizers import l1, l2
from keras import initializers
from keras import layers
from keras.optimizers import SGD, Adam
from keras.constraints import maxnorm

#data = TFile.Open('input_TTZ_Delphes_small_new.root')
data = TFile.Open('input_TTZ_DelphesEvalGen_5275k.root')
tree = data.Get('Tree')


#ADAM = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
#################pre!!!!!!
MEM = True   #######if you want MEM or KIN.
#MEM = False

if(MEM == True):
	KIN = False
else:
	KIN = True
	MEM = False
############################################!!!!!!!#########
if(MEM == True):
	upper_limit = 90000   # 80000
else: 
	upper_limit = 280000

EPOCHS = 1000
NODENUM = 20
BATCH_SIZE_train = 500  # 500
BATCH_SIZE_test = 1  # 1
OPTIMIZER = 'adam'  #'adam', 'adagrad', 'sgd', 'adadelta'
KERNAL_INIT = 'truncated_normal'  #truncated_normal, glorot_uniform, random_uniform, he_uniform, he_normal
L2 = 1e-5  #1e-5
DROP_RATE = 0.0  # 0.2
MAXNORM = 100000 #5

LOAD_WEIGHTS = False   #True #False

if(MEM==True):
	LOAD_MODEL_WEIGHTS = 'self_models/mem_model_weights_EN90000_LN2_E5000_NN30_B500_adam_L1e-05_DR0.0.h5'  #MEM
else:
	LOAD_MODEL_WEIGHTS = 'self_models/starter1_kin_model_weights_EN260000_LN1_E1000_NN50_B500_adam_LR1e-05_DR0.0.h5' 	#KIN
###############################################################################################################
###sgd = SGD(lr=0.1, momentum=0.9, decay=0.0, nesterov=False)
###model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])  # cross_entropy is for classification 

####################################### Input DATA Sets !!!!! 
reco_bj1_Energy_	= tree2array(tree, branches='multilepton_Bjet1_P4->Energy()')			
reco_bj1_Theta_		= tree2array(tree, branches='multilepton_Bjet1_P4->Theta()')			
reco_bj1_Phi_		= tree2array(tree, branches='multilepton_Bjet1_P4->Phi()')			
reco_bj2_Energy_	= tree2array(tree, branches='multilepton_Bjet2_P4->Energy()')			
reco_bj2_Theta_		= tree2array(tree, branches='multilepton_Bjet2_P4->Theta()')			
reco_bj2_Phi_		= tree2array(tree, branches='multilepton_Bjet2_P4->Phi()')			
reco_MW1_Energy_	= tree2array(tree, branches='multilepton_JetClosestMw1_P4->Energy()')
reco_MW1_Theta_		= tree2array(tree, branches='multilepton_JetClosestMw1_P4->Theta()')
reco_MW1_Phi_		= tree2array(tree, branches='multilepton_JetClosestMw1_P4->Phi()')
reco_MW2_Energy_	= tree2array(tree, branches='multilepton_JetClosestMw2_P4->Energy()')
reco_MW2_Theta_		= tree2array(tree, branches='multilepton_JetClosestMw2_P4->Theta()')
reco_MW2_Phi_		= tree2array(tree, branches='multilepton_JetClosestMw2_P4->Phi()')
reco_l1_Energy_		= tree2array(tree, branches='multilepton_Lepton1_P4->Energy()')
reco_l1_Theta_		= tree2array(tree, branches='multilepton_Lepton1_P4->Theta()')
reco_l1_Phi_		= tree2array(tree, branches='multilepton_Lepton1_P4->Phi()')
reco_l2_Energy_		= tree2array(tree, branches='multilepton_Lepton2_P4->Energy()')
reco_l2_Theta_		= tree2array(tree, branches='multilepton_Lepton2_P4->Theta()')
reco_l2_Phi_		= tree2array(tree, branches='multilepton_Lepton2_P4->Phi()')
reco_l3_Energy_		= tree2array(tree, branches='multilepton_Lepton3_P4->Energy()')
reco_l3_Theta_		= tree2array(tree, branches='multilepton_Lepton3_P4->Theta()')
reco_l3_Phi_		= tree2array(tree, branches='multilepton_Lepton3_P4->Phi()')
reco_mET_Pt_		= tree2array(tree, branches='multilepton_mET->Pt()')
reco_mET_Phi_		= tree2array(tree, branches='multilepton_mET->Phi()') 
mHT_			= tree2array(tree, branches='multilepton_mHT')

Gen_BjetTopHad_E_		= tree2array(tree, branches='Gen_BjetTopHad_E')
Gen_WTopHad_mW_		= tree2array(tree, branches='Gen_WTopHad_mW')
Gen_BjetTopLep_E_		= tree2array(tree, branches='Gen_BjetTopLep_E')
Gen_NeutTopLep_Phi_		= tree2array(tree, branches='Gen_NeutTopLep_Phi')
Gen_WTopLep_mW_		= tree2array(tree, branches='Gen_WTopLep_mW')

Kin_BjetTopHad_E_       = tree2array(tree, branches='Kin_BjetTopHad_E')
Kin_WTopHad_mW_     = tree2array(tree, branches='Kin_WTopHad_mW')
Kin_BjetTopLep_E_       = tree2array(tree, branches='Kin_BjetTopLep_E')
Kin_NeutTopLep_Phi_     = tree2array(tree, branches='Kin_NeutTopLep_Phi')
Kin_WTopLep_mW_     = tree2array(tree, branches='Kin_WTopLep_mW')
###############################################################################################################


##################################### Target DATA !!!!!
mc_mem_ttz_weight_evalgenmax_log = tree2array(tree, branches='mc_mem_ttz_weight_evalgenmax_log')
mc_kin_ttz_weight_logmax = tree2array(tree, branches='mc_kin_ttz_weight_logmax')
###############################################################################################################


##################################### MEM/KIN's Valid events !!!!!
if(MEM == True):
	I1 = 0
	for i1 in range(mc_mem_ttz_weight_evalgenmax_log.size):
		if(mc_mem_ttz_weight_evalgenmax_log[i1] > -600):
			I1 = I1 + 1
	print("MEM's Total event number is : ", mc_mem_ttz_weight_evalgenmax_log.size)
	print("MEM's Valid event number is : ",  I1)
	num_Valid = np.zeros(I1)
	I2 = 0
	for i2 in range(mc_mem_ttz_weight_evalgenmax_log.size):
		if(mc_mem_ttz_weight_evalgenmax_log[i2] > -600):
			num_Valid[I2] = i2
			I2 = I2 + 1
else:
	I1 = 0
	for i1 in range(mc_kin_ttz_weight_logmax.size):
	        if(mc_kin_ttz_weight_logmax[i1] > -600):
	                I1 = I1 + 1
	print("KIN's Total event number is : ", mc_kin_ttz_weight_logmax.size)
	print("KIN's Valid event number is : ",  I1)
	num_Valid = np.zeros(I1)
	I2 = 0
	for i2 in range(mc_kin_ttz_weight_logmax.size):
	        if(mc_kin_ttz_weight_logmax[i2] > -600):
	                num_Valid[I2] = i2
	                I2 = I2 + 1
###############################################################################################################



##################################### Pick valid Input/Target DATA  (TARGET is output data) !!!!!
reco_bj1_Energy = np.zeros(I2)
reco_bj1_Theta	= np.zeros(I2)
reco_bj1_Phi	= np.zeros(I2)
reco_bj2_Energy	= np.zeros(I2)
reco_bj2_Theta	= np.zeros(I2)
reco_bj2_Phi	= np.zeros(I2)
reco_MW1_Energy	= np.zeros(I2)
reco_MW1_Theta	= np.zeros(I2)
reco_MW1_Phi	= np.zeros(I2)
reco_MW2_Energy	= np.zeros(I2)
reco_MW2_Theta	= np.zeros(I2)
reco_MW2_Phi	= np.zeros(I2)
reco_l1_Energy	= np.zeros(I2)
reco_l1_Theta	= np.zeros(I2)
reco_l1_Phi	= np.zeros(I2)
reco_l2_Energy	= np.zeros(I2)
reco_l2_Theta	= np.zeros(I2)
reco_l2_Phi	= np.zeros(I2)
reco_l3_Energy	= np.zeros(I2)
reco_l3_Theta	= np.zeros(I2)
reco_l3_Phi	= np.zeros(I2)
reco_mET_Pt	= np.zeros(I2)
reco_mET_Phi	= np.zeros(I2)
mHT		= np.zeros(I2)
TARGET	= np.zeros(I2)

Gen_BjetTopHad_E	= np.zeros(I2)
Gen_WTopHad_mW      = np.zeros(I2)
Gen_BjetTopLep_E	= np.zeros(I2)
Gen_NeutTopLep_Phi  = np.zeros(I2)
Gen_WTopLep_mW	= np.zeros(I2)

Kin_BjetTopHad_E    = np.zeros(I2)
Kin_WTopHad_mW      = np.zeros(I2)
Kin_BjetTopLep_E    = np.zeros(I2)
Kin_NeutTopLep_Phi  = np.zeros(I2)
Kin_WTopLep_mW  = np.zeros(I2)

for j1 in range(reco_bj1_Energy.size):
	jj1 = int(num_Valid[j1])
	reco_bj1_Energy[j1]	= reco_bj1_Energy_[jj1]
	reco_bj1_Theta[j1]	= reco_bj1_Theta_[jj1]
	reco_bj1_Phi[j1]	= reco_bj1_Phi_[jj1]
	reco_bj2_Energy[j1]	= reco_bj2_Energy_[jj1]
	reco_bj2_Theta[j1]	= reco_bj2_Theta_[jj1]
	reco_bj2_Phi[j1]	= reco_bj2_Phi_[jj1]
	reco_MW1_Energy[j1]	= reco_MW1_Energy_[jj1]
	reco_MW1_Theta[j1]	= reco_MW1_Theta_[jj1]
	reco_MW1_Phi[j1]	= reco_MW1_Phi_[jj1]
	reco_MW2_Energy[j1]	= reco_MW2_Energy_[jj1]
	reco_MW2_Theta[j1]	= reco_MW2_Theta_[jj1]
	reco_MW2_Phi[j1]	= reco_MW2_Phi_[jj1]
	reco_l1_Energy[j1]	= reco_l1_Energy_[jj1]
	reco_l1_Theta[j1]	= reco_l1_Theta_[jj1]
	reco_l1_Phi[j1]		= reco_l1_Phi_[jj1]
	reco_l2_Energy[j1]	= reco_l2_Energy_[jj1]
	reco_l2_Theta[j1]	= reco_l2_Theta_[jj1]
	reco_l2_Phi[j1]		= reco_l2_Phi_[jj1]
	reco_l3_Energy[j1]	= reco_l3_Energy_[jj1]
	reco_l3_Theta[j1]	= reco_l3_Theta_[jj1]
	reco_l3_Phi[j1]		= reco_l3_Phi_[jj1]
	reco_mET_Pt[j1]		= reco_mET_Pt_[jj1]
	reco_mET_Phi[j1]	= reco_mET_Phi_[jj1]
	mHT[j1]			= mHT_[jj1]
	if(MEM==True):
		TARGET[j1]		= mc_mem_ttz_weight_evalgenmax_log[jj1]
	else:
		TARGET[j1]      = mc_kin_ttz_weight_logmax[jj1]

	Gen_BjetTopHad_E[j1]    	= Gen_BjetTopHad_E_[jj1]
	Gen_WTopHad_mW[j1]      	= Gen_WTopHad_mW_[jj1]
	Gen_BjetTopLep_E[j1]    	= Gen_BjetTopLep_E_[jj1]
	Gen_NeutTopLep_Phi[j1]    	= Gen_NeutTopLep_Phi_[jj1]
	Gen_WTopLep_mW[j1]      	= Gen_WTopLep_mW_[jj1]

	Kin_BjetTopHad_E[j1]        = Kin_BjetTopHad_E_[jj1]
	Kin_WTopHad_mW[j1]          = Kin_WTopHad_mW_[jj1]
	Kin_BjetTopLep_E[j1]        = Kin_BjetTopLep_E_[jj1]
	Kin_NeutTopLep_Phi[j1]      = Kin_NeutTopLep_Phi_[jj1]
	Kin_WTopLep_mW[j1]          = Kin_WTopLep_mW_[jj1]
###############################################################################################################



##################################### Test length of input DATA !!!!!
if(reco_bj1_Energy.size == reco_bj1_Theta.size == reco_bj1_Phi.size == reco_bj2_Energy.size == reco_bj2_Theta.size == reco_bj2_Phi.size \
	== reco_MW1_Energy.size == reco_MW1_Theta.size == reco_MW1_Phi.size == reco_MW2_Energy.size == reco_MW2_Theta.size \
	== reco_MW2_Phi.size == reco_l1_Energy.size == reco_l1_Theta.size == reco_l1_Phi.size == reco_l2_Energy.size == reco_l2_Theta.size \
	==reco_l2_Phi.size == reco_l3_Energy.size == reco_l3_Theta.size == reco_l3_Phi.size == reco_mET_Pt.size == reco_mET_Phi.size == \
	mHT.size
	):
	print("Same sizes of numpy arrays!")
	SIZE = reco_bj1_Energy.size
else:
	print("Different sizes of numpy arrays. Error.")

#test1 = np.zeros(SIZE*2)
#test1 = test1.reshape(SIZE,2)
#print(test1.shape)
#for jj in range(SIZE):
#	test1[jj,] = np.append(reco_bj1_Energy[jj], reco_bj1_Theta[jj])
###############################################################################################################


##################################### ARRAY is input DATA !!!!!
##################################### TARGET_MEM is output data !!!!!
if(MEM==True):
	ARRAY = np.stack((reco_bj1_Energy, reco_bj1_Theta, reco_bj1_Phi, reco_bj2_Energy, reco_bj2_Theta, reco_bj2_Phi, reco_MW1_Energy, reco_MW1_Theta, reco_MW1_Phi, reco_MW2_Energy, reco_MW2_Theta, reco_MW2_Phi, reco_l1_Energy, reco_l1_Theta, reco_l1_Phi, reco_l2_Energy, reco_l2_Theta, reco_l2_Phi, reco_l3_Energy, reco_l3_Theta, reco_l3_Phi, reco_mET_Pt, reco_mET_Phi, mHT, Gen_BjetTopHad_E, Gen_WTopHad_mW, Gen_BjetTopLep_E, Gen_NeutTopLep_Phi, Gen_WTopLep_mW))
else:
	ARRAY = np.stack((reco_bj1_Energy, reco_bj1_Theta, reco_bj1_Phi, reco_bj2_Energy, reco_bj2_Theta, reco_bj2_Phi, reco_MW1_Energy, reco_MW1_Theta, reco_MW1_Phi, reco_MW2_Energy, reco_MW2_Theta, reco_MW2_Phi, reco_l1_Energy, reco_l1_Theta, reco_l1_Phi, reco_l2_Energy, reco_l2_Theta, reco_l2_Phi, reco_l3_Energy, reco_l3_Theta, reco_l3_Phi, reco_mET_Pt, reco_mET_Phi, mHT, Kin_BjetTopHad_E, Kin_WTopHad_mW, Kin_BjetTopLep_E, Kin_NeutTopLep_Phi, Kin_WTopLep_mW))

print(ARRAY.shape)
ARRAY = ARRAY.T
print(ARRAY.shape)
print(ARRAY[:2])
print(TARGET[:2])
###############################################################################################################

#MEM 109488
#KIN 318342

ndim = ARRAY.shape[1]  #29
nEvents = ARRAY.shape[0]
#nEventsBatch = 4000
#epochs = nEvents//nEventsBatch
#print("Number of epochs : ",epochs)

data_train = ARRAY[0:upper_limit]
target_train = TARGET[0:upper_limit]
data_test = ARRAY[(upper_limit+1):ARRAY.shape[0]]
target_test = TARGET[(upper_limit+1):ARRAY.shape[0]]

#for k1 in range(epochs):
#	test_mask = np.random.choice(data_train.shape[0], 50000)
#	data_test = data_train[test_mask]
#	target_test = target_train[test_mask]

ijkl = 0

modelRegress = Sequential()
#modelRegress.add(Dropout(DROP_RATE, input_shape=(ndim,)))
#modelRegress.add(Dense(NODENUM, kernel_initializer=KERNAL_INIT, activation='elu', W_regularizer=l2(L2), kernel_constraint=maxnorm(3))) 
modelRegress.add(Dense(NODENUM, kernel_initializer=KERNAL_INIT, activation='elu', W_regularizer=l2(L2), input_dim=ndim, kernel_constraint=maxnorm(MAXNORM)))
#modelRegress.add(BatchNormalization())
#modelRegress.add(Activation('elu'))
modelRegress.add(Dropout(DROP_RATE))
ijkl = ijkl + 1


modelRegress.add(Dense(NODENUM, kernel_initializer=KERNAL_INIT, activation='elu', W_regularizer=l2(L2),kernel_constraint=maxnorm(MAXNORM)))
modelRegress.add(Dropout(DROP_RATE))
ijkl = ijkl + 1


modelRegress.add(Dense(NODENUM, kernel_initializer=KERNAL_INIT, activation='elu', W_regularizer=l2(L2),kernel_constraint=maxnorm(MAXNORM)))
modelRegress.add(Dropout(DROP_RATE))
ijkl = ijkl + 1


modelRegress.add(Dense(NODENUM, kernel_initializer=KERNAL_INIT, activation='elu', W_regularizer=l2(L2),kernel_constraint=maxnorm(MAXNORM)))
modelRegress.add(Dropout(DROP_RATE))
ijkl = ijkl + 1

modelRegress.add(Dense(NODENUM, kernel_initializer=KERNAL_INIT, activation='elu', W_regularizer=l2(L2),kernel_constraint=maxnorm(MAXNORM)))
modelRegress.add(Dropout(DROP_RATE))
ijkl = ijkl + 1


'''modelRegress.add(Dense(NODENUM, kernel_initializer=KERNAL_INIT, activation='elu', W_regularizer=l2(L2)))  #additional layer
modelRegress.add(Dropout(DROP_RATE))
ijkl = ijkl + 1'''

'''modelRegress.add(Dense(NODENUM, kernel_initializer=KERNAL_INIT, activation='elu', W_regularizer=l2(L2)))  #additional layer
modelRegress.add(Dropout(DROP_RATE))
ijkl = ijkl + 1'''

'''modelRegress.add(Dense(NODENUM, kernel_initializer=KERNAL_INIT, activation='elu', W_regularizer=l2(L2)))  #additional layer
modelRegress.add(Dropout(DROP_RATE))
ijkl = ijkl + 1'''

modelRegress.add(Dense(1, kernel_initializer=KERNAL_INIT,  activation='linear')) #sigmoid
modelRegress.compile(loss='mean_squared_error', optimizer=OPTIMIZER)
if(LOAD_WEIGHTS == True):  modelRegress.load_weights(LOAD_MODEL_WEIGHTS,by_name=True)
modelRegress.summary()
history_callback = modelRegress.fit(data_train, target_train, validation_data=(data_test, target_test), epochs=EPOCHS, batch_size=BATCH_SIZE_train)

new_loss_history = np.zeros(EPOCHS*2)
new_loss_history = new_loss_history.reshape(EPOCHS,2)
loss_history = history_callback.history["loss"]
loss_val_history = history_callback.history["val_loss"]
np_loss_history = np.array(loss_history)
np_loss_val_history = np.array(loss_val_history)
for ijk in range(EPOCHS):
	new_loss_history[ijk,] = np.append(np_loss_history[ijk], np_loss_val_history[ijk])
#print(new_loss_history)
if(MEM==True):
	np.savetxt("self_loss_values/loss_MEM_EN"+str(upper_limit)+"_LN"+str(ijkl)+"_E"+str(EPOCHS)+"_NN"+str(NODENUM)+"_B"+str(BATCH_SIZE_train)+"_"+str(OPTIMIZER)+"_L"+str(L2)+"_DR"+str(DROP_RATE)+".txt", new_loss_history, delimiter=" ")
else:
	np.savetxt("self_loss_values/loss_KIN_EN"+str(upper_limit)+"_LN"+str(ijkl)+"_E"+str(EPOCHS)+"_NN"+str(NODENUM)+"_B"+str(BATCH_SIZE_train)+"_"+str(OPTIMIZER)+"_L"+str(L2)+"_DR"+str(DROP_RATE)+".txt", new_loss_history, delimiter=" ")

predict_train = modelRegress.predict(data_train, batch_size=BATCH_SIZE_test)
predict_test = modelRegress.predict(data_test, batch_size=BATCH_SIZE_test)
#print(predict_train.shape)
#print(predict_test.shape)
#print(predict_train[:7])
#print(predict_train[:,0])
if(MEM==True):
	modelRegress.save_weights('self_models/mem_model_weights_EN'+str(upper_limit)+"_LN"+str(ijkl)+"_E"+str(EPOCHS)+"_NN"+str(NODENUM)+"_B"+str(BATCH_SIZE_train)+"_"+str(OPTIMIZER)+"_L"+str(L2)+"_DR"+str(DROP_RATE)+'.h5')
else:
	modelRegress.save_weights('self_models/kin_model_weights_EN'+str(upper_limit)+"_LN"+str(ijkl)+"_E"+str(EPOCHS)+"_NN"+str(NODENUM)+"_B"+str(BATCH_SIZE_train)+"_"+str(OPTIMIZER)+"_L"+str(L2)+"_DR"+str(DROP_RATE)+'.h5')


if(MEM==True):
	f = TFile("self_trees/self_tree_mem_EN"+str(upper_limit)+"_LN"+str(ijkl)+"_E"+str(EPOCHS)+"_NN"+str(NODENUM)+"_B"+str(BATCH_SIZE_train)+"_"+str(OPTIMIZER)+"_L"+str(L2)+"_DR"+str(DROP_RATE)+".root", "recreate")
else:
	f = TFile("self_trees/self_tree_kin_EN"+str(upper_limit)+"_LN"+str(ijkl)+"_E"+str(EPOCHS)+"_NN"+str(NODENUM)+"_B"+str(BATCH_SIZE_train)+"_"+str(OPTIMIZER)+"_L"+str(L2)+"_DR"+str(DROP_RATE)+".root", "recreate")


print("saved model/loss name : ")
if(MEM == True):
	print('mem_model_weights_EN'+str(upper_limit)+"_LN"+str(ijkl)+"_E"+str(EPOCHS)+"_NN"+str(NODENUM)+"_B"+str(BATCH_SIZE_train)+"_"+str(OPTIMIZER)+"_L"+str(L2)+"_DR"+str(DROP_RATE)+'.h5')
	print('loss_MEM_EN'+str(upper_limit)+"_LN"+str(ijkl)+"_E"+str(EPOCHS)+"_NN"+str(NODENUM)+"_B"+str(BATCH_SIZE_train)+"_"+str(OPTIMIZER)+"_L"+str(L2)+"_DR"+str(DROP_RATE)+'.txt')
else:
	print('kin_model_weights_EN'+str(upper_limit)+"_LN"+str(ijkl)+"_E"+str(EPOCHS)+"_NN"+str(NODENUM)+"_B"+str(BATCH_SIZE_train)+"_"+str(OPTIMIZER)+"_L"+str(L2)+"_DR"+str(DROP_RATE)+'.h5')
	print('loss_KIN_EN'+str(upper_limit)+"_LN"+str(ijkl)+"_E"+str(EPOCHS)+"_NN"+str(NODENUM)+"_B"+str(BATCH_SIZE_train)+"_"+str(OPTIMIZER)+"_L"+str(L2)+"_DR"+str(DROP_RATE)+'.txt')

hist_target_train = TH1F('TrainData','TrainData',100,-100,0)
hist_target_test = TH1F('TestData','TestData',100,-100,0)
hist_output_train = TH1F('OutputDataTrain','OutputDataTrain',100,-100,0)
hist_output_test = TH1F('OutputDataTest','OutputDataTest',100,-100,0)

tree_train = TTree('tree_train','tree_train')
Ttrain = np.zeros(1, dtype=float)
Otrain = np.zeros(1, dtype=float)
tree_train.Branch('target_train',Ttrain,'target_train/D')
tree_train.Branch('output_train',Otrain,'output_train/D')
for ij1 in range(upper_limit):
	Ttrain[0] = target_train[ij1] 
	Otrain[0] = predict_train[ij1]		
	tree_train.Fill()

tree_test = TTree('tree_test','tree_test')
Ttest = np.zeros(1, dtype=float)
Otest = np.zeros(1, dtype=float)
tree_test.Branch('target_test',Ttest,'target_test/D')
tree_test.Branch('output_test',Otest,'output_test/D')
for ij2 in range(ARRAY.shape[0]-upper_limit-1):
	Ttest[0] = target_test[ij2]
	Otest[0] = predict_test[ij2]
	tree_test.Fill()


#Plots: projection selon l'axe X (premiere variable)
fill_hist(hist_target_train, target_train)
fill_hist(hist_target_test, target_test)
fill_hist(hist_output_train, predict_train[:,0])
fill_hist(hist_output_test, predict_test[:,0])

f.Write()
f.Close()

