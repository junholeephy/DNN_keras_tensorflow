#!/usr/bin/env python

from ROOT import TMVA, TFile, TTree, TCut
from subprocess import call
from os.path import isfile

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.regularizers import l1, l2
from keras import initializers
from keras import layers
from keras.optimizers import SGD

# Setup TMVA
TMVA.Tools.Instance()
TMVA.PyMethodBase.PyInitialize()

output = TFile.Open('TMVA.root', 'RECREATE')
factory = TMVA.Factory('TMVARegression', output,
        '!V:!Silent:Color:DrawProgressBar:Transformations=D,G:AnalysisType=Regression')

# Load data
#if not isfile('tmva_reg_example.root'):
#    call(['curl', '-O', 'http://root.cern.ch/files/tmva_reg_example.root'])

data = TFile.Open('../../input_TTZ_Delphes_small_new.root')
tree = data.Get('Tree')

dataloader = TMVA.DataLoader('dataset')
#for branch in tree.GetListOfBranches():
#    name = branch.GetName()
#    if name != 'fvalue':
#        dataloader.AddVariable(name)

dataloader.AddVariable('multilepton_Bjet1_P4->E()')
dataloader.AddVariable('multilepton_Bjet1_P4->Theta()')
dataloader.AddVariable('multilepton_Bjet1_P4->Phi()')
dataloader.AddVariable('multilepton_Bjet2_P4->E()')
dataloader.AddVariable('multilepton_Bjet2_P4->Theta()')
dataloader.AddVariable('multilepton_Bjet2_P4->Phi()')
dataloader.AddVariable('multilepton_JetClosestMw1_P4->E()')
dataloader.AddVariable('multilepton_JetClosestMw1_P4->Theta()')
dataloader.AddVariable('multilepton_JetClosestMw1_P4->Phi()')
dataloader.AddVariable('multilepton_JetClosestMw2_P4->E()')
dataloader.AddVariable('multilepton_JetClosestMw2_P4->Theta()')
dataloader.AddVariable('multilepton_JetClosestMw2_P4->Phi()')
dataloader.AddVariable('multilepton_Lepton1_P4->E()')
dataloader.AddVariable('multilepton_Lepton1_P4->Theta()')
dataloader.AddVariable('multilepton_Lepton1_P4->Phi()')
dataloader.AddVariable('multilepton_Lepton2_P4->E()')
dataloader.AddVariable('multilepton_Lepton2_P4->Theta()')
dataloader.AddVariable('multilepton_Lepton2_P4->Phi()')
dataloader.AddVariable('multilepton_Lepton3_P4->E()')
dataloader.AddVariable('multilepton_Lepton3_P4->Theta()')
dataloader.AddVariable('multilepton_Lepton3_P4->Phi()')
dataloader.AddVariable('multilepton_mET->Pt()')
dataloader.AddVariable('multilepton_mET->Phi()')
dataloader.AddVariable('multilepton_mHT')
#dataloader.AddVariable('')

dataloader.AddTarget('Target_BjetTopHad_E')
#dataloader.AddTarget('Target_WTopHad_tW')
dataloader.AddTarget('Target_WTopHad_mW*10')

dataloader.AddRegressionTree(tree, 1.0)
dataloader.PrepareTrainingAndTestTree(TCut(''),
        'nTrain_Regression=40000:SplitMode=Random:NormMode=NumEvents:!V')

# Generate model

# Define initialization
def normal(shape, name=None):
    return initializers.normal(shape, scale=0.05, name=name)

# Define model
model = Sequential()
model.add(Dense(256, kernel_initializer='truncated_normal', activation='elu', W_regularizer=l2(1e-5), input_dim=24))
#model.add(Dropout(0.1))
model.add(Dense(256, kernel_initializer='truncated_normal', activation='elu', W_regularizer=l2(1e-5)))
#model.add(Dropout(0.1))
model.add(Dense(256, kernel_initializer='truncated_normal', activation='elu', W_regularizer=l2(1e-5)))
#model.add(Dropout(0.1))
model.add(Dense(256, kernel_initializer='truncated_normal', activation='elu', W_regularizer=l2(1e-5)))
#model.add(Dropout(0.1))
model.add(Dense(256, kernel_initializer='truncated_normal', activation='elu', W_regularizer=l2(1e-5)))
#model.add(Dense(80, kernel_initializer='truncated_normal', activation='tanh', W_regularizer=l2(1e-5), input_dim=24))
#model.add(Dense(80, kernel_initializer='truncated_normal', activation='tanh', W_regularizer=l2(1e-5), input_dim=24))
#model.add(Dense(50, kernel_initializer='random_uniform', activation='tanh', W_regularizer=l2(1e-5), input_dim=24))
#model.add(Dense(50, kernel_initializer='random_uniform', activation='tanh', W_regularizer=l2(1e-5), input_dim=24))
#model.add(Dense(150, kernel_initializer='random_uniform', activation='tanh', W_regularizer=l1(1e-5), input_dim=24))
#model.add(Dense(150, kernel_initializer='random_uniform', activation='tanh', W_regularizer=l2(1e-5), input_dim=12))
#model.add(Dense(32, init=normal, activation='tanh', W_regularizer=l2(1e-5)))
#model.add(layers.normalization.BatchNormalization());
#model.add(Dropout(0.1))
#model.add(Dense(1, kernel_initializer='truncated_normal', activation='linear'))
model.add(Dense(2, kernel_initializer='truncated_normal', activation='linear'))


# Set loss and optimizer
model.compile(loss='mean_squared_error', optimizer='adam')
#model.compile(loss='mean_squared_logarithmic_error', optimizer='adam')
#model.compile(loss='kullback_leibler_divergence', optimizer=SGD(lr=0.01))
#model.compile(loss='mean_squared_error', optimizer=SGD(lr=0.01))

# Store model to file
model.save('model.h5')
model.summary()

# Book methods
factory.BookMethod(dataloader, TMVA.Types.kPyKeras, 'PyKeras','H:!V:VarTransform=G:FilenameModel=model.h5:NumEpochs=15:BatchSize=500:SaveBestOnly=0')
#factory.BookMethod(dataloader, TMVA.Types.kPyKeras, 'PyKeras','H:!V:VarTransform=D,G:FilenameModel=model.h5:NumEpochs=50:BatchSize=20')
#factory.BookMethod(dataloader, TMVA.Types.kPyKeras, 'PyKeras',
#        'H:!V:VarTransform=D,G:FilenameModel=model.h5:NumEpochs=50:BatchSize=32')
#factory.BookMethod(dataloader, TMVA.Types.kBDT, 'BDTG',
#        '!H:!V:VarTransform=D,G:NTrees=1000:BoostType=Grad:Shrinkage=0.1:UseBaggedBoost:BaggedSampleFraction=0.5:nCuts=100:MaxDepth=4')

# Run TMVA

factory.TrainAllMethods()
factory.TestAllMethods()
factory.EvaluateAllMethods()

try:
    from keras.utils.visualize_util import plot
    plot(model, to_file='model.png', show_shapes=True)
except:
    print('[INFO] Failed to make model plot')
