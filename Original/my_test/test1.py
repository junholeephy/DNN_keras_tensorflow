import numpy as np
from A_sample_data import sample_data
from A_Camel import Camel1D
from A_Camel import CamelND
from A_Camel import tfCamelND

ndim=2
nEvent=100     #test
nEvents=100    #train

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

for i in range(nEvents):
    print data_train[i,0]
print data_train[:,0]
#print target_train
#print data_test[:,0]
#print data_test
#print target_test



