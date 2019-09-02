
# coding: utf-8

# In[1]:


#import things to use 
import os
import numpy as np
import tensorflow
import keras
from keras.models import Sequential
from keras.layers import Activation, Masking, Dense 
from keras.layers import Convolution2D as Conv2D 
from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D 
from keras.layers import Softmax, ReLU, BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.utils import np_utils
from sklearn import metrics 


# In[2]:



#import training data
#see getLabels_stackedData3.py for preprocessing

#run for train data only
path = './'
os.chdir(path)

labels = []
allCols = []
allRows = []
allSTFTs = []
paddedSTFTs = []
maxTime = 0


#open file (lists all test/train .npy STFT file names) 
file = open('list.txt', 'r')
data = file.read().split('\n')
file.close()
print(data)

os.chdir(path)

for array in data:
    #loop through to find out if file is a buzz or minibuzz, and add label accordingly 
    nameParse = array.split("_",-1)
    typeParse = nameParse[4].split("u", -1)
    if(typeParse[0] == 'b'):
        labels.append(1)
    elif(typeParse[0] == 'minib'):
        labels.append(0)
    else: 
        print('Error, not a buzz or minibuzz!')
    
    curSTFT = np.load(array)
        
    #find cols (number of time steps) of each STFT and save longest one
    rows, cols = curSTFT.shape
    allCols.append(cols)
    allRows.append(rows)
    if (cols>maxTime):
        maxTime = cols
    if (rows!=1025):
        print('Error, not 1025 STFT coefficients') 
        
    allSTFTs.append(curSTFT)
    
print(labels)
print(allRows)
print(allCols)
print(maxTime) 

for array in data: 
    #loop though STFTs again to zero pad and transpose  
    #NOTE: must be done after we definitvely know the max number of time steps
    curSTFT = np.load(array)
    rows, cols = curSTFT.shape
    pad = maxTime-cols
    
    zeroPad = np.zeros((rows,maxTime-cols))
    paddedSTFT = np.append(curSTFT, zeroPad, axis = 1)
    
    paddedSTFT = np.transpose(paddedSTFT)
    paddedSTFT = np.reshape(paddedSTFT, paddedSTFT.shape + (1,))
    
    paddedSTFTs.append(paddedSTFT)
    

labels = np.array(labels)
hotlabels = keras.utils.to_categorical(labels, num_classes=2, dtype='float32')
samples = np.array(paddedSTFTs)
print(hotlabels)


# In[4]:


#delete


# In[ ]:




#layer 0: input
#labels[]
#paddedSTFTs[] (maxTime rows, 1025 cols) each

print('ready to start model')

# build model
model = Sequential()

print('made model')

#NOTE, CHANGED PADDING ON 2D CONVOLUTIONS from 'valid' to 'same'
#layer 1: 2D convolution between input and 256 filters with 1 row and 1025 cols
model.add(Conv2D(256, input_shape = [1084,1025,1], kernel_size = [1,1025], strides=(1, 1), padding='valid', data_format="channels_last", dilation_rate=(1, 1), activation=None, use_bias=False, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None))
#batch normalization- add in layer? don't understand parameters well
#model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None)
#reLU layer
model.add(ReLU(max_value=None, negative_slope=0.0, threshold=0.0))
print('layer 1 done')

##layer 2: 2D convolution between output of layer 1 and 256 filters with 3 rows and 256 cols
model.add(Conv2D(256, kernel_size = [3,256], strides=(2, 1), padding='same', data_format="channels_last", dilation_rate=(1, 1), activation=None, use_bias=False, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None))
##batch normalization- add in layer? don't understand parameters well
##model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None)
##reLU layer
model.add(ReLU(max_value=None, negative_slope=0.0, threshold=0.0))
print('layer 2 done')

#layer 3: 2D convolution between output of layer 2 and 256 filters with 3 rows and 256 cols
#model.add(Conv2D(256, kernel_size = [3,256], strides=(2, 1), padding='same', data_format="channels_last", dilation_rate=(1, 1), activation=None, use_bias=False, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None))
##batch normalization- add in layer? don't understand parameters well
##model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None)
##reLU layer
#model.add(ReLU(max_value=None, negative_slope=0.0, threshold=0.0))
#print('layer 3 done')

#layer 4: 2D convolution between output of layer 3 and 256 filters with 3 rows and 256 cols
#model.add(Conv2D(256, kernel_size = [3,256], strides=(2, 1), padding='same', data_format="channels_last", dilation_rate=(1, 1), activation=None, use_bias=False, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None))
##batch normalization- add in layer? don't understand parameters well
##model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None)
##reLU layer
#model.add(ReLU(max_value=None, negative_slope=0.0, threshold=0.0))
#print('layer 4 done')

#layer 5: Global max pooling
model.add(GlobalMaxPooling2D(data_format="channels_last"))
print('layer 5 done')

#layer 6: fully connected layer
model.add(Dense(2, activation='softmax', use_bias=False, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None))
print('layer 6 done')

#Compile model [COMPILE]
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'], loss_weights=None, sample_weight_mode=None, weighted_metrics=None, target_tensors=None)
print('compiled')

#Now let us train our model [FIT]
ES = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.01, patience=4, verbose=2, mode='auto', baseline=None, restore_best_weights=True)
#test
model.fit(x=samples, y=hotlabels, batch_size=1, epochs=6, verbose=2, callbacks=[ES], validation_split=0.2, validation_data=None, shuffle=True, class_weight=None, sample_weight=None, initial_epoch=0, steps_per_epoch=None, validation_steps=None)
#actual
#model.fit(x=samples, y=labels, batch_size=26, epochs=26, verbose=2, callbacks=ES, validation_split=0.2, validation_data=None, shuffle=True, class_weight=None, sample_weight=None, initial_epoch=0, steps_per_epoch=None, validation_steps=None, validation_freq=1)
print('ran fit')

#[EVALUATE]
#test
model.evaluate(x=samples, y=hotlabels, batch_size=1, verbose=1, sample_weight=None, steps=None)
#actual
#model.evaluate(x=samples, y=labels, batch_size=26, verbose=1, sample_weight=None, steps=None, callbacks=ES)

#[PREDICT (w/TestData I kept aside)]
#preprocess STFT data in TestData folder!
#see getLabels_stackedData3.py, but don't give labels
##model.predict(testSamples, batch_size=26, verbose=1, steps=None, callbacks=ES)

