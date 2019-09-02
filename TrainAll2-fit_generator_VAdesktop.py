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
from keras.optimizers import SGD
print('finished imports!')


# In[2]:


#?
from keras.backend.tensorflow_backend import set_session
config = tensorflow.ConfigProto()
config.intra_op_parallelism_threads = 1
print(config.intra_op_parallelism_threads)
set_session(tensorflow.Session(config=config))

#before generator, loop through to find maxTime, Don't need to save anything else though

#change to the directory where list of STFT files is 
print(os.getcwd())
os.chdir('/home/ec2-user/SageMaker')
print(os.getcwd())

maxTime = 0
print('made variable')

#open file (lists all train .npy STFT file names) 
file = open('trainListALL.txt', 'r') #note, ran with 169 in TopList, could not handle all 676
data = file.read().split('\n')
file.close()
print('got data files in a list')
#print(data)

os.chdir('/home/ec2-user/SageMaker/aws_pilotwhales2')
print('changed to aws_pilotwhales2 folder')

for array in data:
    #find cols (number of time steps) of each STFT and save longest one
    curSTFT = np.load(array)
    rows, cols = curSTFT.shape
    if (cols>maxTime):
        maxTime = cols
    if (rows!=1025):
        print('Error, not 1025 STFT coefficients') 
print(maxTime) 
print('find max loop done')


# In[4]:


def data_generator(path, file, bs, maxTime, mode='train'):
    # open the text file for reading
    os.chdir(path)
    f = open(file, 'r')
    while True: 
        
        # initialize our batches of images and labels
        samples = []
        labels = []
    
        # keep looping until we reach our batch size
        while len(samples) < bs:
            # attempt to read the next line of the text file
            line = f.readline()
            ##print(line)
    
            # check to see if the line is empty, indicating we have
            # reached the end of the file
            if line == "":
                # reset the file pointer to the beginning of the file
                # and re-read the line
                f.seek(0)
                line = f.readline()
                # if we are evaluating we should now break from our
				# loop to ensure we don't continue to fill up the
				# batch from samples at the beginning of the file
                if mode == "eval":
                    break
    
            # extract the label and construct the sample
            #loop through to find out if file is a buzz or minibuzz, and add label accordingly 
            nameParse = line.split('_',-1)
            ##print(nameParse)
            typeParse = nameParse[4].split('u', -1)
            ##print(typeParse)
            if(typeParse[0] == 'b'):
                #buzz = 1 
                labels.append(1)
            elif(typeParse[0] == 'minib'):
                #minibuzz = 0
                labels.append(0)
            else: 
                print('Error, not a buzz or minibuzz!')
    
            #switch to aws_pilotwhales2 folder
            os.chdir('/home/ec2-user/SageMaker/aws_pilotwhales2')
            #print('changed to aws_pilotwhales2 folder')
    
            lineParse = line.split("\n",-1)
            ##print(lineParse)
            curSTFT = np.load(lineParse[0])
            rows, cols = curSTFT.shape
    
            zeroPad = np.zeros((rows,maxTime-cols))
            paddedSTFT = np.append(curSTFT, zeroPad, axis = 1)
    
            paddedSTFT = np.transpose(paddedSTFT)
            paddedSTFT = np.reshape(paddedSTFT, paddedSTFT.shape + (1,))
    
            samples.append(paddedSTFT)
    
        labels = np.array(labels)
        #one hot encoding for labels
        hotlabels = keras.utils.to_categorical(labels, num_classes=2, dtype='float32')
        samples = np.array(samples)
        #print('hotlabels and samples in np arrays')

        # yield the batch to the calling function
        yield (samples, hotlabels)
        
# initialize both the training and testing image generators
trainGen = data_generator('/home/ec2-user/SageMaker', 'trainList.txt', 13, maxTime, mode = 'train')
print('generated training set')
validGen = data_generator('/home/ec2-user/SageMaker', 'validList.txt', 13, maxTime, mode = 'train')
print('generated vaildation set')


# In[5]:


#layer 0: input
#labels[]
#paddedSTFTs[] (maxTime rows, 1025 cols) each

print('ready to start model')

# build model
model = Sequential()

print('made model')

#NOTE, CHANGED PADDING ON 2D CONVOLUTIONS from 'valid'=no paddinng to 'same'=padding so input and output are same dimensions

#layer 1: 2D convolution between input and 256 filters with 1 row and 1025 cols
model.add(Conv2D(256, input_shape = [maxTime,1025,1], kernel_size = [1,1025], strides=(1, 1), padding='valid', data_format="channels_last", dilation_rate=(1, 1), activation=None, use_bias=False, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None))
#batch normalization- add in layer? don't understand parameters well
#model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None)
#reLU layer
model.add(ReLU(max_value=None, negative_slope=0.0, threshold=0.0))
print('layer 1 done')

##layer 2: 2D convolution between output of layer 1 and 256 filters with 3 rows and 256 cols
model.add(Conv2D(256, kernel_size = [3,1], strides=(2, 1), padding='same', data_format="channels_last", dilation_rate=(1, 1), activation=None, use_bias=False, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None))
##batch normalization- add in layer? don't understand parameters well
##model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None)
##reLU layer
model.add(ReLU(max_value=None, negative_slope=0.0, threshold=0.0))
print('layer 2 done')

#layer 3: 2D convolution between output of layer 2 and 256 filters with 3 rows and 256 cols
model.add(Conv2D(256, kernel_size = [3,1], strides=(2, 1), padding='same', data_format="channels_last", dilation_rate=(1, 1), activation=None, use_bias=False, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None))
##batch normalization- add in layer? don't understand parameters well
##model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None)
##reLU layer
model.add(ReLU(max_value=None, negative_slope=0.0, threshold=0.0))
print('layer 3 done')

#layer 4: 2D convolution between output of layer 3 and 256 filters with 3 rows and 256 cols
model.add(Conv2D(256, kernel_size = [3,1], strides=(2, 1), padding='same', data_format="channels_last", dilation_rate=(1, 1), activation=None, use_bias=False, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None))
##batch normalization- add in layer? don't understand parameters well
##model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None)
##reLU layer
model.add(ReLU(max_value=None, negative_slope=0.0, threshold=0.0))
print('layer 4 done')

#layer 5: Global max pooling
model.add(GlobalMaxPooling2D(data_format="channels_last"))
print('layer 5 done')

#layer 6: fully connected layer
model.add(Dense(2, activation='softmax', use_bias=False, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None))
print('layer 6 done')

#Compile model [COMPILE]
#OLD COMPILE (for fit, not fit_generator)
#model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'], loss_weights=None, sample_weight_mode=None, weighted_metrics=None, target_tensors=None)
opt = SGD(lr=0.02)
model.compile(loss = "binary_crossentropy", optimizer = opt, metrics=["accuracy"])
print('compiled')

print(model.summary())


# In[9]:


#Now let us train our model [FIT]
ES = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.01, patience=4, verbose=2, mode='auto', baseline=None, restore_best_weights=True)

#with fit_generator
print("[INFO] training w/ generator...")
model.fit_generator(trainGen, steps_per_epoch=42, epochs=5, verbose=2, callbacks=[ES], validation_data=validGen, validation_steps=10, class_weight=None, max_queue_size=10, workers=1, use_multiprocessing=False, shuffle=True, initial_epoch=0)
print('ran fit_generator')

#with fit
#model.fit(x=samples, y=hotlabels, batch_size=13, epochs=5, verbose=2, callbacks=[ES], validation_split=0.2, validation_data=None, shuffle=True, class_weight=None, sample_weight=None, initial_epoch=0, steps_per_epoch=None, validation_steps=None)
#print('ran fit')


# In[11]:


#[EVALUATE]
#do for test data, not training data! 
#preprocess STFT data in TestData folder!
#see getLabels_stackedData3.py, and give labels
#results = model.evaluate(x=testSamples, y=testHotlabels, batch_size=26, verbose=1, sample_weight=None, steps=None)
#results = model.evaluate(x=samples, y=hotlabels, batch_size=26, verbose=1, sample_weight=None, steps=None)
#print(results)
#print(model.metrics_names)


# In[12]:


#[PREDICT] (w/TestData I kept aside as well)
#preprocess STFT data in TestData folder!
#see getLabels_stackedData3.py, but don't give labels
#hotlabels_pred = model.predict(samples, batch_size=26, verbose=1, steps=None)


# In[13]:


#see if prediction results are right (compare hot labels I generate with what the model guesses)
#print((np.max(abs(hotlabels-hotlabels_pred),axis=1)>.5))

