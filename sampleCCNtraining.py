"""A python script to train and save a sampleCNN on the Free Spoken Digit Dataset"""

import os
import glob
import scipy.io.wavfile
import numpy as np
from keras import backend as K
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv1D, MaxPool1D, BatchNormalization, ReLU, Flatten
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


### Data extraction and manipulation ###
CURRENT_DIR = os.getcwd()
#directory of recordings in the same directory as the script
RECORDINGS_DIR = os.path.join(CURRENT_DIR, "recordings") 

#retrieve the path of each sample
recordings_name_list = glob.glob(os.path.join(RECORDINGS_DIR, '*.wav')) 
#scipy.io.wavfile.read(file_path)[1] allows to retrieve data in the wav file to feed in the CNN
recordings_data_list = np.array([scipy.io.wavfile.read(file_path)[1] for file_path in recordings_name_list]) 
#retrieve labels contained in the name of the wav file
recordings_labels_list = np.array([os.path.basename(file_path)[0] for file_path in recordings_name_list], dtype = 'int32')
#as each sample has a different length, they must be padded to the max length of sample
recordings_data_pad_list = pad_sequences(recordings_data_list, padding = 'post')
max_length = recordings_data_pad_list.shape[-1]


### Training and test sets creation ###
test_set_size = 150
batch_size = 135
data_train, data_test, labels_train, labels_test = train_test_split(recordings_data_pad_list, 
                                                                    recordings_labels_list, 
                                                                    test_size = test_set_size, 
                                                                    stratify = recordings_labels_list)
n_batches = len(data_train)//batch_size
#must be reshaped for the CNN as (number of samples, max_length, channels)
data_train = data_train.reshape(data_train.shape[0], data_train.shape[1], 1)
data_test = data_test.reshape(data_test.shape[0], data_test.shape[1], 1) 


### Model ###
K.clear_session()
model = Sequential()

#strided convolution layer
model.add(Conv1D(filters = 16, kernel_size = 256, strides = 256, batch_size = None, input_shape = (max_length, 1)))
model.add(BatchNormalization())
model.add(ReLU())

#first convolution layer
model.add(Conv1D(filters = 16, kernel_size = 2, strides = 1, padding = 'same'))
model.add(BatchNormalization())
model.add(ReLU())
model.add(MaxPool1D(2, strides = 2))

#second convolution layer
model.add(Conv1D(filters = 32, kernel_size = 2, strides = 1, padding = 'same'))
model.add(BatchNormalization())
model.add(ReLU())
model.add(MaxPool1D(2, strides = 2))

#third convolution layer
model.add(Conv1D(filters = 64, kernel_size = 1, strides = 1, padding = 'same'))
model.add(BatchNormalization())
model.add(ReLU())

#dense layer
model.add(Flatten())
model.add(Dropout(0.6))
model.add(Dense(10, activation = 'sigmoid'))


### Model training and evaluation ###
model.compile(optimizer = 'adam',
              loss = 'sparse_categorical_crossentropy',
              metrics = ['accuracy'])

model.fit(data_train, 
          labels_train, 
          epochs = 500, 
          batch_size = batch_size, 
          validation_data = (data_test, labels_test))

print("Confusion Matrix :\n", confusion_matrix(labels_test, model.predict_classes(data_test)))

model.save('sampleCNN.h5')