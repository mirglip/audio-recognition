"""A python script to train and save a spectrogram-based CNN on the Free Spoken Digit Dataset"""

import os
import glob
import scipy.io.wavfile
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
from keras import backend as K
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, BatchNormalization, ReLU, Flatten
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


### Data extraction and manipulation ###
CURRENT_DIR = os.getcwd()
#directory of recordings in the same directory as the script
RECORDINGS_DIR = os.path.join(CURRENT_DIR, "recordings") 

max_length = 5000
#retrieve the path of each sample
recordings_name_list = glob.glob(os.path.join(RECORDINGS_DIR, '*.wav')) 
#scipy.io.wavfile.read(file_path)[1] allows to retrieve data in the wav file to feed in the CNN
recordings_data_list = np.array([scipy.io.wavfile.read(file_path)[1][:max_length] for file_path in recordings_name_list]) 
#retrieve labels contained in the name of the wav file
recordings_labels_list = np.array([os.path.basename(file_path)[0] for file_path in recordings_name_list], dtype = 'int32')
#as each sample has a different length, they must be padded to the max length of sample
recordings_data_pad_list = pad_sequences(recordings_data_list, padding = 'post')

#shows 10 spectrograms, one for each label
j = 1
for i in range(0, 1500, 150):
    example = recordings_data_pad_list[i]
    
    plt.figure(10)
    plt.subplot(2, 5, j)
    plt.plot(np.arange(len(example))/8000, example)
    
    plt.figure(100)
    plt.subplot(2, 5, j)
    f, t, Sxx = signal.spectrogram(example, 8000)
    plt.pcolormesh(t, f, Sxx)
    plt.show()
    j+=1


### Training and test sets creation ###
recordings_spec_list = np.array([signal.spectrogram(sample, 8000)[-1]/np.max(signal.spectrogram(sample, 8000)[-1]) for sample in recordings_data_pad_list])
recordings_spec_list = recordings_spec_list.reshape((recordings_spec_list.shape[0],
                             recordings_spec_list.shape[1],
                             recordings_spec_list.shape[2],
                             1))

test_set_size = 150
batch_size = 135
data_train, data_test, labels_train, labels_test = train_test_split(recordings_spec_list, 
                                                                    recordings_labels_list, 
                                                                    test_size = test_set_size, 
                                                                    stratify = recordings_labels_list)
n_batches = len(data_train)//batch_size


### Model ###
K.clear_session()
model = Sequential()

#first convolution layer
model.add(Conv2D(32, kernel_size = (3, 3),input_shape = (129, 22, 1)))
model.add(BatchNormalization())
model.add(ReLU())
model.add(MaxPooling2D(pool_size = (2, 2)))

#second convolution layer
model.add(Conv2D(64, (3, 3)))
model.add(BatchNormalization())
model.add(ReLU())
model.add(MaxPooling2D(pool_size = (2, 2)))

#dense layer
model.add(Flatten())
model.add(Dropout(0.25))
model.add(Dense(128, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation = 'softmax'))


### Model training and evaluation ###
model.compile(optimizer = 'adam',
              loss = 'sparse_categorical_crossentropy',
              metrics = ['accuracy'])

model.fit(data_train, 
          labels_train, 
          epochs = 10, 
          batch_size = batch_size, 
          validation_data = (data_test, labels_test))

print("Confusion Matrix :\n", confusion_matrix(labels_test, model.predict_classes(data_test)))

model.save('spectroCNN.h5')