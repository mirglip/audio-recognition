# Audio recognition with raw waveforms and spectrograms

The goal of this project is to use Deep Learning for audio recognition with 2 different networks.

## Prerequisites

The scripts were created through Spyder IDE with **Python 3** in a conda environment. Some packages are needed and should be installed if not:
 - matplotlib: for data visualization.
 - scipy: processing the wav files and computing the spectrograms.
 - scikit-learn: for prepocessing the dataset and the labels.
 - keras: main package, to create the neural networks and train them.

## Dataset

The dataset on which the neural networks have been constructed is the **Free Spoken Digit Dataset (FSDD)**.  To sum up, it contains 1 500 audio samples divided equally in 10 classes: the digit ranged from 0 to 9.

For more information, the GitHub page of the Dataset can be found [here](https://github.com/Jakobovski/free-spoken-digit-dataset).

All samples are located in the directory `recordings`: it must be in the same directory as the neural network's scripts.
## sampleCNN

The first ConvNet was inspired by this paper:
[Sample-Level Deep Convolutional Neural Networks For Music Auto-Tagging Using Raw Waveforms](https://arxiv.org/pdf/1703.01789.pdf)
The corresponding script is `sampleCNNtraining.py`.

### Input
For each audio sample, the waveform has been extracted and fed as input for the neural network. As each sample has a variable size, zero-padding is used to have the same fixed size. 

### Architecture
The network uses 1 strided convolutional layer and thereafter a succession of 3 intermediate convolutional layers (max pooling is placed after the 1st and the 2nd intermediate ones).  For each conv layer, batch normalization and ReLU are applied. 
The output of the last one is flatted and dropout is applied before going through a dense layer with the sigmoid's activation function. 
The final output is a vector of 10 values, one for each class.

For the training strategy, the cost function is the cross entropy and the optimizer is Adam.

### Test
To train a model, simply run `sampleCNNtraining.py`. 
During training, some useful information such as loss and accuracy are displayed and at the end, the confusion matrix with the test set is computed.
Finally, the model is saved in an external file named _**sampleCNN.h5**_.

## spectrogram-based CNN

The second ConvNet is a vanilla implementation but based on spectrograms.
The corresponding script is `specCNNtraining.py`.

### Input
For each audio sample, its spectrogram has been computed using scipy and more precisely the function `scipy.signal.spectrogram`. This time, the samples are still zero-padded but not to the maximum length in the dataset. The length has been set so that more-or-less 5% of the whole dataset is longer than that.
The spectrograms are fed to the network as how images would be.

### Architecture
The network uses 2 convolutional layers.  For each conv layer, batch normalization, ReLU and max pooling are applied. 
The output of the last one is flatted and dropout is applied before going through a dense layer with the softmax's activation function. 
The final output is a vector of 10 values, each corresponding to the probability to belong to the corresponding class.

For the training strategy, the cost function is still the cross entropy and the optimizer is still Adam.

### Test
To train a model, simply run `spectroCNNtraining.py`. 
To give an idea, 10 spectrograms (one for each class) are displayed before training with the corresponding waveform in time domain.
During training, some useful information such as loss and accuracy are displayed and at the end, the confusion matrix with the test set is computed.
Finally, the model is saved in an external file named _**spectroCNN.h5**_.

## Results
As both networks have been trained for the same task, it's easier to compare them.

For both, the training accuracy reaches more than 90%, if not 99%.
On average, the __sampleCNN__ achieve a test accuracy of 60% while the __spectrogram-based CNN__ performs better with an accuracy of 70%.
However, these metrics imply an overfitting situation : a gap of respectively 30% and 20% are not normal. Moreover, as iterations go on, the test accuracy is instable.

The first reason that comes to mind to explain this overfit is the complexity of both networks which is too high.
Another is the possibility of too few epochs for the training to stabilize. Computational power and time are limited but maybe with more of these resources, both networks could converge.

To limit this phenomenon, more dropout could be applied. Other regularizations techniques as L1 or L2 could be used. 
But the problem can lie somewhere else. 150 samples for each label is too few: the model struggles to find the patterns that could help it to generalize. The solution to this issue is using data augmentation. For the raw audio signals, for example, some noise could be added. With this technique, more spectrograms can be also computed.

## Conclusion
Audio recognition means often time series data. In this case, the recurrent neural networks are the ideal candidate.
However, CNN can also be used. The traditional approach is to compute spectrograms which are fed to the network. Hovewer, with 1D convolutional layers, they can take the raw signals as inputs and process time series data. Mixing several parts that have a priori no links can often result in unexpected scientific insights such as ConvNets for audio data.
