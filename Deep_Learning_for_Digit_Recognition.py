#!/usr/bin/env python
# coding: utf-8

# LIBRARY               VERSION
# Keras                 2.3.1
# Keras-Applications    1.0.8
# Keras-Preprocessing   1.1.0
# scikit-learn          0.22.2.post1
# tensorboard           2.1.1
# tensorflow            2.1.0
# tensorflow-estimator  2.1.0
# tensorflow-gpu        2.1.0

# train.csv and test.csv too big for GitHub! go to https://www.kaggle.com/c/digit-recognizer
import os
os.chdir('/home/tim/Documents/Python/DigitRecognizer') # your working directory here

# import data science libraries

import numpy as np # linear algebra
import pandas as pd # data manipulation
import matplotlib.pyplot as plt # data visualization

# import deep learning and computer vision libraries

from keras import backend as K # Keras backend
from keras.models import Sequential # neural networks
from keras.optimizers import Adam, RMSprop # neural network optimizers
from sklearn.model_selection import train_test_split # model validation
from keras.layers import Dense, Dropout, Lambda, Flatten # neural networks
from keras.preprocessing.image import ImageDataGenerator # image preprocessing

# (for Jupyter notebooks) render figures in the notebook
get_ipython().run_line_magic('matplotlib', 'inline')

# read in training data
train = pd.read_csv("train.csv")
print(train.shape)
train.head()

# read in test data
test = pd.read_csv("test.csv")
print(test.shape)
test.head()

# format data
y_train = train.iloc[:,0].values.astype('int32') # declare Y variable (digit label; 10 classes)
X_train = (train.iloc[:,1:].values).astype('float32') # declare X variables (pixel values)
X_test = test.values.astype('float32') # same for test data

X_train # an array of floating-point pixel values

y_train # an array of digit labels

# convert training data to a format (num_images, img_rows, img_cols)

# Instead of coding each observation as a vector of 784 values (pixels),
# we code each observation as a 28 x 28 (= 784) dimensional array.

# This is an intuitive representation of the 28 pixels in each row
# and the 28 pixels in each column of each image.

# This is also closer to the format Keras needs for image recognition.

X_train = X_train.reshape(X_train.shape[0], 28, 28)

X_train # nested arrays of pixel values

# plot some images by generating colormaps (cmap) of pixel values

for i in range(6, 9):
    plt.subplot(330 + (i+1))
    plt.imshow(X_train[i], cmap = plt.get_cmap('gray'))
    plt.title(y_train[i]);

# It is necessary to specify the number of color channels for keras, even for the
# trivial case (black and white). So we add one more dimension to the training data
# and test data, a column hard-coded as 1.

X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_train.shape

X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
X_test.shape

# Standardizing each pixel by centering and scaling to unit variance is an
# important image preprocessing step for fully connected NNs. 
# Need to extract summary statistics (mean, sd) for this.

mean_px = X_train.mean().astype(np.float32)
std_px = X_train.std().astype(np.float32)

def standardize(x):
    return (x-mean_px)/std_px

# Keras also requires one-hot encoding of digit labels.

from keras.utils.np_utils import to_categorical
y_train = to_categorical(y_train)

# count the number of classes

num_classes = y_train.shape[1]
num_classes # 0, 1, 2, 3, 4, 5, 6, 7, 8, 9

# fix seed
# import tools for generating neural networks

seed = 43
np.random.seed(seed)
from keras.models import Sequential
from keras.layers.core import Lambda , Dense, Flatten, Dropout
from keras.callbacks import EarlyStopping
from keras.layers import BatchNormalization, Convolution2D, MaxPooling2D

# 'Sequential' groups a linear stack of layers into a keras model.
# 'Lambda' performs simple arithmetic (addition, exponentiation, etc).
# In the Lambda layer we standardize inputs and define the input dimensions.
# Format is rows, columns, color channel.
# 'Flatten' transforms the input into a 1D array.
# 'Dense' is fully connected (all neurons connect to all neurons in Lambda).
# In the Dense layer we define the output dimensions.
# 10 output dimensions since there are 10 digit labels.
# The 'softmax' activation function is used to assign class probabilities
# in the output layer.

# This type of neural network is a simple fully connected feedforward neural network
# sometimes referred to as a single-layer perceptron.
# It is only capable of representing linear separable functions or decisions.

# It can be considered as a kind of linear mutinomial logistic regression.

model = Sequential()
model.add(Lambda(standardize, input_shape = (28, 28, 1))) # tune parameters
model.add(Flatten())
model.add(Dense(10, activation = 'softmax'))
print("input shape ", model.input_shape)
print("output shape ", model.output_shape)

# We define the optimizer as RMSprop with learning rate 0.001.
# The loss function is categorical crossentropy, common for multiclass problems.
# We use accuracy to measure model performance.

# RMSprop:
# 1 of 8 optimizers available in Keras for model.compile()
# Maintain a moving (discounted) average of the square of gradients
# Divide the gradient by the root of this average

# import optimizer and define optimization scheme

from keras.optimizers import RMSprop
model.compile(optimizer = RMSprop(lr = 0.001),
             loss = 'categorical_crossentropy',
             metrics = ['accuracy'])

# import tools for passing images to Keras optimizer

from keras.preprocessing import image
gen = image.ImageDataGenerator()

# import tools for cross-validation train/test split

from sklearn.model_selection import train_test_split

# define cross-validation scheme (batch processing)

X = X_train
y = y_train
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.10, random_state = 42)
batches = gen.flow(X_train, y_train, batch_size = 64)
val_batches = gen.flow(X_val, y_val, batch_size = 64)

# By default, keras supports backpropagation during cross-validation.
# At each epoch, nodes in the neural network are tuned and re-weighted based
# on the error rate (i.e. loss) of the previous epoch (i.e. iteration).

history = model.fit_generator(generator = batches, steps_per_epoch = batches.n, epochs = 3, 
                    validation_data = val_batches, validation_steps = val_batches.n)

# print dictionary of CV results from the previous experiment

history_dict = history.history
history_dict.keys()

# visualize cross-validation performance over three epochs
# include loss values and validation loss values (less optimistic)

loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
epochs = range(1, len(loss_values) + 1)

plt.plot(epochs, loss_values, 'bo') # 'bo' is a graphical parameter for blue dots
plt.plot(epochs, val_loss_values, 'b+') # 'b+' is a graphical parameter for blue crosses
plt.xlabel('Epochs')
plt.ylabel('Loss')

plt.show()

plt.clf() # clear figure

# visualize cross-validation performance over three epochs
# include accuracy values and validation accuracy values (less optimistic)

acc_values = history_dict['accuracy'] # accuracy values
val_acc_values = history_dict['val_accuracy'] # validation accuracy values

plt.plot(epochs, acc_values, 'bo')
plt.plot(epochs, val_acc_values, 'b+')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.show()

# We add a hidden layer to the previous neural network to improve performance.
# Using one rule of thumb I defined the number of hidden neurons as 2/3 the size 
# of the input layer, minus the size of the output layer (2/3 * 28 * 28 - 10 = 512).
# The rectified linear unit (ReLU) function is a good activation function for this layer.
# ReLU is less vulnerable to the vanishing gradient problem than many alternatives.
# Adding a new layer improves abstraction at a computational cost.

# The optimizer is 'Adam', a stochastic gradient descent method
# based on adaptive estimation of first-order and order-order moments.
# Kingma et al. claim it is "computationally efficient, has little memory requirement, 
# invariant to diagonal rescaling of gradients, and is well suited for [large/complex problems]"

# A fully connected feedforward neural network with one hidden layer can approximate 
# any function that contains a continuous mapping from one finite space to another.

# It can be considered as a kind of non-linear mutinomial logistic regression.

def get_fc_model():
    model = Sequential([
        Lambda(standardize, input_shape = (28, 28, 1)),
        Flatten(),
        Dense(512, activation = 'relu'),
        Dense(10, activation = 'softmax')
        ])
    model.compile(optimizer = 'Adam', loss = 'categorical_crossentropy',
                 metrics = ['accuracy'])
    return model

fc = get_fc_model()

fc.optimizer.learning_rate = 0.01 # set learning rate

# cross-validation
history = fc.fit_generator(generator = batches,
                          steps_per_epoch = batches.n,
                          epochs = 1,
                          validation_data = val_batches,
                          validation_steps = val_batches.n)

# Convolutional networks excel at computer vision.
# CNNs are regularized versions of fully connected neural networks, which are
# more sensitive to overfitting. 
# CNNs are also less vulnerable to the curse of dimensionality compared to FC networks.
# They also require less image pre-processing, since they excel at feature abstraction.
# A CNN can have many architectures. 
# This one involves Max pooling (1990) and two fully connected layers.

# A deep NN can represent an arbitrary decision boundary to arbitrary accuracy 
# with rational activation functions and can approximate any smooth mapping to any accuracy.
# Deep NNs also support automatic feature engineering by learning complex representations.

from keras.layers import Convolution2D, MaxPooling2D # import tools for CNNs

def get_cnn_model():
    model = Sequential([
            Lambda(standardize, input_shape = (28, 28, 1)),
            Convolution2D(32, (3, 3), activation = 'relu'),
            Convolution2D(32, (3, 3), activation = 'relu'),
            MaxPooling2D(),
            Convolution2D(64, (3, 3), activation = 'relu'),
            Convolution2D(64, (3, 3), activation = 'relu'),
            MaxPooling2D(),
            Flatten(),
            Dense(512, activation = 'relu'),
            Dense(10, activation = 'softmax')
            ])
    model.compile(Adam(), loss = 'categorical_crossentropy',
                 metrics = ['accuracy'])
    return model

model = get_cnn_model()

model.optimizer.learning_rate = 0.01

# cross-validation (computationally intensive: ~ 1-2 hours)

history = model.fit_generator(generator = batches,
                             steps_per_epoch = batches.n,
                             epochs = 1,
                             validation_data = val_batches,
                             validation_steps = val_batches.n)

# One method to improve accuracy is data augmentation,
# which reduces generalization error and overfitting.
# For images, methods include cropping, rotating, scaling, 
# translating, flipping, and adding Gaussian noise.
# Transformations should be reasonable depending on the task.

gen = ImageDataGenerator(rotation_range = 8,            # rotation
                         width_shift_range = 0.08,      # horizontal scaling
                         shear_range = 0.3,             # cropping
                         height_shift_range = 0.08,     # vertical scaling
                         zoom_range = 0.08)             # zoom

batches = gen.flow(X_train, y_train, batch_size = 64)
val_batches = gen.flow(X_val, y_val, batch_size = 64)

model.optimizer.learning_rate = 0.001

# cross-validation (computationally intensive: ~ 1-2 hours)

history = model.fit_generator(generator = batches, 
                             steps_per_epoch = batches.n,
                             epochs = 1,
                             validation_data = val_batches,
                             validation_steps = val_batches.n)

# https://github.com/keras-team/keras/issues/13684
# fix issue where a line of keras code in the backend was deprecated

import tensorflow as tf
import keras.backend.tensorflow_backend as tfback
print("tf.__version__ is", tf.__version__)
print("tf.keras.__version__ is:", tf.keras.__version__)

def _get_available_gpus():
    """Get a list of available gpu devices (formatted as strings).

    # Returns
        A list of available GPU devices.
    """
    #global _LOCAL_DEVICES
    if tfback._LOCAL_DEVICES is None:
        devices = tf.config.list_logical_devices()
        tfback._LOCAL_DEVICES = [x.name for x in devices]
    return [x for x in tfback._LOCAL_DEVICES if 'device:gpu' in x.lower()]

tfback._get_available_gpus = _get_available_gpus

# Batch normalization (2015) is a way to normalize the input layer by re-centering and re-scaling.
# It has been shown to improve speed, performance, stability, and accuracy of hyperparameters.
# The exact mechanism is debated. It is thought to bring bring back the benefits of normalization 
# at each layer, perhaps by smoothing and reducing internal covariance shift.
# By integrating batch normalization with the previous model, we should obtain the best model yet.
# Accuracy can be further improved by increasing the number of epochs.

from keras.layers.normalization import BatchNormalization

def get_bn_model():
    model = Sequential([
                    Lambda(standardize, input_shape = (28, 28, 1)),
                    Convolution2D(32, (3, 3), activation = 'relu'),
                    BatchNormalization(axis = 1),
                    Convolution2D(32, (3, 3), activation = 'relu'),
                    MaxPooling2D(),
                    BatchNormalization(axis = 1),
                    Convolution2D(64, (3, 3), activation = 'relu'),
                    BatchNormalization(axis = 1),
                    Convolution2D(64, (3, 3), activation = 'relu'),
                    MaxPooling2D(),
                    Flatten(),
                    BatchNormalization(),
                    Dense(512, activation = 'relu'),
                    BatchNormalization(),
                    Dense(10, activation = 'softmax')              
                    ])
    model.compile(Adam(), loss = 'categorical_crossentropy', metrics = ['accuracy'])
    return model

model = get_bn_model()

# backpropagation (computationally intensive: ~ 6-7 hours calculation)

model.optimizer.learning_rate = 0.01

gen = image.ImageDataGenerator()

batches = gen.flow(X, y, batch_size = 64)

history = model.fit_generator(generator = batches, 
                              steps_per_epoch = batches.n,
                              epochs = 3)

# save predictions

predictions = model.predict_classes(X_test, verbose = 0)

submissions = pd.DataFrame({"ImageId": list(range(1, len(predictions) + 1)),
                            "Label": predictions})

submissions.to_csv("DigitRecognizerSubmissions.csv", index = False, header = True)

# Final accuracy: 0.99271
# Leaderboard: 793/3206 (Top 25%)
