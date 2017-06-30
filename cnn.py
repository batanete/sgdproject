"""
Based on the tflearn example located here:
https://github.com/tflearn/tflearn/blob/master/examples/images/convnet_cifar10.py
"""
from __future__ import division, print_function, absolute_import

# Import tflearn and some helpers
import tflearn
from tflearn.data_utils import shuffle
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.data_utils import image_preloader

from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
from imutils import paths
import os
import pickle
import numpy as np

PATH='food-101/images'

nclasses=len([i for i in os.listdir(PATH)])

print(nclasses)

#load dataset
images,labels=image_preloader(PATH, image_shape=(32, 32),   mode='folder', categorical_labels=True,   normalize=True,filter_channel=True)
print('loaded dataset',len(labels))


images=list(images)
labels=list(labels)



labelstrain=[]
labelsclassify=[]
imagestrain=[]
imagesclassify=[]

# split the data into 90% training data and 10% test data
for i in range(len(labels)):
    if i%100>10:
        labelstrain.append(labels[i])
        imagestrain.append(images[i])
    else:
        labelsclassify.append(labels[i])
        imagesclassify.append(images[i])

i=0

X=imagestrain
Y=labelstrain
X_test = imagesclassify
Y_test = labelsclassify

"""
with open('x.pickle','wb') as f:
    pickle.dump(X,f)
with open('y.pickle','wb') as f:
    pickle.dump(Y,f)
with open('xt.pickle','wb') as f:
    pickle.dump(X_test,f)
with open('yt.pickle','wb') as f:
    pickle.dump(Y_test,f)
print('saved stuff')"""

# Make sure the data is normalized
img_prep = ImagePreprocessing()
img_prep.add_featurewise_zero_center()
img_prep.add_featurewise_stdnorm()

# Create extra synthetic training data by flipping, rotating and blurring the
# images on our data set.
img_aug = ImageAugmentation()
img_aug.add_random_flip_leftright()
img_aug.add_random_rotation(max_angle=25.)
img_aug.add_random_blur(sigma_max=3.)

# Define our network architecture:

# Input is a 32x32 image with 3 color channels (red, green and blue)
network = input_data(shape=[None, 32, 32,3],
                     data_preprocessing=img_prep,
                     data_augmentation=img_aug)

# Step 1: Convolution
network = conv_2d(network, 32, 3, activation='relu')

# Step 2: Max pooling
network = max_pool_2d(network, 2)

# Step 3: Convolution again
network = conv_2d(network, 64, 3, activation='relu')

# Step 4: Convolution yet again
network = conv_2d(network, 64, 3, activation='relu')

# Step 5: Max pooling again
network = max_pool_2d(network, 2)

# Step 6: Fully-connected 512 node neural network
network = fully_connected(network, 512, activation='relu')

# Step 7: Dropout - throw away some data randomly during training to prevent over-fitting
network = dropout(network, 0.5)

# Step 8: Fully-connected neural network with 101 outputs, for each food type
network = fully_connected(network, nclasses, activation='softmax')

# Tell tflearn how we want to train the network
network = regression(network, optimizer='adam',
                     loss='categorical_crossentropy',
                     learning_rate=0.001)

# Wrap the network in a model object
model = tflearn.DNN(network, tensorboard_verbose=0)


model.load('food-classifier.tfl')
print('model loaded')

# Train it! We'll do 30 training passes and monitor it as it goes.
model.fit(X, Y, n_epoch=100, shuffle=True, validation_set=(X_test, Y_test),
          show_metric=True,
          snapshot_epoch=True,
          run_id='food-classifier')



# Save model when training is complete to a file
model.save("food-classifier.tfl")
print("Network trained and saved as food-classifier.tfl!")
