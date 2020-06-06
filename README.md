# Image-retrieval-using-CNN
Image retrieval on Oxford-IIIT Pet Dataset

## Description
In this algorithm we retrieved images from Oxford-IIIT Pet Dataset using a convolutional neural network (CNN) for feature extraction.

## Data
The dataset can be found here: https://www.robots.ox.ac.uk/~vgg/data/pets/.
It contains annotated images from 37 breeds of cats and dogs.

## Preprocessing
We resized all the images to 224x224, removed any grayscale image and converted the images to [0,1] range.

## Model
We used the VGG-16 CNN with pretrained weights on ImageNet but we did not include the fully connected (FC) block. After several tries to find the optimal network structure, we replaced the FC block with a global average pooling layer, followed by a FC layer, followed by a dropout layer, followed by a FC layer, followed by a dropout layer and finally the 37 neurons FC layer for the prediction. We trained these layers and the last convolutional block to get more accurate results. At the end we extracted the feature vectors of images from the last FC layer and we used the k-nearest neighbors algorithms to retrieve the most similar images.
