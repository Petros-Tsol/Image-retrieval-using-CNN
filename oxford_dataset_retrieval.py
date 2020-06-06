from keras.utils import get_file, to_categorical
from keras.preprocessing import image
from keras.applications.vgg16 import VGG16, decode_predictions, preprocess_input
from keras.layers import Flatten, Dense, Dropout, GlobalAveragePooling2D, GlobalMaxPooling2D
from keras.models import Model
from keras.optimizers import Adam

from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split

import skimage
from skimage.io import imsave, imread, imshow
from skimage.transform import resize, rotate
from skimage.util import random_noise
from skimage.color import rgba2rgb
import matplotlib.pyplot as plt

import numpy as np
import os ,re, io, sys
import random


#download the dataset
path = get_file('images.tar.gz', origin='http://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz', extract=True, cache_dir='/content')

#remove .mat files
os.chdir('/content/datasets/images')
os.makedirs('../rescaled_images')
os.chdir('/content/datasets/images')
!rm *.mat


#rescale images to (224,224) size
#index = 0
for filename in sorted(os.listdir()):
  get_s = re.search("(_\d.+)",filename)
  if get_s is not None:
    	
    try:
      img = imread(filename)
      img = (rgba2rgb(img)*255).astype(np.uint8)
      rescaled_image = resize(img,(224,224)).astype(np.uint8)
      imsave('/content/datasets/rescaled_images/'+filename+'.png',rescaled_image);
    except:
      img = imread(filename)
      rescaled_image = resize(img,(224,224))
      imsave('/content/datasets/rescaled_images/'+filename+'.png',rescaled_image);

    #index += 1
    #print(index)

os.chdir('/content/datasets/rescaled_images')

#remove black and white images
index = 0
for filename in sorted(os.listdir()):
  get_s = re.search("(_\d.+)",filename)
  if get_s is not None:
    I=imread('/content/datasets/rescaled_images/'+filename)
    if I.shape == (224,224):
      #print(filename)
      os.remove('/content/datasets/rescaled_images/'+filename)
      continue
    index += 1




#copy images to an array
os.chdir('/content/datasets/rescaled_images')
data = np.empty((index, 224, 224, 3), dtype='uint8')
i = 0
for filename in sorted(os.listdir()):
  get_s = re.search("(_\d.+)",filename)
  if get_s is not None:
    I=imread('/content/datasets/rescaled_images/'+filename)
    data[i,:,:,:] = I
    i += 1

#labels of images
category_to_int = {
    'abyssinian': 1,
    'american_bulldog': 2,
    'american_pit_bull_terrier': 3,
    'basset_hound': 4,
    'beagle': 5,
    'bengal': 6,
    'birman': 7,
    'bombay': 8,
    'boxer': 9,
    'british_shorthair': 10,
    'chihuahua': 11,
    'egyptian_mau': 12,
    'english_cocker_spaniel': 13,
    'english_setter': 14,
    'german_shorthaired': 15,
    'great_pyrenees': 16,
    'havanese': 17,
    'japanese_chin': 18,
    'keeshond': 19,
    'leonberger': 20,
    'maine_coon': 21,
    'miniature_pinscher': 22,
    'newfoundland': 23,
    'persian': 24,
    'pomeranian': 25,
    'pug': 26,
    'ragdoll': 27,
    'russian_blue': 28,
    'saint_bernard': 29,
    'samoyed': 30,
    'scottish_terrier': 31,
    'shiba_inu': 32,
    'siamese': 33,
    'sphynx': 34,
    'staffordshire_bull_terrier': 35,
    'wheaten_terrier': 36,
    'yorkshire_terrier': 37
}

#label the images
os.chdir('/content/datasets/rescaled_images')
labels = np.empty((index,2))
i=0
for filename in sorted(os.listdir()):
  get_s = re.search("(_\d.+)",filename)
  if get_s is not None:
    pos = get_s.span()
    breed = filename[0:pos[0]]

    first_letter = re.match("^[a-zA-Z]",filename)

    labels[i,0] = category_to_int[breed.lower()] - 1
    if first_letter.group(0).isupper() == 1:
      labels[i,1] = 1
    else:
      labels[i,1] = 2
      
    i+=1


#split the dataset to 85% train data - 15% test data
train_data, test_data, train_labels, test_labels = train_test_split(data,labels,test_size=0.15,shuffle=True)

#clear unused arrays to save memory
data = None
labels = None
#import gc
#gc.collect()

#one hot encoding of breed label
train_labels_ohe = to_categorical(train_labels[:,0])
test_labels_ohe = to_categorical(test_labels[:,0])

#make data in range [0,1]
data_preprocess = image.ImageDataGenerator(
  rescale=1./255
)

data_preprocess.fit(train_data)
data_preprocess.fit(test_data)


#load the vgg16 model with imagenet weights and do not include the FC layers on the end
vgg16_model = VGG16(weights = 'imagenet', include_top = False, input_shape=(224,224,3))

for i, layer in enumerate(vgg16_model.layers):
    print(i, layer.name, layer.output_shape)

x = vgg16_model.output
####some of the models we try along as with the validation accuracy


####1st model####### val accuracy: 62%
#x = Flatten(name='flatten_1')(x)
#x = Dense(512, activation='relu', name='dense_1')(x)
#x = Dense(512, activation='relu', name='dense_2')(x)
#x = Dense(37, activation='softmax', name = 'predictions')(x)
####1st model######

####2nd model###### val accuracy: 26%
#x = Flatten(name='flatten_1')(x)
#x = Dense(2048, activation='relu', name='dense_1')(x)
#x = Dense(37, activation='softmax', name = 'predictions')(x)
####2nd model####

####3rd model####### val accuracy: 65%
#x = Flatten(name='flatten_1')(x)
#x = Dense(512, activation='relu', name='dense_1')(x)
#x = Dropout(0.2)(x)
#x = Dense(512, activation='relu', name='dense_2')(x)
#x = Dropout(0.2)(x)
#x = Dense(37, activation='softmax', name = 'predictions')(x)
####3rd model######

####4th method####### val accuracy: 74%
#x = GlobalAveragePooling2D(name='gap2d')(x)
#x = Dense(512, activation='relu', name='dense_1')(x)
#x = Dense(512, activation='relu', name='dense_2')(x)
#x = Dense(37, activation='softmax', name = 'predictions')(x)
####4th method######

####5th model####### val accuracy:  74%
#x = GlobalAveragePooling2D(name='gap2d')(x)
#x = Dense(512, activation='relu', name='dense_1')(x)
#x = Dropout(0.2)(x)
#x = Dense(512, activation='relu', name='dense_2')(x)
#x = Dropout(0.2)(x)
#x = Dense(37, activation='softmax', name = 'predictions')(x)
####5th model######

####6th model####### val accuracy: 43%
#x = GlobalMaxPooling2D(name='gmp2d')(x)
#x = Dense(512, activation='relu', name='dense_1')(x)
#x = Dense(512, activation='relu', name='dense_2')(x)
#x = Dense(37, activation='softmax', name = 'predictions')(x)
####6th model######


####7th model####### val accuracy: 80% ---- 86% after training the last convolution block
x = GlobalAveragePooling2D(name='gap2d')(x)
x = Dense(1024, activation='relu', name='dense_1')(x)
x = Dropout(0.2)(x)
x = Dense(1024, activation='relu', name='dense_2')(x)
x = Dropout(0.2)(x)
x = Dense(37, activation='softmax', name = 'predictions')(x)
####7th model######

####8th model####### val accuracy: 78% ---- 86% after training the last convolution block 
#x = GlobalAveragePooling2D(name='gap2d')(x)
#x = Dense(1024, activation='relu', name='dense_1')(x)
#x = Dropout(0.4)(x)
#x = Dense(1024, activation='relu', name='dense_2')(x)
#x = Dropout(0.4)(x)
#x = Dense(37, activation='softmax', name = 'predictions')(x)
####8th model######

#create the new model
new_model = Model(inputs=vgg16_model.input, outputs=x)

for i, layer in enumerate(new_model.layers):
    print(i, layer.name, layer.output_shape)
    

#train only the new layers we just added ie only the FC layers
for i, layer in enumerate(new_model.layers):
  if i <= 18:
    layer.trainable = False
  else:
    layer.trainable = True

#train the model
#normally we first test the validation of model and after that we train it with no validation data
#now we omit the validation test because we have the accuracy on it
new_model.compile(optimizer = Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['acc'])
new_model.fit(train_data,train_labels_ohe, batch_size=128, epochs=15, verbose=1, shuffle=True)
new_model.evaluate(test_data,test_labels_ohe)

#now train the last convolution block
for i, layer in enumerate(new_model.layers):
  if i <= 14:
    layer.trainable = False
  else:
    layer.trainable = True

new_model.summary()

new_model.compile(optimizer = Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['acc'])
new_model.fit(train_data,train_labels_ohe, batch_size=128, epochs=15, verbose=1, shuffle=True)
new_model.evaluate(test_data,test_labels_ohe)



#layer we want to take the feature vector
layer_names = ['gap2d', 'dense_1', 'dense_2', 'predictions']

#set the k nearest images you want to get
n_nearest_imgs = 20
#calculate the total number of images that we will get back
max_images = n_nearest_imgs * test_data.shape[0]

#create an array to count the correct returned images
#rows are the different layers
#columns are the precision of breed, precision of pet, recall of breed

score = np.zeros((4,3))
instances_per_class = np.zeros((37,1))

for i in range(0,37):
  elements = np.where(train_labels[:,0]==i)[0]
  instances_per_class[i,0] = np.size(elements)


for ind, cnn_layer in enumerate(layer_names): #iterate through layers and count precision and recall for test data
  print(cnn_layer)
  features = Model(input = new_model.input, output=new_model.get_layer(cnn_layer).output)
  train_pred = features.predict(train_data)
  test_pred = features.predict(test_data)
  
  #find the k nearest neighbors of an image
  nn_model = NearestNeighbors(n_neighbors=n_nearest_imgs, metric='cosine')
  nn_model.fit(train_pred)
  
  for j in range(0,test_data.shape[0]):  #iterate throught test images
    an_img = test_pred[j,:].reshape(1, -1)
    distances, indices = nn_model.kneighbors(an_img)
    
    s=0
    for w in np.nditer(indices): #iterate through the most similar images
      if train_labels[w,0] == test_labels[j,0]:
        score[ind,0] += 1
        s += 1
        
      if train_labels[w,1] == test_labels[j,1]:
        score[ind,1] += 1
    
    score[ind,2] += s/instances_per_class[int(test_labels[j,0]),0] #recall per image
    
    
  score[ind,2] /= test_data.shape[0] #recall of breed
  score[ind,0] /= max_images #precision of breed
  score[ind,1] /= max_images #precision of pet
  print(score[ind,:])

#get the most relevant images from the last FC layer (predictions layer)
features = Model(input = new_model.input, output=new_model.get_layer('predictions').output)
train_pred = features.predict(train_data)
test_pred = features.predict(test_data)

#find the k nearest neighbors of an image
nn_model = NearestNeighbors(n_neighbors=n_nearest_imgs, metric='cosine')
nn_model.fit(train_pred)

an_img_ind = 10

an_img = test_pred[an_img_ind,:].reshape(1, -1)
distances, indices = nn_model.kneighbors(an_img)


print("label of query image ",test_labels[an_img_ind])
for k in np.nditer(indices):
  print("label of relevant image ",train_labels[k])


plt.figure(figsize=(3, 3))
plt.imshow(test_data[an_img_ind,:,:,:])
plt.axis('off')
plt.show()


www=1
plt.figure(figsize=(20,20))
for k in np.nditer(indices):  
  plt.subplot(2,10,www)
  plt.imshow(train_data[k,:,:,:])
  plt.axis('off')
  #plt.show()
  
  www +=1
plt.show()