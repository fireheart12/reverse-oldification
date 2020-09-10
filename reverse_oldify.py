# -*- coding: utf-8 -*-
"""reverse_oldify.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1pbvEjkDIZolNaMIRKqI5iqYSVRpZVq3i
"""

from google.colab import drive
drive.mount("/content/gdrive")

# Commented out IPython magic to ensure Python compatibility.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
# %matplotlib inline
import cv2
import skimage.color as sk
from tqdm import tqdm_notebook as tqdm

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16

# path : /content/gdrive/My Drive/Datasets/Landscapes_snapshots/landscapes
path = "/content/gdrive/My Drive/Datasets/Landscapes_snapshots"

train_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale = 1/255.0, validation_split= 0.15)

train = train_gen.flow_from_directory(path, target_size = (IMAGE_SIZE[0], IMAGE_SIZE[1]), 
                                      batch_size = BATCH_SIZE, class_mode = None, subset = "training")

val = train_gen.flow_from_directory(path, target_size = (IMAGE_SIZE[0], IMAGE_SIZE[1]), 
                                    batch_size = BATCH_SIZE, class_mode = None, subset = "validation")

print(train[0].shape) 
print(train[1].shape) 
print(train[2].shape)

print(val[0].shape) 
print(val[1].shape) 
print(val[2].shape)

print(3681/BATCH_SIZE)
print(649/BATCH_SIZE)

# we have approx 230 batches. Finding the last one so we can iterate on 'train'
# we have approx 40 batches. Finding the last one so we can iterate on 'val'
print(train[230].shape)
print(val[40].shape)

"""As number of images in 231st batch here(numbering starts from 0)  = 1, this implies it is the final batch.
Similar conclusion is drawn about the validation set too.
"""

def convert_lab(image) : 
  lab_image = sk.rgb2lab(image)
  return lab_image

def convert_rgb(image) : 
  rgb_image = sk.lab2rgb(image)
  return rgb_image
  
def plot_image(image) : 
  plt.figure(figsize = (12, 8))
  plt.imshow(image, cmap = "gray")
  plt.grid(False)

batch_of_images = train[0]
for i in batch_of_images : 
  plot_image(i)

# plotting the L channel
for i in batch_of_images : 
  plot_image(convert_lab(i)[:,:,0])

x_train = []
y_train = []
for i in tqdm(range(230)) : 
  for image in train[i] : 
    try : 
      lab_image = convert_lab(image)
      x_train.append(lab_image[:,:,0])
      y_train.append(lab_image[:,:,1:] / 128)
    except : 
      print("Unexpected error. Maybe broken image.")
x_train = np.array(x_train)
y_train = np.array(y_train)
print(x_train.shape)
print(y_train.shape)

x_val = []       
y_val = []
for i in tqdm(range(40)) : 
  for image in val[i] : 
    try : 
      lab_image = convert_lab(image)
      x_val.append(lab_image[:,:,0])
      y_val.append(lab_image[:,:,1:] / 128)
    except : 
      print("Unexpected error. Maybe broken image.")
x_val = np.array(x_val)
y_val = np.array(y_val)
print(x_val.shape)     
print(y_val.shape)

x_train = x_train.reshape(x_train.shape[0], IMAGE_SIZE[0], IMAGE_SIZE[1], 1)
x_val = x_val.reshape(x_val.shape[0], IMAGE_SIZE[0], IMAGE_SIZE[1], 1)
print(x_train.shape)
print(y_train.shape)
print(x_val.shape)
print(y_val.shape)

np.save("x_train", x_train)
np.save("y_train", y_train)
np.save("x_val", x_val)
np.save("y_val", y_val)

x_train = np.load("x_train.npy")  
y_train = np.load("y_train.npy")
x_val = np.load("x_val.npy")        
y_val = np.load("y_val.npy")

"""# Deep Learning Architecture - AutoEncoder"""

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor= "loss", factor = 0.5, patience = 10,
                                                 min_lr = 0.000001, verbose = 1)
monitor_es = tf.keras.callbacks.EarlyStopping(monitor= "loss", patience = 25, restore_best_weights= False, verbose = True)

# transfer learning
vgg_model = tf.keras.applications.vgg16.VGG16()
transfer_learned_encoder_model = tf.keras.models.Sequential()
for i, layer in enumerate(vgg_model.layers):
  if i < 19 : 
    transfer_learned_encoder_model.add(layer)
for layer in transfer_learned_encoder_model.layers:
  layer.trainable = False

transfer_learned_encoder_model.summary()

vgg_features = []
for i, image in tqdm(enumerate(x_train)) : 
  image = cv2.merge((image, image, image))
  image = image.reshape((1,IMAGE_SIZE[0],IMAGE_SIZE[1],3))
  prediction = transfer_learned_encoder_model.predict(image)
  prediction = prediction.reshape((7,7,512))
  vgg_features.append(prediction)
vgg_features = np.array(vgg_features)
print(vgg_features.shape)

# Encoder
input_shape = (7, 7, 512)
i = tf.keras.layers.Input(shape = input_shape)  

#decoder
output = tf.keras.layers.Conv2D(filters = 128, kernel_size= (3,3), padding = "same", activation= "relu")(i)
output = tf.keras.layers.Conv2D(filters = 128, kernel_size= (3,3), padding = "same", activation= "relu")(i)
output = tf.keras.layers.Conv2D(filters = 128, kernel_size= (3,3), padding = "same", activation= "relu")(i)
output = tf.keras.layers.UpSampling2D((2, 2))(output)
output = tf.keras.layers.Conv2D(filters = 64, kernel_size= (3,3), padding = "same", activation= "relu")(output)
output = tf.keras.layers.Conv2D(filters = 64, kernel_size= (3,3), padding = "same", activation= "relu")(output)
output = tf.keras.layers.Conv2D(filters = 64, kernel_size= (3,3), padding = "same", activation= "relu")(output)
output = tf.keras.layers.UpSampling2D((2, 2))(output)
output = tf.keras.layers.Conv2D(filters = 32, kernel_size= (3,3), padding = "same", activation= "relu")(output)
output = tf.keras.layers.Conv2D(filters = 32, kernel_size= (3,3), padding = "same", activation= "relu")(output)
output = tf.keras.layers.Conv2D(filters = 32, kernel_size= (3,3), padding = "same", activation= "relu")(output)
output = tf.keras.layers.UpSampling2D((2, 2))(output)
output = tf.keras.layers.Conv2D(filters = 16, kernel_size= (3,3), padding = "same", activation= "relu")(output)
output = tf.keras.layers.Conv2D(filters = 16, kernel_size= (3,3), padding = "same", activation= "relu")(output)
output = tf.keras.layers.Conv2D(filters = 16, kernel_size= (3,3), padding = "same", activation= "relu")(output)
output = tf.keras.layers.UpSampling2D((2, 2))(output)
output = tf.keras.layers.Conv2D(filters = 2, kernel_size= (3,3), padding = "same", activation= "tanh")(output)
output = tf.keras.layers.UpSampling2D((2, 2))(output)

decoder_model = tf.keras.models.Model(inputs = i, outputs = output)
decoder_model.summary()

decoder_model.compile(optimizer= tf.keras.optimizers.Adam(lr = 0.001), loss = "MSE", metrics = ["accuracy"])

vgg_features.shape

y_train.shape

EPOCHS = 1000
with tf.device("/device:GPU:0"):
  history = decoder_model.fit(vgg_features, y_train, epochs = EPOCHS, verbose = 1, 
                      callbacks = [reduce_lr, monitor_es], batch_size = BATCH_SIZE)

x = np.arange(0, EPOCHS, 1)                                                                  
plt.figure(1, figsize = (20, 12))                                  
plt.subplot(121)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.plot(x, history.history["loss"], label = "Training Loss")
plt.plot(x, history.history["accuracy"], label = "Training Accuracy")
plt.grid(True)                           
plt.legend()

!mkdir -p saved_model
transfer_learned_encoder_model.save('saved_model/encoder_model') 
decoder_model.save('saved_model/decoder_model')

validation_images = x_val[:15]
vgg_features_val = []
for i, image in tqdm(enumerate(validation_images)) :
  image = cv2.merge((image, image, image))
  image = image.reshape((1,IMAGE_SIZE[0],IMAGE_SIZE[1],3))
  prediction = transfer_learned_encoder_model.predict(image)
  prediction = prediction.reshape((7,7,512))
  vgg_features_val.append(prediction)
vgg_features_val = np.array(vgg_features_val)
print(vgg_features_val.shape)

ab_pred = decoder_model.predict(vgg_features_val)
ab_pred = ab_pred * 128
print(ab_pred.shape)

print(validation_images.shape)

for i in range(validation_images.shape[0]) : 
  image = validation_images[i]
  image = image.reshape((IMAGE_SIZE[0] , IMAGE_SIZE[1]))
  
  reconstructed_image =  np.zeros((IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
  reconstructed_image[:,:,0] =  image
  reconstructed_image[:,:,1:] = ab_pred[i]

  #reconstructed_image = reconstructed_image.astype(np.uint8)
  reconstructed_image = convert_rgb(reconstructed_image)
  
  image = cv2.resize(image, (1024, 1024))
  reconstructed_image = cv2.resize(reconstructed_image, (1024, 1024))

  plot_image(image)
  plot_image(reconstructed_image)

from google.colab import files

!zip -r saved_model.zip saved_model

"""Simply download the model. I saved it at : 

**https://drive.google.com/drive/folders/1agiL9_gxeeJ4YTx_pGeglHfGwf1tTI5G?usp=sharing**
"""