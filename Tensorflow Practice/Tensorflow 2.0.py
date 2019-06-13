#!/usr/bin/env python
# coding: utf-8

# In[177]:


# %load load_packages.py
from __future__ import absolute_import, division, print_function

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2
import warnings
warnings.filterwarnings(action='ignore')

def print_package_versions(log_flag=False):
    if(log_flag):
        print("Tensorflow Version :",tf.__version__)
        print("Pandas Version :",pd.__version__)
        print("Numpy Version :",np.__version__)
        print("Keras Version :",keras.__version__)
        print("OpenCV Version :",cv2.__version__)
    else:
        pass

print_package_versions(True)


# ### <a href="https://github.com/zalandoresearch/fashion-mnist">MNIST Dataset for Fashion</a>

# In[49]:


fashion_mnist=keras.datasets.fashion_mnist
(train_images,train_labels),(validation_images,validation_labels)=fashion_mnist.load_data()


# In[57]:


print("No. of images in Train Dataset : ",train_images.shape[0])
print("Shape of images in Train Dataset :",train_images.shape[1:])
print("*"*50)
print("*"*50)
print("No. of images in Validation Dataset : ",validation_images.shape[0])
print("Shape of images in Validation Dataset :",validation_images.shape[1:])


# In[65]:


print("No. of Fashion Categories in Training :",np.unique(train_labels).shape[0])
print("*"*50)
print("*"*50)
print("No. of Fashion Categories in Validation :",np.unique(validation_labels).shape[0])


# In[66]:


labels=['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


# ### PreProcess the data

# In[165]:


def render_images(images,image_labels,grid_flag=False,flag_cv2=False):
    if(grid_flag is not True):
        if(flag_cv2):
            image=cv2.imshow("Fashion_Image",images)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            plt.figure(figsize=(2,2))
            plt.imshow(images,cmap=plt.cm.binary)
            plt.colorbar()
            plt.xlabel(image_labels)
    else:
        plt.figure(figsize=(15,15))
        for i in range(images.shape[0]):
            plt.subplot(10,10,i+1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imshow(images[i], cmap=plt.cm.binary)
            plt.xlabel(labels[image_labels[i]])
        plt.show()
    

for i in range(2):
    render_images(train_images[i],labels[train_labels[i]])


# In[166]:


print("Grid Rendering :")    
render_images(train_images[0:30],train_labels[0:30],grid_flag=True)


# ### Model Building
# 
# 1. Define the Layers
# 2. Compile the Model
# 3. Train the Model
# 4. Evaluated the Model Accuracy on Hold-Out Set or Validation-Set
# 5. Make Prediction for Scoring Data-Set

# In[174]:


# import warnings
# warnings.filterwarnings(action='ignore')


# In[181]:


train_images=train_images/255.
validation_images=validation_images/255.


# In[246]:


from sklearn.model_selection import train_test_split
train_x,test_x,train_y,test_y=train_test_split(train_images,train_labels,stratify=train_labels,
                                               test_size=0.16666,shuffle=True)
train_x.shape,test_x.shape


# In[279]:


model=keras.Sequential([
    keras.layers.Flatten(input_shape=train_images.shape[1:]),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax),
    keras.layers.Dropout(rate=0.0)
])


# In[280]:


from tensorflow.keras import optimizers
model.compile(optimizer=optimizers.Adam(lr=0.01),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'],
             )


# In[ ]:


model.fit(train_x, train_y, epochs=100,verbose=0,use_multiprocessing=True,workers=4,validation_data=(test_x,test_y))
# model.fit(train_images, train_labels, epochs=100,verbose=1,use_multiprocessing=True,workers=4,validation_split=0.16666)


# In[ ]:


training_loss,training_accuracy=model.evaluate(train_images,train_labels)
validation_loss,validation_accuracy=model.evaluate(validation_images,validation_labels)


# In[ ]:


print("Train Loss : %s and Traing Accuracy : %s"%(training_loss,training_accuracy))
print("*"*50)
print("*"*50)
print("Validation Loss : %s and Validation Accuracy : %s"%(validation_loss,validation_accuracy))

