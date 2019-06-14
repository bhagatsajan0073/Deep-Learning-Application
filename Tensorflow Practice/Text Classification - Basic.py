#!/usr/bin/env python
# coding: utf-8

# ### <a href="https://www.tensorflow.org/api_docs/python/tf/keras/datasets/imdb"> We'll use the IMDB dataset that contains the text of 50,000 movie reviews from the Internet Movie Database></a>

# In[1]:


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
from random import randint
from tensorflow.keras import optimizers
from sklearn.model_selection import train_test_split

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


# In[2]:


imdb=keras.datasets.imdb
(train_reviews,train_sentiments),(validation_reviews,validation_sentiments)=imdb.load_data(num_words=10000)


# In[3]:


print("No.of Training Reviews :%s and Valdiation Reviews : %s"%(train_reviews.shape[0],validation_reviews.shape[0]))


# In[4]:


for i in range(5):
   print("no. of words in ",str(i)+"th review :",len(train_reviews[i]))


# ### convert reviews containing word index into actual review

# In[5]:


word_index=imdb.get_word_index()
word_index={k:v+3 for k,v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2  # unknown
word_index["<UNUSED>"] = 3
index_word_mapping={value:key for key,value in word_index.items()}


# In[6]:


index_word_mapping[0]


# In[7]:


def orignal_review(review_array):
    return (" ".join([index_word_mapping.get(i,'?') for i in review_array]))

for i in range(100,102):
    print(str(i+1)+"th Review :",orignal_review(train_reviews[i]))
    print("")


# ### Pre-Process the data to have max length of review to be 256

# In[8]:


train_reviews=keras.preprocessing.sequence.pad_sequences(train_reviews,value=word_index['<PAD>'],padding='post',maxlen=256)
validation_reviews=keras.preprocessing.sequence.pad_sequences(validation_reviews,value=word_index['<PAD>'],padding='post',maxlen=256)


# In[9]:


assert(train_reviews.shape[1]==validation_reviews.shape[1])


# In[10]:


orignal_review(train_reviews[0])


# In[11]:


orignal_review(validation_reviews[0])


# ### Build Model

# In[184]:


# input shape is the vocabulary count used for the movie reviews (10,000 words)
vocab_size = 10000

model = keras.Sequential()
model.add(keras.layers.Embedding(vocab_size, 16,input_length=256))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation=tf.nn.relu))
model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))

model.summary()


# In[185]:


model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['acc'])


# In[186]:


x_val = train_reviews[:10000]
partial_x_train = validation_reviews[10000:]

y_val = train_sentiments[:10000]
partial_y_train = validation_sentiments[10000:]


# In[187]:


gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
tf.Session(config=tf.ConfigProto(gpu_options=gpu_options,allow_soft_placement=True))


# In[188]:


history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=30,
                    batch_size=512,
                    validation_data=(x_val, y_val),
                    verbose=0)


# In[189]:


history_dict=history.history


# In[190]:


import matplotlib.pyplot as plt

acc = history_dict['acc']
val_acc = history_dict['val_acc']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(acc) + 1)

# "bo" is for "blue dot"
plt.plot(epochs, loss, 'bo', label='Training loss')
# b is for "solid blue line"
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()


# In[191]:


import matplotlib.pyplot as plt

acc = history_dict['acc']
val_acc = history_dict['val_acc']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(acc) + 1)

# "bo" is for "blue dot"
plt.plot(epochs, acc, 'ro', label='Training Accuracy')
# b is for "solid blue line"
plt.plot(epochs, val_acc, 'r', label='Validation Accuracy')
plt.title('Training and validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()


# In[192]:


results = model.evaluate(validation_reviews, validation_sentiments)
print(results)


# In[ ]:




