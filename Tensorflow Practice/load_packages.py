from __future__ import absolute_import, division, print_function

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import cv2
import warnings
from random import randint
from tensorflow.keras import optimizers
from dominance_analysis import Dominance_Datasets
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
