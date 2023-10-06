from keras.preprocessing.image import Iterator, ImageDataGenerator
from train.py import traindf
# from pyimagesearch import config
import tensorflow as tf
from tensorflow import keras
from keras.applications.vgg16 import VGG16
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Input
from keras.layers import Dropout
from keras.models import Model
from keras.optimizers import Adam
from tensorflow.keras.utils import img_to_array
from tensorflow.keras.utils import load_img
from tensorflow.keras.utils import Sequence
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import tensorflow_addons as tfa
from boundingbox import train_df

import shutil
from data_aug import *
from bbox_util import *

IMAGES_PATH = r'train/'

class MyIterator(Iterator):
  """This is a toy example of a wrapper around ImageDataGenerator"""

  def __init__(self, n, batch_size, shuffle, seed, dataframe, **kwargs):
    super().__init__(n, batch_size, shuffle, seed)

    # Load any data you need here (CSV, HDF5, raw stuffs). The code
    # below is just a pseudo-code for demonstration purpose.
    filenames = dataframe['imageid']
    self.input_images = []
    self.ground_truths = []
    for filename in filenames:
        imagePath = os.path.sep.join([IMAGES_PATH, filename])
        image = cv2.imread(imagePath)
        self.input_images.append(image)
    self.ground_truths.append(dataframe['target'])

    print("input images:", self.input_images)
    print("ground truths:", self.ground_truths)
    # Here is our beloved image augmentator <3
    self.generator = ImageDataGenerator(**kwargs)
    
  def _get_batches_of_transformed_samples(self, index_array):
    """Gets a batch of transformed samples from array of indices"""

    # Get a batch of image data
    batch_x = self.input_images[index_array].copy()
    batch_y = self.ground_truths[index_array].copy()

    # Transform the inputs and correct the outputs accordingly
    for i, (x, y) in enumerate(zip(batch_x, batch_y)):
      transform_params = self.generator.get_random_transform(x.shape)
      batch_x[i] = self.generator.apply_transform(x, transform_params)
      batch_y[i] = process_outputs_accordingly(y, transform_params)

    return batch_x, batch_y

MyIterator(Sequence(RandomHorizontalFlip(), RandomScale(), RandomTranslate(), RandomRotate(10), RandomShear()), dataframe = traindf)