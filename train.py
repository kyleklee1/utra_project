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

from keras.preprocessing.image import ImageDataGenerator
import albumentations as A
from argparse import ArgumentParser
import random
from PIL import Image
import pandas as pd
import re
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()

# load the contents of the CSV annotations file
IMAGES_PATH = r'train/'
TEST_FILENAMES = os.path.sep.join(["output", "test_images.txt"])
MODEL_PATH = os.path.sep.join(["output", "detector.h5"])
LOSS_PLOT_PATH = os.path.sep.join(["output", "plot.png"])
ACC_PLOT_PATH = os.path.sep.join(["output", "acc.png"])
INIT_LR = 1e-4
NUM_EPOCHS = 25
BATCH_SIZE = 32
IMG_SIZE = 224
PREPROCESS_SAMPLE_SIZE = 500

data = []
targets = []
filenames = []

print("[INFO] loading dataset...")
rows = train_df['annotation']
count = 0
file_list = []
for row in rows:
    filename = train_df['image_name'][count]
    (_, startX, startY, endX, endY) = row
    row = [filename, startX, startY, endX, endY]
    count += 1
    # derive the path to the input image, load the image (in OpenCV
    # format), and grab its dimensions
    imagePath = os.path.sep.join([IMAGES_PATH, filename])
    file_list.append(imagePath)
    image = cv2.imread(imagePath)
    (h, w) = image.shape[:2]
    # scale the bounding box coordinates relative to the spatial
    # dimensions of the input image
    startX = float(startX) / w
    startY = float(startY) / h
    endX = float(endX) / w
    endY = float(endY) / h
    # load the image and preprocess it
    image = load_img(imagePath, target_size=(224, 224))
    image = img_to_array(image)
    # update our list of data, targets, and filenames
    data.append(image)
    targets.append((startX, startY, endX, endY))
    filenames.append(filename)
print("File List: ",  file_list)
# convert the data and targets to NumPy arrays, scaling the input
# pixel intensities from the range [0, 255] to [0, 1]
data = np.array(data, dtype="float32") / 255.0
targets = np.array(targets, dtype="float32")
# partition the data into training and testing splits using 90% of
# the data for training and the remaining 10% for testing

split = train_test_split(data, targets, filenames, test_size=0.10,
    random_state=42)
# unpack the data split
(trainImages, validationImages) = split[:2]
(trainTargets, validationTargets) = split[2:4]
(trainFilenames, validationFilenames) = split[4:]

traindf = pd.DataFrame({'image_id' : trainFilenames, 'target': trainTargets})
validdf = pd.DataFrame({'image_id' : validationFilenames, 'target': validationTargets})

# # write the testing filenames to disk so that we can use then
# # when evaluating/testing our bounding box regressor
TRAIN_PATH = r'trainImages'
VALID_PATH = r'validImages'
for filename in trainFilenames:
    original_image = IMAGES_PATH + "/" + filename
    new_image = os.path.sep.join([TRAIN_PATH, filename])
    shutil.copy(original_image, new_image)
for filename in validationFilenames:
    original_image = IMAGES_PATH + "/" + filename
    new_image = os.path.sep.join([VALID_PATH, filename])
    shutil.copy(original_image, new_image)
# print("[INFO] saving testing filenames...")
# f = open(TEST_FILENAMES, "w")
# f.write("\n".join(testFilenames))
# f.close()
 
def IoU_metric(bboxes_1, bboxes_2):
    # https://github.com/shjo-april/Tensorflow_GIoU/blob/master/README.md
    # https://www.ai.rug.nl/~mwiering/GROUP/ARTICLES/DNN_IOU_SEGMENTATION.pdf
    # 1. calulate intersection over union
    area_1 = (bboxes_1[..., 2] - bboxes_1[..., 0]) * (bboxes_1[..., 3] - bboxes_1[..., 1])
    area_2 = (bboxes_2[..., 2] - bboxes_2[..., 0]) * (bboxes_2[..., 3] - bboxes_2[..., 1])
    
    intersection_wh = tf.minimum(bboxes_1[:, 2:], bboxes_2[:, 2:]) - tf.maximum(bboxes_1[:, :2], bboxes_2[:, :2])
    intersection_wh = tf.maximum(intersection_wh, 0)
    
    intersection = intersection_wh[..., 0] * intersection_wh[..., 1]
    union = (area_1 + area_2) - intersection
    
    ious = intersection / tf.maximum(union, 1e-10)
    #print("IoUS:", ious)
    threshold = 0.2
    greater_elts = tf.math.greater(ious, threshold)
    num_greater = tf.math.reduce_sum(tf.cast(greater_elts, tf.int32))
    total = tf.size(ious)
    # print("Num_greater: ", num_greater)
    # print("Total:", total)
    # print("ious:", ious)
    return tf.math.divide(num_greater, total)


def calc_mean_std(file_list):
    # Shuffle filepaths
    random.shuffle(file_list)

    # Take sample of file paths
    file_list = file_list[:PREPROCESS_SAMPLE_SIZE]

    # Allocate space in memory for images
    data_sample = np.zeros( (PREPROCESS_SAMPLE_SIZE, IMG_SIZE, IMG_SIZE, 3))

    # Import images
    for i, file_path in enumerate(file_list):
        img = Image.open(file_path)
        img = img.resize((224, 224))
        img = np.array(img, dtype=np.float32)
        img /= 255.

        # Grayscale -> RGB
        if len(img.shape) == 2:
            img = np.stack([img, img, img], axis=-1)
        data_sample[i] = img

    return np.mean(data_sample, axis=0), np.std(data_sample, axis=0) + 1.0e-8


def custom_preprocess_fn(img):
    #implement image standardization
    img = (img - MEAN)/STD
    return img


MEAN, STD = calc_mean_std(file_list)
####################################
# class AugmentDataGenerator(Sequence):
#     def __init__(self, datagen, augment=None):
#         self.datagen = datagen
#         if augment is None:
#             self.augment = A.Compose([])
#         else:
#             self.augment = augment

#     def __len__(self):
#         return len(self.datagen)

#     def __getitem__(self, x):
#         images, *rest = self.datagen[x]
#         augmented = []
#         for image in images:
#             image = self.augment(image=image)['image']
#             augmented.append(image)
#         return (np.array(augmented), *rest)

#####################################
seq = Sequence(RandomHorizontalFlip(), RandomScale(), RandomTranslate(), RandomRotate(10), RandomShear())

train_datagen = ImageDataGenerator(preprocessing_function = seq).flow_from_dataframe(
        dataframe=traindf,
        directory=TRAIN_PATH,
        x_col='image_id',
        y_col='target',
        target_size=(224, 224),
    )
# create the training and validation sets
# train_generator = train_datagen.flow(TRAIN_PATH,
#                                                     target_size=(256, 256),
#                                                     batch_size=BATCH_SIZE)

# validation_generator = validation_datagen.flow_from_directory(VALID_PATH,
#                                                             target_size=(256, 256),
#                                                             batch_size=BATCH_SIZE)

valid_datagen = ImageDataGenerator(preprocessing_function = seq).flow_from_dataframe(
        dataframe=validdf,
        directory=VALID_PATH,
        x_col='image_id',
        y_col='target',
        target_size=(224, 224),
    )

# load the VGG16 network, ensuring the head FC layers are left off
vgg = tf.keras.applications.vgg16.VGG16(weights="imagenet", include_top=False,
    input_tensor=Input(shape=(224, 224, 3)))
# freeze all VGG layers so they will *not* be updated during the
# training process
vgg.trainable = False
# flatten the max-pooling output of VGG
flatten = vgg.output
flatten = Dropout(0.2)(flatten)
flatten = Flatten()(flatten)
# construct a fully-connected layer header to output the predicted
# bounding box coordinates
bboxHead = Dense(128, activation="relu")(flatten)
bboxHead = Dense(64, activation="relu")(bboxHead)
bboxHead = Dense(32, activation="relu")(bboxHead)
bboxHead = Dense(4, activation="sigmoid")(bboxHead)
# construct the model we will fine-tune for bounding box regression
model = Model(inputs=vgg.input, outputs=bboxHead)

# initialize the optimizer, compile the model, and show the model
# summary
opt = Adam(lr=INIT_LR)
#model.compile(loss="mse", optimizer=opt)
model.compile(loss="mse", optimizer=opt, run_eagerly=True, metrics = [IoU_metric])
print(model.summary())
# train the network for bounding box regression
print("[INFO] training bounding box regressor...")
H = model.fit(
    #trainImages, trainTargets,
    train_datagen,
    #validation_data=(testImages, testTargets),
    validation_data = valid_datagen,
    batch_size=BATCH_SIZE,
    epochs=NUM_EPOCHS,
    verbose=1)

# serialize the model to disk
print("Keys: ", H.history.keys())

print("[INFO] saving object detector model...")
model.save(MODEL_PATH, save_format="h5")
# plot the model training history for loss
N = NUM_EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.title("Bounding Box Regression Loss on Training Set")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.legend(loc="lower left")
plt.savefig(LOSS_PLOT_PATH)

# plot the model training history for accuracy
N = NUM_EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["IoU_metric"], label="train_accuracy")
plt.plot(np.arange(0, N), H.history["val_IoU_metric"], label="val_accuracy")
plt.title("Bounding Box Regression Accuracy on Training Set")
plt.xlabel("Epoch #")
plt.ylabel("Accuracy")
plt.legend(loc="lower left")
plt.savefig(ACC_PLOT_PATH)





