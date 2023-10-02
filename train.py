# from pyimagesearch import config
import tensorflow as tf
from tensorflow import keras
from keras.applications.vgg16 import VGG16
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Input
from keras.models import Model
from keras.optimizers import Adam
from tensorflow.keras.utils import img_to_array
from tensorflow.keras.utils import load_img
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import tensorflow_addons as tfa
from boundingbox import train_df
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()

# load the contents of the CSV annotations file
IMAGES_PATH = r'train'
TEST_FILENAMES = os.path.sep.join(["output", "test_images.txt"])
MODEL_PATH = os.path.sep.join(["output", "detector.h5"])
LOSS_PLOT_PATH = os.path.sep.join(["output", "plot.png"])
ACC_PLOT_PATH = os.path.sep.join(["output", "acc.png"])
INIT_LR = 1e-4
NUM_EPOCHS = 25
BATCH_SIZE = 32

data = []
targets = []
filenames = []

print("[INFO] loading dataset...")
rows = train_df['annotation']
count = 0
for row in rows:
    filename = train_df['image_name'][count]
    (_, startX, startY, endX, endY) = row
    row = [filename, startX, startY, endX, endY]
    count += 1
    # derive the path to the input image, load the image (in OpenCV
    # format), and grab its dimensions
    imagePath = os.path.sep.join([IMAGES_PATH, filename])
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

# convert the data and targets to NumPy arrays, scaling the input
# pixel intensities from the range [0, 255] to [0, 1]
data = np.array(data, dtype="float32") / 255.0
targets = np.array(targets, dtype="float32")
# partition the data into training and testing splits using 90% of
# the data for training and the remaining 10% for testing
split = train_test_split(data, targets, filenames, test_size=0.10,
    random_state=42)
# unpack the data split
(trainImages, testImages) = split[:2]
(trainTargets, testTargets) = split[2:4]
(trainFilenames, testFilenames) = split[4:]
# write the testing filenames to disk so that we can use then
# when evaluating/testing our bounding box regressor
print("[INFO] saving testing filenames...")
f = open(TEST_FILENAMES, "w")
f.write("\n".join(testFilenames))
f.close()

# def generalized_IOU_loss(y_true, y_predict):
#     #print(tf.keras.backend.eval(y_true))
#     #print(tf.keras.backend.eval(y_true[:, 0]))
#     #(x1p, y1p, x2p, y2p) = y_predict
#     print(tf.keras.backend.eval(y_true))
#     print(tf.keras.backend.eval(y_predict))
#     x1p = tf.keras.backend.eval(y_predict[:, 0])
#     y1p = tf.keras.backend.eval(y_predict[:, 1])
#     x2p = tf.keras.backend.eval(y_predict[:, 2])
#     y2p = tf.keras.backend.eval(y_predict[:, 3])
#     #(x1g, y1g, x2g, y2g) = y_true
#     x1g = tf.keras.backend.eval(y_true[:, 0])
#     y1g = tf.keras.backend.eval(y_true[:, 1])
#     x2g = tf.keras.backend.eval(y_true[:, 2])
#     y2g = tf.keras.backend.eval(y_true[:, 3])
#     #if x2p > x1p and y2p > y1p:
#     x1phat = np.minimum(x1p, x2p)
#     x2phat = np.maximum(x1p, x2p)
#     y1phat = np.minimum(y1p, y2p)
#     y2phat = np.maximum(y1p, y2p)
#     Ag = np.multiply((x2g - x1g), (y2g - y1g))
#     Ap = np.multiply((x2phat - x1phat), (y2phat - y1phat))
#     x1I = np.maximum(x1phat, x1g)
#     x2I = np.minimum(x2phat, x2g)
#     y1I = np.maximum(y1phat, y1g)
#     y2I = np.minimum(y2phat, y2g)
#     I = np.where(np.logical_and(x2I > x1I, y2I > y1I), (x2I - x1I) * (y2I - y1I), 0)
#     #(x2I - x1I) * (y2I - y1I) if (x2I > x1I and y2I > y1I) else 0
#     x1c = np.minimum(x1phat, x1g)
#     x2c = np.maximum(x2phat, x2g)
#     y1c = np.minimum(y1phat, y1g)
#     y2c = np.maximum(y2phat, y2g)
#     Ac = np.multiply((x2c - x1c), (y2c - y1c))
#     U = Ap + Ag - I
#     IoU = np.divide(I, U)
#     GIoU = np.divide(IoU - (Ac - U), Ac)
#     GIOU_loss = 1-GIoU
#     return tf.math.reduce_sum(GIOU_loss)

# def GIoU(bboxes_1, bboxes_2):
#     # https://github.com/shjo-april/Tensorflow_GIoU/blob/master/README.md
#     # https://www.ai.rug.nl/~mwiering/GROUP/ARTICLES/DNN_IOU_SEGMENTATION.pdf
#     # 1. calulate intersection over union
#     area_1 = (bboxes_1[..., 2] - bboxes_1[..., 0]) * (bboxes_1[..., 3] - bboxes_1[..., 1])
#     area_2 = (bboxes_2[..., 2] - bboxes_2[..., 0]) * (bboxes_2[..., 3] - bboxes_2[..., 1])
    
#     intersection_wh = tf.minimum(bboxes_1[:, 2:], bboxes_2[:, 2:]) - tf.maximum(bboxes_1[:, :2], bboxes_2[:, :2])
#     intersection_wh = tf.maximum(intersection_wh, 0)
    
#     intersection = intersection_wh[..., 0] * intersection_wh[..., 1]
#     union = (area_1 + area_2) - intersection
    
#     ious = intersection / tf.maximum(union, 1e-10)

#     # 2. (C - (A U B))/C
#     C_wh = tf.maximum(bboxes_1[..., 2:], bboxes_2[..., 2:]) - tf.minimum(bboxes_1[..., :2], bboxes_2[..., :2])
#     C_wh = tf.maximum(C_wh, 0.0)
#     C = C_wh[..., 0] * C_wh[..., 1]
    
#     giou = ious - (C - union) / tf.maximum(C, 1e-10)
#     return 1-giou

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
    
# load the VGG16 network, ensuring the head FC layers are left off
vgg = tf.keras.applications.vgg16.VGG16(weights="imagenet", include_top=False,
    input_tensor=Input(shape=(224, 224, 3)))
# freeze all VGG layers so they will *not* be updated during the
# training process
vgg.trainable = False
# flatten the max-pooling output of VGG
flatten = vgg.output
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
    trainImages, trainTargets,
    validation_data=(testImages, testTargets),
    batch_size=BATCH_SIZE,
    epochs=NUM_EPOCHS,
    verbose=1)

# serialize the model to disk
print("Keys: ", H.history.keys())

print("[INFO] saving object detector model...")
model.save(MODEL_PATH, save_format="h5")
# plot the model training history
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

# plot the model training history
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