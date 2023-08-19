# import sys
# print(sys.executable)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

import tensorflow as tf

def hasmultiple(s):
    return (';' in s)

def listcoordinates(item):
    coord = item.split(' ')
    anno = []
    for i in coord:
        if hasmultiple(i):
            return []
        else:
            anno.append(int(i))
    if len(anno) != 5:
        return []
    return anno

train_df = pd.read_csv('train.csv')
train_df = train_df.dropna()
train_df = train_df.reset_index()
train_df['annotation'] = train_df['annotation'].map(listcoordinates)
del train_df['index']
train_df = train_df[train_df['annotation'].str[0] == 0]
train_df = train_df.reset_index()
del train_df['index']
print(train_df)

TRAIN_IMG_PATH = r'train'

def conv_str2annot(anno):
    anno = anno.split(' ')
    annotation = []
    
    for i in anno:
        annotation.append(int(i))
        
    return annotation

def find_coord(anno):
    
    annotation = anno
    coordinates = [0, 0, 0, 0]

    x1 = coordinates[0] = annotation[1]
    y1 = coordinates[1] = annotation[2]
    x2 = coordinates[2] = annotation[3]
    y2 = coordinates[3] = annotation[4]

    #print(coordinates)
    return [x1, y1, x2,y2]

def box_corner_to_center(boxes):
    """Convert from (upper-left, lower-right) to (center, width, height)."""
    x1, y1, x2, y2 = boxes[0], boxes[1], boxes[2], boxes[3]
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    boxes = tf.stack((cx, cy, w, h), axis=-1)
    return np.array(boxes)

def box_center_to_corner(boxes):
    """Convert from (center, width, height) to (upper-left, lower-right)."""
    cx, cy, w, h = boxes[0], boxes[1], boxes[2], boxes[3]
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    boxes = tf.stack((x1, y1, x2, y2), axis=-1)
    return np.array(boxes)

def check_bbox_val(coordinates):
    boxes = tf.constant((coordinates))
    box_center_to_corner(box_corner_to_center(coordinates)) == boxes
    return boxes

def bbox_to_rect(bbox, color):
    """Convert bounding box to matplotlib format."""
    # Convert the bounding box (upper-left x, upper-left y, lower-right x,
    # lower-right y) format to the matplotlib format: ((upper-left x,
    # upper-left y), width, height)
    return plt.Rectangle(
        xy=(bbox[0], bbox[1]), width=bbox[2]-bbox[0], height=bbox[3]-bbox[1],
        fill=False, edgecolor=color, linewidth=2)

coordinates = []
for annot in train_df['annotation']:
    anno = []
    for i in annot:
        anno.append(int(i))
    coordinates.append(anno)

#print(coordinates)

coordinates_1 = []
for i in coordinates:
    coordinates_1.append(find_coord(i))

#print(coordinates_1)

for idx,img in enumerate(train_df['image_name']):
    image = cv2.imread(os.path.join(TRAIN_IMG_PATH, img), cv2.COLOR_RGB2GRAY)
    fig = plt.imshow(image)
    fig.axes.add_patch(bbox_to_rect(coordinates_1[idx], 'red'))   
    plt.show()
    #print(coordinates_1[idx])