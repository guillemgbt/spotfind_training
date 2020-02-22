import os
import numpy as np
from random import shuffle
from pathlib import Path
from shutil import copy2, rmtree
import sys
import pickle
from optparse import OptionParser
import xml.etree.ElementTree as ET
from glob import glob
import pandas as pd


def compute_aligned_boxes(contour_node, box_type):

    points = np.zeros((4, 2))

    i = 0
    for point_node in contour_node:
        points[i][0] = int(point_node.attrib['x'])
        points[i][1] = int(point_node.attrib['y'])
        i += 1

    if box_type == "mean":
        x_list = list(points[:, 0])
        y_list = list(points[:, 1])
        x_list.sort()
        y_list.sort()

        x_min = int((x_list[0]+x_list[1])*0.5)
        y_min = int((y_list[0]+y_list[1])*0.5)
        x_max = int((x_list[2]+x_list[3])*0.5)
        y_max = int((y_list[2]+y_list[3])*0.5)
        return x_min, y_min, x_max, y_max

    elif box_type == "minmax":
        x_min = np.min(points[:, 0])
        y_min = np.min(points[:, 1])
        x_max = np.max(points[:, 0])
        y_max = np.max(points[:, 1])
        return x_min, y_min, x_max, y_max

    return None


parser = OptionParser()
parser.add_option("-p", "--path",
                  dest="dataset_root",
                  help="Path to PKLot dataset root directory.")
parser.add_option("-b", "--boxtype",
                  dest="box_type",
                  help="How to align bounding boxes. Using mean of minmax",
                  default="mean")

(options, args) = parser.parse_args()

if not options.dataset_root:   # if filename is not given
    parser.error('Error: path to test data must be specified. Pass --path to command line')

DATASET_ROOT = options.dataset_root
BOX_TYPE = options.box_type
TRAINING_PERCENTAGE = 0.8
TRAINING_DIR = './training/'
TEST_DIR = './test/'
TRAINING_IMG_DIR = TRAINING_DIR + 'images/'
TEST_IMG_DIR = TEST_DIR + 'images/'


# CLEARING DIRECTORIES
rmtree(TEST_DIR)
rmtree(TRAINING_DIR)

# CREATING DIRECTORIES
try:
    os.mkdir(TRAINING_DIR)
    os.mkdir(TEST_DIR)
    os.mkdir(TRAINING_IMG_DIR)
    os.mkdir(TEST_IMG_DIR)

except OSError:
    print("Creation of the directories")
else:
    print("Successfully created the directories")


# LOADING ALL XML FILES
annotations = glob(DATASET_ROOT+'/PKLot/**/**/**/*.xml')
print("-> Annotations length:", len(annotations))

table_info = []
img_count = 0

# PARSING ALL SPOTS DATA IN ALL XML
for file in annotations:
    xml_file = file
    print("-> Parsing:", file)
    img_file = file.split('.xml')[0] + '.jpg'
    new_img_name = str(img_count) + '.jpg'

    parsedXML = ET.parse(file)

    for node in parsedXML.getroot().iter('space'):
        if 'occupied' in node.attrib:
            occupied = int(node.attrib['occupied'])

            occupancy = 'occupied' if int(node.attrib['occupied']) == 1 else 'free'

            x_min, y_min, x_max, y_max = compute_aligned_boxes(contour_node=node.find('contour'),
                                                               box_type=BOX_TYPE)

            row_info = [img_file, new_img_name, occupancy, x_min, y_min, x_max, y_max]
            table_info.append(row_info)

    img_count += 1


full_data_frame = pd.DataFrame(table_info,
                               columns=['prev_img_path', 'img_path', 'occupancy', 'xmin', 'ymin', 'xmax', 'ymax'])

groups = [df for _, df in full_data_frame.groupby('prev_img_path')]
shuffle(groups)
first_group_test_index = int(len(groups)*TRAINING_PERCENTAGE)


training_df = pd.concat(groups[:first_group_test_index]).reset_index(drop=True)
test_df = pd.concat(groups[first_group_test_index:]).reset_index(drop=True)


for index, row in training_df.iterrows():
    image_path = row['prev_img_path']
    new_img_path = TRAINING_IMG_DIR + row['img_path']
    training_df.set_value(index, 'img_path', new_img_path)

    img_file = Path(new_img_path)
    if not img_file.exists():
        copy2(image_path, new_img_path)

training_df[['img_path', 'occupancy', 'xmin', 'ymin', 'xmax', 'ymax']].to_csv(TRAINING_DIR + 'training.csv', index=False)


for index, row in test_df.iterrows():
    image_path = row['prev_img_path']
    new_img_path = TEST_IMG_DIR + row['img_path']
    test_df.set_value(index, 'img_path', new_img_path)

    img_file = Path(new_img_path)
    if not img_file.exists():
        copy2(image_path, new_img_path)

test_df[['img_path', 'occupancy', 'xmin', 'ymin', 'xmax', 'ymax']].to_csv(TEST_DIR + 'test.csv', index=False)

print(training_df.head())
print(test_df.head())

