from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import pandas as pd
from PIL import Image

# HOW TO USE: python tf_record.py --output_path training.record

flags = tf.app.flags
flags.DEFINE_string('dataset_type', 'training', 'training or test')
flags.DEFINE_string('num_images', None, 'number of imagesa conforming the output data set')
FLAGS = flags.FLAGS

LABEL_DICT = {
    "occupied": 1,
    "free": 2
}


def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def int64_list_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def bytes_list_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def float_list_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def read_examples_list(path):
    with tf.gfile.GFile(path) as fid:
        lines = fid.readlines()
    return [line.strip().split(' ')[0] for line in lines]


def create_tf_example(image_path, same_image_samples):

    image = Image.open(image_path)

    height = image.size[1]
    width = image.size[0]
    filename = image_path.encode() # Filename of the image. Empty if image is not from file
    image_format = "jpg".encode() # b'jpeg' or b'png'

    with tf.gfile.GFile(image_path, 'rb') as fid:
        encoded_image_data = fid.read()

    xmins = [] # List of normalized left x coordinates in bounding box (1 per box)
    xmaxs = [] # List of normalized right x coordinates in bounding box
             # (1 per box)
    ymins = [] # List of normalized top y coordinates in bounding box (1 per box)
    ymaxs = [] # List of normalized bottom y coordinates in bounding box
             # (1 per box)
    classes_text = [] # List of string class name of bounding box (1 per box)
    classes = [] # List of integer class id of bounding box (1 per box)

    for index, row in same_image_samples.iterrows():
        xmins.append(float(row['xmin'])/width)
        xmaxs.append(float(row['xmax'])/width)
        ymins.append(float(row['ymin'])/height)
        ymaxs.append(float(row['ymax'])/height)
        classes_text.append(row['occupancy'].encode())
        classes.append(int(LABEL_DICT[row['occupancy']]))

    tf_label_and_data = tf.train.Example(features=tf.train.Features(feature={
        'image/height': int64_feature(height),
        'image/width': int64_feature(width),
        'image/filename': bytes_feature(filename),
        'image/source_id': bytes_feature(filename),
        'image/encoded': bytes_feature(encoded_image_data),
        'image/format': bytes_feature(image_format),
        'image/object/bbox/xmin': float_list_feature(xmins),
        'image/object/bbox/xmax': float_list_feature(xmaxs),
        'image/object/bbox/ymin': float_list_feature(ymins),
        'image/object/bbox/ymax': float_list_feature(ymaxs),
        'image/object/class/text': bytes_list_feature(classes_text),
        'image/object/class/label': int64_list_feature(classes),
    }))

    return tf_label_and_data


def main(_):

    num_images = None

    dataset_type = FLAGS.dataset_type
    output_path = dataset_type + "/" + dataset_type + ".record"

    data_csv_path = "./" + dataset_type + "/" + dataset_type + ".csv"

    if FLAGS.num_images is not None:
        num_images = int(FLAGS.num_images)
        output_path = dataset_type + "/" + dataset_type + str(num_images) + ".record"

    print("Num img:", num_images)

    df = pd.read_csv(data_csv_path)
    df.head()

    writer = tf.python_io.TFRecordWriter(output_path)

    image_count = 0
    for image_path in df['img_path'].unique():
        image_filtered_df = df[df.img_path == image_path]
        tf_example = create_tf_example(image_path=image_path, same_image_samples=image_filtered_df)
        writer.write(tf_example.SerializeToString())
        image_count += 1
        if num_images == image_count:
            break


if __name__ == '__main__':
    tf.app.run()
