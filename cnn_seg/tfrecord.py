import tensorflow as tf
import numpy as np
import os
import glob
from data import generator_input, gt_label

slim = tf.contrib.slim

def _int64_feature(value):
    """Wrapper for inserting int64 features into Example proto."""
    if not isinstance(value, list):
        value = value.tolist()
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _float64_feature(value):
    """Wrapper for inserting float64 features into Example proto."""
    if not isinstance(value, list):
        value = value.tolist()
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _bytes_feature(value):
    """Wrapper for inserting bytes features into Example proto."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))



def _convert_to_example(input, label):
    # all_cats = cats.tolist()
    feature = input.flatten().tostring()
    gt = label.flatten().tostring()
    example = tf.train.Example(features=tf.train.Features(feature={
        'input': _bytes_feature(feature),
        'label': _bytes_feature(gt)
    }))

    return example

def create_kitti(input_path, label_path, width, height, in_channel, lab_channel, range_, maxh, minh):
    input = generator_input(input_path, width, height, in_channel, range_, maxh, minh)
    label = gt_label(label_path, width, height, lab_channel)
    return input, label

def get_data_paths(bin_id, data_dir):
    #image_id = '006961'
    bin_file = '{}/{}.bin'.format(data_dir, bin_id)
    label_file = os.path.join(data_dir, "{}.txt".format(bin_id))
    print(bin_file, label_file)
    return bin_file, label_file

def start(width, height, in_channel, lab_channel, range_, maxh, minh):
    data_dir = '/home/bai/Project/cnn_seg/dataset'
    bin_id = glob.glob('%s/*.bin' % (data_dir))
    out_dir = '/home/bai/Project/cnn_seg/data'
    if os.path.isfile(os.path.join(out_dir, "kitti.tfrecords")):
        os.remove(os.path.join(out_dir, "kitti.tfrecords"))

    # bin_indices = map(lambda x: os.path.basename(x).split('.')[0], bin_id)
    writer = tf.python_io.TFRecordWriter(os.path.join(out_dir, "kitti.tfrecords"))
    #for i, idx in enumerate(bin_indices):
    for i in range(100):
        bin_path, label_path = get_data_paths('007480', data_dir)
        print("final path: {} {}".format(bin_path, label_path))
        input, label = create_kitti(bin_path, label_path, width, height, in_channel, lab_channel, range_, maxh, minh)
        example = _convert_to_example(input, label)
    # if i % 100 == 0:
    #     print("%i files are processed" % i)
        writer.write(example.SerializeToString())
    writer.close()
    print("create done!!!")

def creat_test(width, height, in_channel, lab_channel, range_, maxh, minh):
    data_dir = '/home/users/tongyao.bai/data/kitti/testing/velodyne'
    bin_id = glob.glob('%s/*.bin'%(data_dir))
    out_dir = '/home/bai/Project/cnn_seg/data'
    if os.path.isfile(os.path.join(out_dir, "test.tfrecords")):
        os.remove(os.path.join(out_dir, "test.tfrecords"))
    bin_indices = map(lambda x: os.path.basename(x).split('.')[0], bin_id)
    writer = tf.python_io.TFRecordWriter(os.path.join(out_dir, "test.tfrecords"))
    for i, idx in enumerate(bin_indices):
        bin_path, label_path = get_data_paths(idx, data_dir)
        input, label = create_kitti(bin_path, label_path, width, height, in_channel, lab_channel, range_, maxh, minh)
        example = _convert_to_example(input, label)
        if i % 100 == 0:
            print("%i files are processed" % i)
        writer.write(example.SerializeToString())
    writer.close()

if __name__ == "__main__":
    start(640, 640, 8, 12, 60, 5, -5)
    creat_test(640, 640, 8, 12, 60, 5, -5)