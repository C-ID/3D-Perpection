import tensorflow as tf
import os
import glob
from data import *
from multiprocessing import Pool, Process
import traceback
import numpy as np
from tqdm import tqdm
import time
data_dir = '/home/users/tongyao.bai/tongyao.bai/hobot_lidar/lidar'
label_dir = '/home/users/tongyao.bai/tongyao.bai/hobot_lidar/labels'
out_dir = '/home/users/tongyao.bai/tongyao.bai/hobot_record'
# class record(object):
#
#     def __init__(self, train_path, test_path, width, height, in_channel, lab_channel, range_, num_thread):
#         # self.data_provider = data_provider(width,height,in_channel,lab_channel,range_)
#         self.train = train_path
#         self.test = test_path
#         self.pool = Pool(num_thread)
#         self.num_thread = num_thread

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


def create_kitti(input_path):
    input = data_provider.generator_input(input_path)
    label = self.data_provider.generator_label(label_path)
    return input#, label

def get_data_paths(bin_id, data_dir):
    #image_id = '006961'
    bin_file = '{}/{}.pkl'.format(data_dir, bin_id)
    label_file = os.path.join(label_dir, "{}.txt".format(bin_id))
#     print(bin_file, label_file)
    return bin_file, label_file

def _convert_to_example(input, label):
    # all_cats = cats.tolist()
    feature = input.flatten().tostring()
    gt = label.flatten().tostring()
    example = tf.train.Example(features=tf.train.Features(feature={
        'input': _bytes_feature(feature),
        'label': _bytes_feature(gt)
    }))
    return example

def prepare2(bin_id, i):
    option = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.ZLIB)
    writer = tf.python_io.TFRecordWriter(os.path.join(out_dir, "train-{}.tfrecords".format(i)), options=option)
    cnt = 0
    for path in bin_id:
        input_path, label_path = get_data_paths(path, data_dir)
        name = os.path.basename(input_path).split('.')[0]
        name = name.encode()
        print("final path: {} {}, PID:{}".format(os.path.basename(input_path), os.path.basename(label_path), i))
#         inputs = generator_input(input_path, 640, 640, 8, 60, 5, -5)
#         inputs_ = inputs[:320,:,:]
#         label = gt_label(label_path, 640, 640, 12)
#         label_ = label[:320, :, :]
        normal, trans, zoom, rotate = process(input_path, label_path, False)
        label = normal[1] 
#         if label[:,:,0].sum() == 0:
#             continue
        cnt += 1
        features = [normal, trans, zoom, rotate]
        for data in features:
            fea, lab = data[0], data[1]
#             fea = fea[:320,:,:]
#             lab = lab[:320,:,:]
            fea = fea.flatten().tostring()
            lab = lab.flatten().tostring()
            
            example = tf.train.Example(features=tf.train.Features(feature={
                'input': _bytes_feature(fea),
                'label': _bytes_feature(lab),
                'name' : _bytes_feature(name)
            }))
            writer.write(example.SerializeToString())
            writer.flush()
    print("{} ID process: {} frames".format(i, cnt*4))       
    writer.close()
    


def start(num):
    bin_id = glob.glob('%s/*.pkl' % (data_dir))
    pool = Pool(num)
    
    bin_indices = list(map(lambda x: os.path.basename(x).split('.')[0], bin_id))
    length = len(bin_indices)
    batch = int(length / num)
#     batchs = np.arange(length)
    slices = np.arange(length)[0:length:batch]
#     print(length, slices)
    for i in range(num):
#         if i == num-1 : x = bin_indices[slice[i]:]
        x = bin_indices[slices[i]:slices[i]+batch]
#         print(len(x))
        pool.apply_async(prepare2, (x, i))
#         print(a.get())
    pool.close()
    pool.join()
    print("process done!!!")


if __name__ == "__main__":
#     prepare2('007480', 1)
    t1 = time.time()
    start(10)
    print(time.time() - t1)