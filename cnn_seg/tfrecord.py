import tensorflow as tf
import os
import glob
from data import data_provider, generator_input, gt_label
from multiprocessing import Pool, Process, connection
import traceback
import numpy as np
from tqdm import tqdm

data_dir = '/home/bai/kitti/2011_09_26/2011_09_26_drive_0011_sync/velodyne_points/data'
out_dir = '/home/bai/Project/cnn_seg/dataset'
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
    # label = self.data_provider.generator_label(label_path)
    return input#, label

def get_data_paths(bin_id, data_dir):
    #image_id = '006961'
    bin_file = '{}/{}.bin'.format(data_dir, bin_id)
    # label_file = os.path.join(data_dir, "{}.txt".format(bin_id))
    print(bin_file)
    return bin_file

def _convert_to_example(input, label):
    # all_cats = cats.tolist()
    feature = input.flatten().tostring()
    # gt = label.flatten().tostring()
    example = tf.train.Example(features=tf.train.Features(feature={
        'input': _bytes_feature(feature)
        # 'label': _bytes_feature(gt)
    }))
    return example

def prepare2(bin_id, i):
    option = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.ZLIB)
    writer = tf.python_io.TFRecordWriter(os.path.join(out_dir, "train-{}.tfrecords".format(i)), options=option)
    for path in tqdm(bin_id):
        input_path = get_data_paths(path, data_dir)
        print("final path: {}".format(input_path))
        input = generator_input(input_path, 640, 640,8,12,5,-5)
        # label = gt_label(label_path, 640, 640, 12)
        feature = input.flatten().tostring()
        print(type(feature))
        # lab = label.flatten().tostring()
        input_path = input_path.encode()
        example = tf.train.Example(features=tf.train.Features(feature={
            'input': _bytes_feature(feature),
            'name': _bytes_feature(input_path)
        }))

        writer.write(example.SerializeToString())
        writer.flush()
    writer.close()




    # def __getstate__(self):
    #     self_dict = self.__dict__.copy()
    #     del self_dict['pool']
    #     return self_dict
    #
    # def __setstate__(self, state):
    #     self.__dict__.update(state)

    # def creat_test(self, width, height, in_channel, lab_channel, range_, maxh, minh):pass



def start(num):
    bin_id = glob.glob('%s/*.bin' % (data_dir))
    pool = Pool(num)
    bin_indices = list(map(lambda x: os.path.basename(x).split('.')[0], bin_id))
    length = len(bin_indices)
    batch = int(length / num)
    # batchs = np.arange(length)
    slice = np.arange(length)[0:length:batch]
    for i in range(num):
        if i ==num : x = bin_indices[slice[i]:]
        else: x = bin_indices[slice[i]:slice[i+1]]
        a = pool.apply_async(prepare2, (x, i))
        print(a.get())
    pool.close()
    pool.join()
    print("create done!!!")


if __name__ == "__main__":
    # tfs = record(0,0, 640, 640, 8, 12, 60, 4)
    start(1)
    #creat_test(640, 640, 8, 12, 60, 5, -5)