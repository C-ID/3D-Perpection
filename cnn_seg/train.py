import tensorflow as tf
import numpy as np
import os
from cnn_seg import net, computeloss
import time
import logging
import glob
import pdb
from render import *
from diff_record import *

log = logging.getLogger()
os.environ["CUDA_VISIBLE_DEVICES"]= "2"
class apollo(object):
    def __init__(self, train_tfrecord, val_tfrecord, learning_rate, global_step, train_dir, batch_size, epoch, shape):
        
        self.train_tfrecord = train_tfrecord
        self.val_tfrecord = val_tfrecord
        self.train_dir = train_dir
        self.learning_rate = learning_rate
        self.global_step = global_step
        self.batch_size = batch_size
        self.epoch = epoch
        self.shape = shape
        
    def train(self):
        inp, label, name = self.read_tfrecord(self.train_tfrecord)
        image_batch = tf.cast(inp, tf.float32)
        label_batch = tf.cast(label, tf.float32)
        
        #construct network graph
        feature = tf.placeholder(tf.float32, [None, None, None, 8], name="input-feature")
        unknow_label = tf.placeholder(tf.float32, [None, None, None, 12], name="input-label")
        category_pt, instacnce_pt, confidence_pt, classify_pt, heading_pt, height_pt = \
                        net(feature, self.batch_size, self.shape[0], self.shape[1],istest=False)
        train_loss = computeloss(category_pt, instacnce_pt, \
                        confidence_pt, classify_pt, heading_pt, height_pt, unknow_label, self.batch_size, self.shape)

        #construct optimizer
        global_step_ = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(self.learning_rate,
                                         global_step=global_step_,
                                         decay_steps=75000,
                                         decay_rate=0.1,
                                         staircase=True)
        train_op = tf.train.AdamOptimizer(learning_rate).minimize(train_loss, global_step=global_step_)

        summary_op = tf.summary.merge_all()
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=50, keep_checkpoint_every_n_hours=1)

        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                              log_device_placement=False)) as sess:
            summary_writer = tf.summary.FileWriter(self.train_dir, sess.graph)
            init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
            sess.run(init_op)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            try:
                for i in range(self.epoch):
                    for step in range(self.global_step):
                        start_time = time.time()
                        image_batch_in, label_batch_in, in_name= sess.run([image_batch, label_batch, name])
                        loss, _, summary_str, lr = sess.run([train_loss, train_op, summary_op, learning_rate],\
                                       feed_dict={ feature: image_batch_in, unknow_label: label_batch_in })
                        duration = time.time() - start_time
                        num_examples_per_step = self.batch_size
                        examples_per_sec = num_examples_per_step / duration
                        sec_per_batch = float(duration)
                        format_str = ('step: {}, loss={}, lr={} ({} examples/sec; {} '
                                      'sec/batch)')
                        print(format_str.format(i*self.global_step+step, loss, lr, examples_per_sec, sec_per_batch))
                        summary_writer.add_summary(summary_str, i*self.global_step+step)
                        summary_writer.flush()
                        if step % 274 == 0 and step > 0:
                            checkpoint_path = os.path.join(self.train_dir, 'cnn_seg-{}.ckpt'.format(i))
                            saver.save(sess, checkpoint_path, global_step=step)
                summary_writer.close()
            except tf.errors.OutOfRangeError:
                print("Training Completed!!!")
            finally:
                coord.request_stop()
                coord.join(threads)
    def confirm_record(self):
        inp, label, name = self.read_tfrecord(self.train_tfrecord)
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,log_device_placement=False)) as sess:
                init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
                sess.run(init_op)
                coord = tf.train.Coordinator()
                threads = tf.train.start_queue_runners(coord=coord)
                try:
                    image_batch_in, label_batch_in, in_name= sess.run([inp, label, name])
                    for i in range(8):
                        basename = in_name[i].decode().split('.')[0]
                        a = image_batch_in[i,:,:,:]
                        b = label_batch_in[i,:,:,:]
                        print(a.shape, b.shape)
                        record_confirm(a[np.newaxis, :], b[np.newaxis, :], basename)
                except tf.errors.OutOfRangeError:
                    print("things done")

                finally:
                    coord.request_stop()
                    coord.join(threads)
                    
    def __call__(self, *args, **kwargs):
        pass

    def read_tfrecord(self, filename):
        filename_queue = tf.train.string_input_producer(filename, num_epochs=None)
        option = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.ZLIB)
        reader = tf.TFRecordReader(options=option)
        _, serialized_example = reader.read(filename_queue)

        features = tf.parse_single_example(
            serialized_example,
            features={
                'input': tf.FixedLenFeature([], tf.string),
                'label': tf.FixedLenFeature([], tf.string),
                'name' : tf.FixedLenFeature([], tf.string)
            })

        inputs = tf.decode_raw(features['input'], tf.float64)
        label = tf.decode_raw(features['label'], tf.float64)
        name = features['name']
        
        inputs = tf.reshape(inputs, [self.shape[0], self.shape[1], 8])
        label = tf.reshape(label, [self.shape[0], self.shape[1], 12])
        
        inp, label, name = tf.train.shuffle_batch([inputs, label, name], num_threads=4,
                batch_size=self.batch_size,
                capacity=32,
                min_after_dequeue=8
                )
        return inp, label, name
    

if __name__ == "__main__":
    record_path = "/home/users/tongyao.bai/tongyao.bai/hobot_record" 
    records = glob.glob('%s/*.tfrecords'%(record_path))
    #tfrecord = os.path.join(os.getcwd(), "../../data/tongyao.bai/kitti.tfrecords")
    train_dir = "/home/users/tongyao.bai/tongyao.bai/hb_full640"
    train_records = records   #2360 frame
    vali_records = records[:-2]   #1496 frame
    shape = [640, 640]
    if not os.path.exists(train_dir):
        os.mkdir(train_dir)
    apollo = apollo(train_records, vali_records, 0.0001, 275, train_dir, 8, 150, shape)
    apollo.train()
#23636
