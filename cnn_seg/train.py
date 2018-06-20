import tensorflow as tf
import numpy as np
import os
from cnn_seg import net, computeloss
import time
import logging
import glob

log = logging.getLogger()
os.environ["CUDA_VISIBLE_DEVICES"]= "1" 
class apollo(object):
    def __init__(self, train_tfrecord, val_tfrecord, learning_rate, global_step, train_dir, batch_size, epoch):
        
        self.train_tfrecord = train_tfrecord
        self.val_tfrecord = val_tfrecord
        self.train_dir = train_dir
        self.learning_rate = learning_rate
        self.global_step = global_step
        self.batch_size = batch_size
        self.epoch = epoch
        
    def train(self):
        inp, label = self.read_tfrecord(self.train_tfrecord)
        image_batch = tf.cast(inp, tf.float32)
        label_batch = tf.cast(label, tf.float32)
        #image_batch1 = tf.reshape(image_batch, [self.batch_size, 640, 640, 8])
        #label_batch1 = tf.reshape(label_batch, [self.batch_size, 640, 640, 12])
        
#         val_input, val_label = self.read_tfrecord(self.val_tfrecord)
#         val_mage_batch = tf.cast(inp, tf.float32)
#         val_label_batch = tf.cast(label, tf.float32)
        
#         global_step = self.global_step
#         print(init_learning_rate)
        #construct network graph
        feature = tf.placeholder(dtype=tf.float32, shape=[None, 640, 640, 8], name="input-feature")
        unknow_label = tf.placeholder(dtype=tf.float32, shape=[None, 640, 640, 12], name="input-label")
        category_pt, instacnce_pt, confidence_pt, classify_pt, heading_pt, height_pt = net(feature, self.batch_size, 640, 640,
                                                                                           istest=False)
        train_loss, clas_pt, cla_loss = computeloss(category_pt, instacnce_pt, confidence_pt, classify_pt, heading_pt, height_pt, unknow_label, self.batch_size)
#         val_loss =  computeloss(category_pt, instacnce_pt, confidence_pt, classify_pt, heading_pt, height_pt, unknow_label, self.batch_size, False)

        #construct optimizer
        
        learning_rate = tf.train.exponential_decay(self.learning_rate,
                                         global_step=self.global_step*self.epoch,
                                         decay_steps=1000,
                                         decay_rate=0.96,
                                         staircase=True)
        train_op = tf.train.AdamOptimizer(learning_rate).minimize(train_loss)

        summary_op = tf.summary.merge_all()
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=50, keep_checkpoint_every_n_hours=1)

        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                              log_device_placement=False)) as sess:
            summary_writer = tf.summary.FileWriter(self.train_dir, sess.graph)
            init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
            sess.run(init_op)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            for i in range(self.epoch):
                for step in range(self.global_step):
                    start_time = time.time()
                    try:
                        image_batch_in, label_batch_in= sess.run([image_batch, label_batch])
#                         a, b = sess.run([clas_pt, cla_loss], feed_dict={feature:image_batch_in,
#                        unknow_label:label_batch_in})
                        loss, _, summary_str, lea_rat = sess.run([train_loss, train_op, summary_op, learning_rate], feed_dict={
                            feature:image_batch_in,
                            unknow_label:label_batch_in
                            })
#                         print("sums: ", s) 
                    except tf.errors.OutOfRangeError:
                        break
                    duration = time.time() - start_time
                    num_examples_per_step = self.batch_size
                    examples_per_sec = num_examples_per_step / duration
                    sec_per_batch = float(duration)
                    format_str = ('step: {}, loss={}, lr={} ({} examples/sec; {} '
                                  'sec/batch)')
                    print(format_str.format(i*self.global_step+step, loss, lea_rat, examples_per_sec, sec_per_batch))
#                     print("forward: ", a, b)
                    summary_writer.add_summary(summary_str, i*self.global_step+step)

                    if step % 999 == 0 and step > 0:
                        summary_writer.flush()
                        checkpoint_path = os.path.join(self.train_dir, 'cnn_seg-{}.ckpt'.format(i))
                        saver.save(sess, checkpoint_path, global_step=step)     
#                  for val_step in range(50):
#                     val_image_batch_in, val_label_batch_in= sess.run([val_mage_batch, val_label_batch])
#                     val_loss, summary_val = sess.run([val_loss, summary_op], feed_dict={feature:val_image_batch_in,
#                                                                  unknow_label:val_label_batch_in
#                                                                })
#                     val += val_loss 
#                 summary_writer.add_summary(summary_val, val_step)
#                 if step % 100 == 0 and step > 0:summary_writer.flush()
            summary_writer.close()
        if coord.should_stop():
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
                'label': tf.FixedLenFeature([], tf.string)
            })

        inputs = tf.decode_raw(features['input'], tf.float64)
        label = tf.decode_raw(features['label'], tf.float64)

        inputs = tf.reshape(inputs, [640, 640, 8])
        label = tf.reshape(label, [640, 640, 12])

        inp, label = tf.train.shuffle_batch([inputs, label],num_threads=4,
                batch_size=self.batch_size,
                capacity=32,
                min_after_dequeue=16
                )
        return inp, label

if __name__ == "__main__":
    record_path = "/home/users/tongyao.bai/data/tongyao.bai/train" 
    records = glob.glob('%s/*.tfrecords'%(record_path))
    #tfrecord = os.path.join(os.getcwd(), "../../data/tongyao.bai/kitti.tfrecords")
    train_dir = os.path.join(os.getcwd(), "train_dir/train_dir-4")
    train_records = records
    vali_records = records[:-2] 
    if not os.path.exists(train_dir):
        os.mkdir(train_dir)
    apollo = apollo(train_records, vali_records, 0.0001, 1000, train_dir, 8, 10)
    apollo.train()

