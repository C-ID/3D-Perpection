import tensorflow as tf
import numpy as np
import os
from cnn_seg import net, computeloss
import time
import logging

log = logging.getLogger()

class apollo(object):
    def __init__(self, train_tfrecord, test_tfrecord, learning_rate, global_step, train_dir):
        self.train_tfrecord = train_tfrecord
        self.test_tfrecord = test_tfrecord
        self.train_dir = train_dir

    def train(self):
        input, label = self.read_tfrecord(self.train_tfrecord)
        category_pt, instacnce_pt, confidence_pt, classify_pt, heading_pt, height_pt=net(input,16,640,640,istest=False)
        total_loss = computeloss(category_pt, instacnce_pt, confidence_pt, classify_pt, heading_pt, height_pt, label)
        global_step = 1000000
        learing_rate = 0.01
        opt = tf.train.AdamOptimizer(learing_rate)
        train_op = opt.compute_gradients(total_loss)
        summary_op = tf.summary.merge_all()
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=1000, keep_checkpoint_every_n_hours=1)

        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                              log_device_placement=False)) as sess:
            summary_writer = tf.summary.FileWriter(self.train_dir, sess.graph)
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())

            for step in  range(global_step):
                start_time = time.time()
                try:
                    loss = sess.run([total_loss])
                except (tf.errors.OutOfRangeError, tf.errors.CancelledError):
                    break
                duration = time.time() - start_time
                num_examples_per_step = input.shape[0]
                examples_per_sec = num_examples_per_step / duration
                sec_per_batch = float(duration)
                format_str = ('step %d, loss = %.2f, lr=%.3f (%.1f examples/sec; %.3f '
                              'sec/batch)')
                log.info(format_str % (step, loss, -np.log10(learing_rate),
                                       examples_per_sec, sec_per_batch))
                if step % 100 == 0:
                    summary_str = sess.run(summary_op)
                    summary_writer.add_summary(summary_str, step)

                if step % 1000 == 0 and step > 0:
                    summary_writer.flush()
                    log.debug("Saving checkpoint...")
                    checkpoint_path = os.path.join(self.train_dir, 'cnn_seg.ckpt')
                    saver.save(sess, checkpoint_path, global_step=step)

            summary_writer.close()

        return input

    def test(self):
        input, label = self.read_tfrecord(self.test_tfrecord)

    def __call__(self, *args, **kwargs):
        pass

    def read_tfrecord(self, filename):
        filename_queue = tf.train.string_input_producer([filename])
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)

        features = tf.parse_single_example(
            serialized_example,
            features={
                'input': tf.FixedLenFeature([], tf.string),
                'label': tf.FixedLenFeature([], tf.string)
            })

        input = tf.decode_raw(features['input'], tf.float32)
        label = tf.decode_raw(features['label'], tf.float32)

        input = tf.reshape(input, [640, 640, 8])
        label = tf.reshape(label, [640, 640, 12])

        input, label = tf.train.shuffle_batch([input, label],num_threads=4,
                batch_size=16,
                capacity=100,
                min_after_dequeue=50)
        return input, label

if __name__ == "__main__":
    tfrecord = os.path.join(os.getcwd(), "dataset/kitti.tfrecords")
    train_dir = os.path.join(os.getcwd(), "train_dir")
    if not os.path.exists(train_dir):
        os.mkdir(train_dir)
    apollo = apollo(tfrecord, tfrecord, 0.01, 1000000, train_dir)
    input = apollo.train()
    print(input)
