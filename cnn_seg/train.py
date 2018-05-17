import tensorflow as tf
import numpy as np
import os
from cnn_seg import net, computeloss
import time
import logging

log = logging.getLogger()

class apollo(object):
    def __init__(self, train_tfrecord, test_tfrecord, learning_rate, global_step, train_dir, batch_size):
        self.train_tfrecord = train_tfrecord
        self.test_tfrecord = test_tfrecord
        self.train_dir = train_dir
        self.learning_rate = learning_rate
        self.global_step = global_step
        self.batch_size = batch_size

    def train(self):
        input, label = self.read_tfrecord(self.train_tfrecord)
        image_batch = tf.reshape(input, [self.batch_size, 640, 640, 8])
        label_batch = tf.reshape(label, [self.batch_size, 640, 640, 12])
        global_step = self.global_step
        learing_rate = self.learning_rate

        #construct network graph
        feature = tf.placeholder(dtype=tf.float32, shape=[None, 640, 640, 8], name="input-feature")
        unknow_label = tf.placeholder(dtype=tf.float32, shape=[None, 640, 640, 12], name="input-label")
        category_pt, instacnce_pt, confidence_pt, classify_pt, heading_pt, height_pt = net(input, self.batch_size, 640, 640,
                                                                                           istest=False)
        total_loss = computeloss(category_pt, instacnce_pt, confidence_pt, classify_pt, heading_pt, height_pt, label)

        #construct optimizer
        train_op = tf.train.AdamOptimizer(learing_rate).minimize(total_loss)

        summary_op = tf.summary.merge_all()
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=1000, keep_checkpoint_every_n_hours=1)

        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                              log_device_placement=False)) as sess:
            summary_writer = tf.summary.FileWriter(self.train_dir, sess.graph)
            init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
            sess.run(init_op)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            try:
                for step in range(global_step):
                    start_time = time.time()
                    image_batch_in, label_batch_in= sess.run([image_batch, label_batch])
                    loss, _ = sess.run([total_loss, train_op], feed_dict={
                        feature:image_batch_in,
                        unknow_label:label_batch_in
                    })
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
            except tf.errors.OutOfRangeError:
                pass
            finally:
                coord.request_stop()
        coord.request_stop()
        coord.join(threads)

    def test(self):
        input, label = self.read_tfrecord(self.test_tfrecord)

    def __call__(self, *args, **kwargs):
        pass

    def read_tfrecord(self, filename):
        filename_queue = tf.train.string_input_producer([filename], num_epochs=None)
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

        input, label = tf.train.batch([input, label],num_threads=1,
                batch_size=self.batch_size,
                capacity=4,

                # shapes=[[640,640,8], [640,640,12]],
                # allow_smaller_final_batch=True
                )
        return input, label

if __name__ == "__main__":
    tfrecord = os.path.join(os.getcwd(), "dataset/kitti.tfrecords")
    train_dir = os.path.join(os.getcwd(), "train_dir")
    if not os.path.exists(train_dir):
        os.mkdir(train_dir)
    apollo = apollo(tfrecord, tfrecord, 0.01, 10, train_dir, 1)
    # sess = tf.Session()
    # for i in range(1):
    #     a, b = apollo.read_tfrecord(tfrecord)
    #     print(sess.run([a, b]))
    apollo.train()
    # print(input)
