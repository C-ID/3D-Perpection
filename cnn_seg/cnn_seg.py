import tensorflow as tf
import numpy as np





def net(input, batch_size, width, height, test=False):

    conv = tf.layers.conv2d(input, 24, 1, strides=(1,1), padding="VALID", activation=tf.nn.relu, name="conv0_1")
    conv = tf.layers.conv2d(conv, 24, 3, strides=(1,1), padding="SAME", activation=tf.nn.relu, name="conv0")
    conv = tf.layers.conv2d(conv, 48, 3, strides=(2,2), padding="SAME", activation=tf.nn.relu, name="conv1_1")
    conv_1 = tf.layers.conv2d(conv, 48, 3, strides=(1, 1), padding="SAME", activation=tf.nn.relu, name="conv1")
    conv = tf.layers.conv2d(conv_1, 64, 3, strides=(2, 2), padding="SAME", activation=tf.nn.relu, name="conv2_1")
    conv = tf.layers.conv2d(conv, 64, 3, strides=(1, 1), padding="SAME", activation=tf.nn.relu, name="conv2_2")
    conv_2 = tf.layers.conv2d(conv, 64, 3, strides=(1, 1), padding="SAME", activation=tf.nn.relu, name="conv2")
    conv = tf.layers.conv2d(conv_2, 96, 3, strides=(2, 2), padding="SAME", activation=tf.nn.relu, name="conv3_1")
    conv = tf.layers.conv2d(conv, 96, 3, strides=(1, 1), padding="SAME", activation=tf.nn.relu, name="conv3_2")
    conv_3 = tf.layers.conv2d(conv, 96, 3, strides=(1, 1), padding="SAME", activation=tf.nn.relu, name="conv3")
    conv = tf.layers.conv2d(conv_3, 128, 3, strides=(2, 2), padding="SAME", activation=tf.nn.relu, name="conv4_1")
    conv = tf.layers.conv2d(conv, 128, 3, strides=(1, 1), padding="SAME", activation=tf.nn.relu, name="conv4_2")
    conv_4 = tf.layers.conv2d(conv, 128, 3, strides=(1, 1), padding="SAME", activation=tf.nn.relu, name="conv4")
    conv = tf.layers.conv2d(conv_4, 192, 3, strides=(2, 2), padding="SAME", activation=tf.nn.relu, name="conv5_1")
    conv = tf.layers.conv2d(conv, 192, 3, strides=(1, 1), padding="SAME", activation=tf.nn.relu, name="conv5")
    conv = tf.layers.conv2d(conv, 192, 3, strides=(1, 1), padding="SAME", activation=tf.nn.relu, name="deconv5_1")
    conv = tf.layers.conv2d_transpose(conv, 128, 4, strides=(2,2), padding="SAME", activation=tf.nn.relu, name="deconv4")
    conv_cat4 = tf.concat([conv, conv_4], axis=3, name="Concat_concat4")
    conv = tf.layers.conv2d_transpose(conv_cat4, 128, 3, strides=(1, 1), padding="SAME", activation=tf.nn.relu,
                                      name="deconv4_1")
    conv = tf.layers.conv2d_transpose(conv, 96, 4, strides=(2, 2), padding="SAME", activation=tf.nn.relu,
                                      name="deconv3")
    conv_cat3 = tf.concat([conv, conv_3], axis=3, name="Concat_concat3")
    conv = tf.layers.conv2d_transpose(conv_cat3, 96, 3, strides=(1, 1), padding="SAME", activation=tf.nn.relu,
                                      name="deconv3_1")
    conv = tf.layers.conv2d_transpose(conv, 64, 4, strides=(2, 2), padding="SAME", activation=tf.nn.relu,
                                      name="deconv2")
    conv_cat2 = tf.concat([conv, conv_2], axis=3, name="Concat_concat2")
    conv = tf.layers.conv2d_transpose(conv_cat2, 64, 3, strides=(1, 1), padding="SAME", activation=tf.nn.relu,
                                      name="deconv2_1")
    conv = tf.layers.conv2d_transpose(conv, 48, 4, strides=(2, 2), padding="SAME", activation=tf.nn.relu,
                                      name="deconv1")
    conv_cat1 = tf.concat([conv, conv_1], axis=3, name="Concat_concat1")
    conv = tf.layers.conv2d_transpose(conv_cat1, 48, 3, strides=(1, 1), padding="SAME", activation=tf.nn.relu,
                                      name="deconv1_1")
    conv_final = tf.layers.conv2d_transpose(conv, 12, 4, strides=(2, 2), padding="SAME", activation=tf.nn.relu,
                                      name="deconv0")


    #slice feature map to each size
    category_pt = tf.slice(conv_final, [0,0,0,0], [batch_size, width, height, 1])
    instacnce_pt = tf.slice(conv_final, [0, 0, 0, 1], [batch_size, width, height, 2])
    confidence_pt = tf.slice(conv_final, [0, 0, 0, 3], [batch_size, width, height, 1])
    classify_pt = tf.slice(conv_final, [0, 0, 0, 4], [batch_size, width, height, 5])
    heading_pt = tf.slice(conv_final, [0, 0, 0, 9], [batch_size, width, height, 2])
    height_pt = tf.slice(conv_final, [0, 0, 0, 11], [batch_size, width, height, 1])
    mask = tf.slice(input, [0,0,0,6], [batch_size, width, height, 1])


    if test:
        all_category_score = tf.sigmoid(category_pt, name="all_category_score")
        category_score = tf.multiply(mask, all_category_score, name="instance_ignore_layer")
        confidence_score = tf.sigmoid(confidence_pt, name="confidence_score")
        class_score = tf.sigmoid(classify_pt, name="class_score")

        return instacnce_pt, height_pt, class_score, confidence_score, category_score

    return conv_final




def loss(pre, gt):
    pass



if __name__ == "__main__":
    input = np.ones((1, 640, 640, 8))
    sess = tf.Session()
    x = tf.placeholder(dtype=tf.float32, shape=[None, 640, 640, 8],name="input")
    y = tf.placeholder(dtype=tf.float32, shape=[None, 12], name="output")
    a, b, c, d, e = net(x, 1, 640, 640, test=True)
    sess.run(tf.global_variables_initializer())
    print(sess.run(tf.shape(e), feed_dict={x:input}))