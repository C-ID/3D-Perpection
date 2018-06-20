import tensorflow as tf
import numpy as np


def net(input, batch_size, width, height, istest=False):
    
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
    conv_final = tf.layers.conv2d_transpose(conv, 12, 4, strides=(2, 2), padding="SAME", activation=None,
                                      name="deconv0")


    #slice feature map to each size
    category_pt = tf.slice(conv_final, [0,0,0,0], [batch_size, width, height, 1])
    instacnce_pt = tf.slice(conv_final, [0, 0, 0, 1], [batch_size, width, height, 2])
    confidence_pt = tf.slice(conv_final, [0, 0, 0, 3], [batch_size, width, height, 1])
    classify_pt = tf.slice(conv_final, [0, 0, 0, 4], [batch_size, width, height, 5])
    heading_pt = tf.slice(conv_final, [0, 0, 0, 9], [batch_size, width, height, 2])
    height_pt = tf.slice(conv_final, [0, 0, 0, 11], [batch_size, width, height, 1])

    if istest:
        mask = tf.slice(input, [0, 0, 0, 6], [batch_size, width, height, 1])
        all_category_score = tf.sigmoid(category_pt, name="all_category_score")
        category_score = tf.multiply(mask, all_category_score, name="instance_ignore_layer")
        confidence_score = tf.sigmoid(confidence_pt, name="confidence_score")
        class_score = tf.sigmoid(classify_pt, name="class_score")

        return instacnce_pt, height_pt, class_score, confidence_score, category_score

    return category_pt, instacnce_pt, confidence_pt, classify_pt, heading_pt, height_pt




def computeloss(category_pt, instance_pt, confidence_pt, classify_pt, heading_pt, height_pt, gt, batch_size, istrain=True):

    obj = tf.slice(gt, [0,0,0,0], [batch_size,640,640,1])
    ins = tf.slice(gt, [0,0,0,1], [batch_size,640,640,2])
    conf = tf.slice(gt, [0,0,0,3], [batch_size, 640,640,1])
    cla = tf.slice(gt, [0,0,0,4], [batch_size, 640,640,5])
    hea = tf.slice(gt, [0,0,0,9], [batch_size, 640,640,2])
    heig = tf.slice(gt, [0,0,0,11], [batch_size, 640,640,1])
    
    
    #coarse segment
    category_pt = tf.reshape(category_pt, [batch_size*640*640, 1])
    obj = tf.reshape(obj, [batch_size*640*640, 1])
    
    #fine segment
#     classify_unknow = tf.reshape(classify_pt[:,:,:,0], [batch_size*640*640, 1])
#     classify_smallcar = tf.reshape(classify_pt[:,:,:,1], [batch_size*640*640, 1])
#     classify_bigcar = tf.reshape(classify_pt[:,:,:,2], [batch_size*640*640, 1])
#     classify_bicycle = tf.reshape(classify_pt[:,:,:,3], [batch_size*640*640, 1])
#     classify_person = tf.reshape(classify_pt[:,:,:,4], [batch_size*640*640, 1])
#     classify_pt = tf.reshape(classify_pt, [batch_size*640*640, 5])
    
    
    cla_unknow = tf.reshape(cla[:,:,:,0], [batch_size*640*640]) *  0
    cla_smallcar = tf.reshape(cla[:,:,:,1], [batch_size*640*640]) * 1
    cla_bigcar = tf.reshape(cla[:,:,:,2], [batch_size*640*640]) * 2
    cla_bicycle = tf.reshape(cla[:,:,:,3], [batch_size*640*640]) * 3
    cla_person = tf.reshape(cla[:,:,:,4], [batch_size*640*640]) * 4
    cla_six = cla_unknow + cla_smallcar + cla_bigcar + cla_bicycle + cla_person
    
    
    
    objectness_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=category_pt, labels=obj))
#     objectness_loss = tf.reduce_mean(tf.multiply(tf.reshape(objectness_loss,[batch_size*640*640, 1]), obj) + tf.reshape(objectness_loss,[batch_size*640*640, 1]))
    
    instance_loss = tf.losses.mean_squared_error(labels=ins, predictions=instance_pt)
    
    confidence_loss = tf.losses.mean_squared_error(labels=conf, predictions=confidence_pt)
    classify_loss = tf.losses.mean_squared_error(labels=cla, predictions=classify_pt)
    classify_loss = tf.reduce_sum(tf.multiply(classify_loss, tf.reshape(conf, [-1]))) / tf.reduce_sum(conf)
    
    
    heading_loss = tf.losses.mean_squared_error(labels=hea, predictions=heading_pt)
    
    height_loss = tf.losses.mean_squared_error(labels=heig, predictions=height_pt)
    
#     total_loss = instance_loss
    total_loss = objectness_loss + instance_loss + confidence_loss + classify_loss + heading_loss + height_loss
    summary(objectness_loss, instance_loss, confidence_loss, classify_loss, heading_loss, height_loss, total_loss, istraining=istrain)
    return total_loss, classify_pt, classify_loss

def summary(objectness_loss, instance_loss, confidence_loss, classify_loss, heading_loss, height_loss, total_loss, istraining=True):
    if istraining:
        tf.summary.scalar('training-loss/objectness_loss',objectness_loss)
        tf.summary.scalar('training-loss/instance_loss', instance_loss)
        tf.summary.scalar('training-loss/confidence_loss', confidence_loss)
        tf.summary.scalar('training-loss/classify_loss', classify_loss)
        tf.summary.scalar('training-loss/heading_loss', heading_loss)
        tf.summary.scalar('training-loss/height_loss', height_loss)
        tf.summary.scalar('training-loss/total_loss', total_loss)
    else:
        tf.summary.scalar('val-loss/total_loss', total_loss)


if __name__ == "__main__":
    input = np.ones((1, 640, 640, 8))
    label = np.ones((1, 640, 640, 12))
    sess = tf.Session()
    x = tf.placeholder(dtype=tf.float32, shape=[None, 640, 640, 8],name="input")
    y = tf.placeholder(dtype=tf.float32, shape=[None, 640, 640, 12], name="output")

    a, b, c, d, e, f = net(x, 1, 640, 640, istest=False)
    lo, objectness_loss = computeloss(a, b, c, d, e, f, y, 1, True)
    opt = tf.train.AdamOptimizer(learning_rate=0.01)
    train_op = opt.compute_gradients(loss=lo)
    sess.run(tf.global_variables_initializer())
    for i in range(10):
        d, _ = sess.run([tf.shape(objectness_loss), train_op], feed_dict = {x:input, y:label})
        print(d)