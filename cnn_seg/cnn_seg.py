import tensorflow as tf
import numpy as np


def net(inputs, batch_size, width, height, istest=False):
    
    conv = tf.layers.conv2d(inputs, 24, 1, strides=(1,1), padding="VALID", activation=tf.nn.relu, name="conv0_1")
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
    for i in range(batch_size):
        category_p = tf.slice(conv_final, [i, 0, 0, 0], [1, width, height, 1])
        instance_p = tf.slice(conv_final, [i, 0, 0, 1], [1, width, height, 2])
        confidence_p = tf.slice(conv_final, [i, 0, 0, 3], [1, width, height, 1])
        classify_p = tf.slice(conv_final, [i, 0, 0, 4], [1, width, height, 5])
        heading_p = tf.slice(conv_final, [i, 0, 0, 9], [1, width, height, 2])
        height_p = tf.slice(conv_final, [i, 0, 0, 11], [1, width, height, 1])
        if i == 0: 
            category_pt = category_p
            instance_pt = instance_p
            confidence_pt = confidence_p
            classify_pt = classify_p
            heading_pt = heading_p
            height_pt = height_p
        else:
            category_pt = tf.concat([category_pt, category_p], 0)
            instance_pt = tf.concat([instance_pt, instance_p], 0)
            confidence_pt = tf.concat([confidence_pt, confidence_p], 0)
            classify_pt = tf.concat([classify_pt, classify_p], 0)
            heading_pt = tf.concat([heading_pt, heading_p], 0)
            height_pt = tf.concat([height_pt, height_p], 0)

    if istest:
        mask = tf.split(inputs, [7, 1], 3)
        all_category_score = tf.sigmoid(category_pt, name="all_category_score")
        category_score = tf.multiply(mask[1], all_category_score, name="instance_ignore_layer")
        confidence_score = tf.sigmoid(confidence_pt, name="confidence_score")
        class_score = tf.sigmoid(classify_pt, name="class_score")
        return instance_pt, height_pt, class_score, confidence_score, category_score

    return category_pt, instance_pt, confidence_pt, classify_pt, heading_pt, height_pt




def computeloss(category_pt, instance_pt, confidence_pt, classify_pt, heading_pt, \
                height_pt, gt, batch_size, shape, istrain=True):

    #slice gt
    for i in range(batch_size):
        obj_ = tf.slice(gt, [i,0,0,0], [1, shape[0], shape[1],1])
        ins_ = tf.slice(gt, [i,0,0,1], [1, shape[0], shape[1],2])
        conf_ = tf.slice(gt, [i,0,0,3], [1, shape[0],shape[1],1])
        cla_ = tf.slice(gt, [i,0,0,4], [1, shape[0], shape[1],5])
        hea_ = tf.slice(gt, [i,0,0,9], [1, shape[0], shape[1],2])
        heig_ = tf.slice(gt, [i,0,0,11], [1, shape[0], shape[1],1])
        if i==0:
            obj = obj_
            ins = ins_
            conf = conf_
            cla = cla_
            hea = hea_
            heig = heig_
        else:
            obj = tf.concat([obj, obj_], 0)
            ins = tf.concat([ins, ins_], 0)
            conf = tf.concat([conf, conf_], 0)
            cla = tf.concat([cla, cla_], 0)
            hea = tf.concat([hea, hea_], 0)
            heig = tf.concat([heig, heig_], 0)
            
    #coarse segment
    #objectness loss
    category_pt = tf.reshape(category_pt, [-1, 1])
    obj = tf.reshape(obj, [-1, 1])

    objectness_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=category_pt, labels=obj))
    # objectness_loss = tf.reduce_mean(objectness_loss + objectness_loss * obj * 6)

    #instance loss
    # instance_pt = tf.multiply(instance_pt, conf)
#     ins = ins * 2
    instance_loss = tf.losses.mean_squared_error(labels=ins, predictions=instance_pt)

    #classify loss
    cla = tf.reshape(cla, [-1, 5])
    classify_pt = tf.reshape(classify_pt, [-1, 5])

    category_pt_ = tf.reshape(category_pt, [-1])
    category_pt_ = category_pt_ >= 0.5
    category_pt_ = tf.cast(category_pt_, tf.float32)

    conf_ = tf.reshape(conf, [-1])
    intersection_num = category_pt_ == conf_
    intersection_num = tf.reduce_sum(tf.cast(intersection_num, tf.float32))

    classify_loss = tf.nn.softmax_cross_entropy_with_logits(labels=cla, logits=classify_pt)
    classify_loss = tf.multiply(classify_loss, category_pt_)
    classify_loss = tf.multiply(classify_loss, conf_)
    classify_loss = tf.reduce_mean(tf.reduce_sum(tf.reshape(classify_loss, [batch_size,-1]), axis=1) / (intersection_num+1e-10))

    #height loss
    height_pt = tf.multiply(height_pt, conf)
    height_loss = tf.losses.mean_squared_error(labels=heig, predictions=height_pt)

    #confidence loss
    confidence_pt = tf.reshape(confidence_pt, [-1, 1])
    conf = tf.reshape(conf, [-1, 1])
    confidence_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=confidence_pt, labels=conf))
    # confidence_loss = tf.reduce_mean(confidence_loss + confidence_loss * conf * 6)

    heading_loss = tf.losses.mean_squared_error(labels=hea, predictions=heading_pt)


    total_loss = objectness_loss + instance_loss + height_loss + confidence_loss
    summary(objectness_loss, instance_loss, confidence_loss, classify_loss, heading_loss, height_loss, total_loss, istraining=istrain)
    return total_loss

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
    x = tf.placeholder(dtype=tf.float32, shape=[None, None, None, 8],name="input")
    y = tf.placeholder(dtype=tf.float32, shape=[None, None, None, 12], name="output")

    a, b, c, d, e, f = net(x, 1, 320, 640, istest=False)
    lo, objectness_loss = computeloss(a, b, c, d, e, f, y, 1, True)
    opt = tf.train.AdamOptimizer(learning_rate=0.01)
    train_op = opt.compute_gradients(loss=lo)
    sess.run(tf.global_variables_initializer())
    for i in range(10):
        d, _ = sess.run([tf.shape(objectness_loss), train_op], feed_dict = {x:input, y:label})
        print(d)
