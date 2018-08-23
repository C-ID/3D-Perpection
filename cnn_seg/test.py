import numpy as np
import tensorflow as tf
import os
from extert_cpp import cluster_for_py
import json
from data import generator_input, F2I
import cv2
import glob
from cnn_seg import net
from render import *
import json

import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"]= "3"
DATA_DIR = '/home/users/tongyao.bai/tongyao.bai/hobot_lidar/vaild'

type_color={
            "pedestrian":(255, 128, 128),  # pink
            "bicycle":(0, 0, 255),  # blue
            "car":(0, 255, 0),  # green
            "unknown":(0, 255, 255)  # yellow
            }

def read_output_channel(path):
    with open(path, 'rb') as f: file = json.load(f)
    for channel in file['output']:
        instance_x = np.asarray(channel['instance_pt_x'])
        instance_y = np.asarray(channel['instance_pt_y'])
        category_pt = np.asarray(channel['category_pt'])
        classify_pt = np.asarray(channel['classify_pt'])
        confidence_pt = np.asarray(channel['confidence_pt'])
        height_pt = np.asarray(channel['height_pt'])
    return category_pt, instance_x, instance_y, confidence_pt, height_pt, classify_pt

# _so = os.path.join(os.getcwd(), "extert_cpp/cluster.cpython-36m-x86_64-linux-gnu.so")

class Inference(object):
    def __init__(self, test_path, ckpt_path):
        self.cluster = cluster_for_py.PyCluster()
        self.cluster.init(640,640,60)
        self.confidence_thresh = 0.1
        self.height_thresh = 0.9
        self.min_pts_num = 3
        self.test_path = test_path
        self.ckpt_path = ckpt_path
        self.sess = tf.Session()

    def test(self, bin_path, json_path):
        cat, ins_x, ins_y, confidence, height, cla = read_output_channel(json_path)
        bin = np.fromfile(bin_path, np.float32)
        pt_nums = len(bin)
        print(pt_nums)
        valid_indices = np.arange(0, pt_nums / 4, 1)
        print(len(valid_indices))
        for i in range(5):
            self.cluster.Cluster2Py(cat, ins_x, ins_y, bin, valid_indices, pt_nums, 0.5, False)
            self.cluster.Filter2Py(confidence, height)
            self.cluster.Classify2Py(cla)
            self.cluster.GetObjects2Py(self.confidence_thresh, self.height_thresh, self.min_pts_num)
            ori_obj = self.cluster.Obstacle2Py()
            score = self.cluster.Score2Py()
            name = np.asarray(self.cluster.Typrname2Py())
            cloud = self.cluster.Cloud2py()
            print(len(ori_obj), i, len(score), len(name), len(cloud))
            print("name :",np.char.decode(name))
            self.save(bin_path, score, np.char.decode(name), cloud)
    
    
    
            
    @staticmethod
    def save(bin_, score, name, cloud, basename):
        
        image3 = np.zeros([640, 640, 3])
        channel = generator_input(bin_)
#         channel = channel[:320,:,:]
#         channel = channel[::-1,::-1,:]
        x, y = np.where(channel[:, :, 2] >= 1)
        image3[x, y, :] = (127, 127, 127)
        
        for i in range(len(score)):
            points = np.asarray(cloud[i]).reshape([-1, 4])
            x = F2I(points[:,0], 60, 0.5*640/60)
            y = F2I(points[:,1], 60, 0.5*640/60)
            color = type_color[name[i]]
            pos_x_min = int(x.min())
            pos_y_min = int(y.min())
            pos_x_max = int(x.max())
            pos_y_max = int(y.max())
            if (pos_y_max - pos_y_min) > (pos_x_max - pos_x_min):
                length =  int(pos_y_max - pos_y_min)
                width = int(pos_x_max - pos_x_min)
            else:
                length = int(pos_x_max - pos_x_min)
                width = int(pos_y_max - pos_y_min)
            cv2.rectangle(image3, (int(y.min()), int(x.min())),(int(y.max()), int(x.max())), color, 1)
        cv2.imwrite(os.path.join(os.getcwd(),"inference/forward-test-{}.jpg".format(os.path.basename(basename).split('.')[0])), image3)

    def __call__(self):
        feature = tf.placeholder(dtype=tf.float32, shape=[None, None, None, 8], name="input-feature")
        instance_pt, height_pt, class_score, confidence_score, category_score = net(feature, 1, 640, 640,
                                                                                           istest=True)
        saver = tf.train.Saver()
        if self.ckpt_path:
            saver.restore(self.sess, self.ckpt_path)
        else:
            raise FileNotFoundError
#         cloud_points = np.fromfile(self.test_path, np.float32).reshape([-1, 4])
        cloud_points = np.load(self.test_path, encoding='bytes')[b'points']
        input_x = generator_input(cloud_points)

#         input_x = input_x[::-1,::-1,:]
#         input_x1 = input_x[::-1,::-1,3]
#         input_x2 = input_x[::-1,::-1,6]
#         input_x[:,:,3] = input_x1
        input_x = input_x[np.newaxis, :]
                
        instance, heights, class_scores, confidences, categorys = self.sess.run([instance_pt, height_pt, class_score, confidence_score, category_score], feed_dict = {feature: input_x})
        print(instance.shape)
        in_x = instance[:,:,:,0].flatten().tolist()
        in_y = instance[:,:,:,1].flatten().tolist()
#         print(in_x)
        tmp = [{"instance_x": in_x},
               {"instance_y": in_y}]
        f = open("./tmp.json", 'w')
        json.dump(tmp, f)
        f.close()
        #prepare for pyx api
        cat = categorys.flatten()
        
        ins_x = instance[:,:,:,0].flatten()
        ins_y = instance[:,:,:,1].flatten()
        point_cloud = np.fromfile(self.test_path, np.float32)
        pt_nums = len(point_cloud)
        height = heights.flatten()
        confidence = confidences.flatten()
        valid_indices = np.arange(0, pt_nums / 4, 1)
        cla = class_scores.flatten()

        
        print(640*640)
        categorys = categorys.reshape([640,640])
        x, y = categorys[:,:].max(), categorys[:,:].min() 
        x_, y_ = np.where(categorys[:,:] >= 0.5)
        print("categorys: ", x, y)
        print("num of pixel greater than 0.5: ",len(x_))
        categorys = categorys[:, :] > 0.5
        categorys = categorys.astype(np.uint8) * 65534
        cv2.imwrite(os.path.join(os.getcwd(), \
                  "inference/categorys-{}.png".format(os.path.basename(self.test_path).split('.')[0])), categorys)
   
        self.cluster.Cluster2Py(cat, ins_x, ins_y, point_cloud, valid_indices, pt_nums, 0.5, False)
        self.cluster.Filter2Py(confidence, height)
        self.cluster.Classify2Py(cla)
        self.cluster.GetObjects2Py(self.confidence_thresh, self.height_thresh, self.min_pts_num)
        ori_obj = self.cluster.Obstacle2Py()
        print("ori_obj: ", len(ori_obj))
        score = self.cluster.Score2Py()
        name = np.asarray(self.cluster.Typrname2Py())
        cloud = self.cluster.Cloud2py()
        assert len(score) == len(name) == len(cloud), "pred wrong num of foreground"
        print(len(cloud))
        if len(cloud) != 0:
            self.save(cloud_points, score, np.char.decode(name), cloud, self.test_path)

            


if __name__ == "__main__":
    #for test
#     json_path = os.path.join(os.getcwd(), "dataset/output.json")
#     bin_path = os.path.join(os.getcwd(), "dataset/007480.bin")
#     forward = Inference(json_path, bin_path)
#     forward.test(bin_path, json_path)

    #for inference
#     bin_id = glob.glob('%s/*.bin' % (DATA_DIR))
#     test_path = bin_id[:10]
    test_path = os.path.join(DATA_DIR, "2018-06-19-14-14-55_1_00000864.pkl")
#     ckpt_path = os.path.join(os.getcwd(), "train_dir/lr-epo-6-0.1-instance*2/cnn_seg-7.ckpt-1499")
    ckpt_path = "/home/users/tongyao.bai/tongyao.bai/hb_full640/cnn_seg-62.ckpt-274"
    forward = Inference(test_path, ckpt_path)
    forward()
