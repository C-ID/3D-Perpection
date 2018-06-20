import numpy as np
import tensorflow as tf
import os
from extert_cpp import cluster_for_py
import json
from data import generator_input, F2I
import cv2
import glob


type_color={"pedestrian":(255, 128, 128),  # pink
            "bicycle":(0, 0, 255),  # blue
            "car":(0, 255, 0),  # green
            "unknow":(0, 255, 255)  # yellow
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

class inference(object):
    def __init__(self, test_path, ckpt_path):
        self.cluster = cluster_for_py.PyCluster()
        self.cluster.init(640,640,60)
        self.confidence_thresh = 0.1
        self.height_thresh = 0.5
        self.min_pts_num = 3
        self.test_path = test_path
        self.ckpt_path = ckpt_path

    def test(self, bin_path, json_path):
        cat, ins_x, ins_y, confidence, height, cla = read_output_channel(json_path)
        bin = np.fromfile(bin_path, np.float32)
        pt_nums = len(bin)
        print(pt_nums)
        valid_indices = np.arange(0, pt_nums / 4, 1)
        print(len(valid_indices))
        for i in range(100):
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
    def save(bin, score, name, cloud):
        image3 = np.zeros([640, 640, 3])
        channel = generator_input(bin, 640, 640, 8, 60, 5, -5)
        x, y = np.where(channel[:, :, 2] >= 1)
        image3[x, y, :] = (255, 255, 255)
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
            cv2.rectangle(image3, (int(y.min()), int(x.min())),(int(y.max()), int(x.max())), color, 2)
        cv2.imwrite("./test.jpg", image3)

    def inference(self):
        testfor = 1
        cv2.imwrite()








if __name__ == "__main__":
    json_path = os.path.join(os.getcwd(), "dataset/output.json")
    bin_path = os.path.join(os.getcwd(), "dataset/007480.bin")
    forward = inference()
    forward.test(bin_path, json_path)
