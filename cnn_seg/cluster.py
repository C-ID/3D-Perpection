import numpy as np
import os


obj_type={
    "META_UNKNOWN": 1,
    "META_SMALLMOT": 2,
    "META_BIGMOT": 3,
    "META_NONMOT": 4,
    "META_PEDESTRIAN": 5,
    "MAX_META_TYPE":6
}

class Object(object):
    def __init__(self):
        self.grid = np.zeros([640, 640])
        self.cloud = None
        self.score = 0
        self.height = 0
        self.meta_type = None
        self.meta_type_probs = None


class UnionFind(object):
    def __init__(self):
        self.center_node = None
        self.parent = None
        self.node_rank =None
        self.traversed = None
        self.is_center = False
        self.is_object = False
        self.point_num = 0
        self.obstacle_id = 0




class Cluster(object):
    def __init__(self, row, col, range):
        self.rows_ = row
        self.cols_ = col
        self.grids_ = row*col
        self.range_ = range
        self.scale_ = 0.5 * row / range
        self.inv_res_x_ = 0.5 * col / range
        self.inv_res_y_ = 0.5 * row / range
        self.point2grid = None
        self.onstacle_ = None
        self.id_img_ = None
        self.bin = None

    def Cluster2Grid(self, category_pt, instance_pt, bin, valid_indices):pass

    def Filter(self, confidence_pt, height_pt):pass

    def Classify(self, classify_pt):pass

    def GetObjects(self, confidence_thresh, height_thresh, min_pt_num):pass

    def IsValidRowCol(self):pass

    def IsValidRow(self, row): return row >= 0 and row < self.rows_

    def IsValidCol(self, col): return col >= 0 and col < self.cols_

    def RowCol2Grid(self, row, col): return row * self.cols_ + col





