import numpy as np
import os

obj_type={
    1 : "META_UNKNOWN",
    2 : "META_SMALLMOT",
    3 : "META_BIGMOT",
    4 : "META_NONMOT",
    5 : "META_PEDESTRIAN",
    6 : "MAX_META_TYPE"
}

def DisjointSetMakeSet():
    pass

def DisjointSetUnion():
    pass

def DisjointSetFind():
    pass

def DisjointSetFindRecursive():
    pass



class Object(object):
    def __init__(self):
        self.grid = None
        self.cloud = None
        self.score = 0
        self.height = 0
        self.meta_type = None
        self.meta_type_probs = None

class Node(object):
    def __init__(self):
        self.center_node = None
        self.parent = None
        self.node_rank = None
        self.traversed = None
        self.is_center = False
        self.is_object = False
        self.point_num = 0
        self.obstacle_id = 0


class Cluster(object):
    def __init__(self, row, col, range):
        """
        :param row: row of grid in clustering range
        :param col: colum of grid in clustering range
        :param range: before and after distance(meters) in 3D cloud points
        """
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


    def IsValidRowCol(self, row, col): return self.IsValidRow(row) and self.IsValidCol(col)
    def IsValidRow(self, row): return row >= 0 and row < self.rows_
    def IsValidCol(self, col): return col >= 0 and col < self.cols_
    def RowCol2Grid(self, row, col): return row * self.cols_ + col
    def GetTypeText(self, obj_type):
        if obj_type == "VEHICLE": return "car"
        elif obj_type == "PEDESTRIAN": return "pedestrian"
        elif obj_type == "BICYCLE": return "bicycle"
        else: return "unknown"

    def Cluster2Grid(self, category_pt, instance_pt, bin_path, valid_indices):
        bin = np.fromfile(bin_path, np.float32).reshape([-1, 4])
        total_point_num = bin.shape[0]
        assert len(valid_indices) == total_point_num, \
            "size of 3d cloud points is not equal to valid indices"



    def Filter(self, confidence_pt, height_pt):pass

    def Classify(self, classify_pt):pass

    def GetObjects(self, confidence_thresh, height_thresh, min_pt_num):pass




if __name__ == "__main__":
    print(obj_type[6])