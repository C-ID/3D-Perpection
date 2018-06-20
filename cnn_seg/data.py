import numpy as np
import os
import argparse
import time
from render import *
import cv2


class data_provider(object):
    def __init__(self, width, height, in_channel, out_channel, range_):

        assert isinstance(width, int) and isinstance(height, int), "Wrong type, need int type for channel size"
        self.width = width
        self.height = height
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.range = range_
        self.max_height = 5
        self.min_height = -5
        self.inv_res_x = 0.5 * width / range_  # length of each grid(x: meters)
        self.inv_res_y = 0.5 * height / range_  # length of each grid(y: meters)
        # log_table_ = np.arange(0, 256 + 1)
        # self.log_table_ = np.log1p(1 + log_table_)

        #prepare func fo eight channel input feature
        self.functions = self.func()

        self.grid_limits = [[-range_, range_], [-range_, range_]]  # axis for X, Y, Z
        self.cell_sizes = [self.inv_res_x, self.inv_res_y]

        self.num_cells = [640, 640]
        self.start_limits = np.array([l[0] for l in self.grid_limits]).reshape((-1, 2))



    def pix2pc(self, in_pixel, in_size, out_range):
        res = 2.0 * out_range / in_size
        return out_range - (in_pixel + .5) * res

    def F2I(self, val, ori, scale):
        return np.floor((ori - val) * scale)

    def LogCount(self, count):
        log_table_ = np.arange(0, 256 + 1)
        log_table_ = np.log1p(1+log_table_)
        if count < len(log_table_):
            return log_table_[count]
        return np.log(1+count)

    def get_point_cloud_grids(self, bin):
        data = np.fromfile(bin, np.float32).reshape([-1, 4])
        cell_id_quantized = np.floor((data[:, :2] - self.start_limits) * np.array(self.cell_sizes).reshape((-1, 2))).astype(np.int32)
        cells = {}
        for i in range(data.shape[0]):
            pt = data[i]
            cell_id = cell_id_quantized[i]
            if np.any(cell_id < 0) or np.any(cell_id >= np.array(self.num_cells)):
                continue
            cell_id_key = '_'.join(map(str, cell_id))
            if cell_id_key not in cells:
                cells[cell_id_key] = []
            cells[cell_id_key].append(pt)
        for k in cells:
            cells[k] = np.vstack(cells[k])
        return cells

    def get_cell_features(self, cells):
        channel_map = np.zeros(list(self.num_cells) + [len(self.functions)], dtype='float32')
        channel_map[:,:,0].fill(-5.)
        for i, func in enumerate(self.functions):
            fs = map(lambda k: [map(int, k.split('_')), func(cells[k])], cells)
            for ids, f in fs:
                ids = list(ids)
                channel_map[ids[0],ids[1], i] = f

        # for i in range(self.width):
        #     for j in range(self.height):
        #         if channel_map[i,j,2] <= 1e-6: channel_map[i,j,0] = 0.
        #         else:
        #             channel_map[i,j,1] /= channel_map[i,j,2]
        #             channel_map[i,j,5] /= channel_map[i,j,2]
        #             channel_map[i,j,7] = 1.
        #         channel_map[i,j,2] = self.LogCount(int(channel_map[i, j, 2]))


        for i in range(self.width):
            for j in range(self.height):
                center_x = self.pix2pc(i, self.height, self.range)
                center_y = self.pix2pc(j, self.width, self.range)
                channel_map[i,j,3] = np.arctan2(center_y, center_x) / (2. * np.pi)  # direction data
                channel_map[i,j,6] = np.hypot(center_x, center_y) / 60.0 - 0.5  # distance data
        return channel_map

    def gen(self, bin):
        cells = self.get_point_cloud_grids(bin)
        return self.get_cell_features(cells)

    def func(self):
        z_axes = 2
        ref_axes = 3

        zmax_func = lambda x: x[:, z_axes].max()
        zmean_func = lambda x: x[:, z_axes].mean()

        z_max_refmax_func = lambda x: x[x[:, z_axes].argmax(), ref_axes] / 255.
        refmean_func = lambda x: x[:, ref_axes].mean()

        dist_func = lambda x: 0.
        angle_func = lambda x: 0.
        counts_func = lambda x: self.LogCount(int(x.shape[0]))
        nonempty_func = lambda x: float(x.shape[0] > 0)

        encoding_funcs = [zmax_func, zmean_func, counts_func, angle_func, z_max_refmax_func, refmean_func,
                          dist_func, nonempty_func]

        return encoding_funcs

    def gen_data(self, bin, channel_map):
        # compute for vaild points
        idx = np.where(bin[:, 2] < self.max_height)
        bin = bin[idx]
        idx = np.where(bin[:, 2] > self.min_height)
        bin = bin[idx]


        for i in range(len(bin)):
            pos_x = int(self.F2I(bin[i, 1], self.range, self.inv_res_x))  #col
            pos_y = int(self.F2I(bin[i, 0], self.range, self.inv_res_y))  #row

            if(pos_x >= self.width or pos_x < 0 or pos_y >= self.height or pos_y < 0):continue

            pz = bin[i, 2]   #max height data
            pi = bin[i, 3] / 255.  #top intensity data
            render_pi = bin[i, 3]

            if(channel_map[pos_y,pos_x,0]<pz):
                channel_map[pos_y,pos_x,0] = pz   #max height data
                channel_map[pos_y,pos_x,4] = pi   #top intensity data
            channel_map[pos_y,pos_x,1] += pz    #mean height data
            channel_map[pos_y,pos_x,5] += pi    #mean intensity data
            channel_map[pos_y,pos_x,2] += 1.     #count data
            # print(channel_map[pos_y, pos_x, 0])

        for i in range(self.width):
            for j in range(self.height):
                if channel_map[i,j,2] <= 1e-6: channel_map[i,j,0] = 0.
                else:
                    channel_map[i,j,1] /= channel_map[i,j,2]
                    channel_map[i,j,5] /= channel_map[i,j,2]
                    channel_map[i,j,7] = 1.
                channel_map[i,j,2] = self.LogCount(int(channel_map[i, j, 2]))
        return channel_map


    def generator_input(self, path):
        start = time.time()
        bin = np.fromfile(path, np.float32).reshape([-1, 4])

        channel_map = np.zeros([self.width, self.height, self.in_channel], dtype=np.float32)
        channel_map[:, :, 0].fill(-5.)

        channel_map = self.gen_data(bin, channel_map)
        for i in range(self.width):
            for j in range(self.height):
                center_x = self.pix2pc(i, self.height, self.range)
                center_y = self.pix2pc(j, self.width, self.range)
                channel_map[i][j][3] = np.arctan2(center_y, center_x) / (2. * np.pi)  # direction data
                channel_map[i][j][6] = np.hypot(center_x, center_y) / 60.0 - 0.5  # distance data
        print(time.time() - start)
        return channel_map

    def generator_label(self, label_path):
        objs = self.parse_kitti_label(label_path)
        label = np.zeros([self.width, self.height, self.out_channel], dtype=np.float32)
        feature = self.get_label_channel(label, objs)
        return feature

    def parse_kitti_label(self, label_file):
        lines = open(label_file).readlines()
        lines = map(lambda x: x.strip().split(), lines)
        objs = []
        for l in lines:
            o = {}
            o['type'] = l[0]
            o['truncation'] = float(l[1])
            o['occlusion'] = int(l[2])
            o['alpha'] = float(l[3])
            o['box2d'] = [float(l[4]), float(l[5]), float(l[6]), float(l[7])]
            o['h'] = float(l[8])
            o['w'] = float(l[9])
            o['l'] = float(l[10])
            o['t'] = [float(l[11]), float(l[12]), float(l[13])]
            o['yaw'] = float(l[14])
            objs.append(o)
        return objs

    def compute_3d_corners(self, l, w, h, t, yaw):
        R = np.array([[np.cos(yaw), 0, np.sin(yaw)],
                      [0, 1, 0],
                      [-np.sin(yaw), 0, np.cos(yaw)]])
        # 3D bounding box corners
        x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
        y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
        z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]
        corners_3D = np.dot(R, np.array([x_corners, y_corners, z_corners]))
        corners_3D += np.array(t).reshape((3, 1))
        return corners_3D

    def get_label_channel(self, channel, obj):
        car = ['Car', 'Van', 'Truck', 'Tram', 'Misc']
        person = ['Pedestrian', 'Person_sitting']
        for o in obj:
            box3d = self.compute_3d_corners(o['l'], o['w'], o['h'], o['t'], o['yaw'])

            x = self.F2I(box3d[0, :], 60, 0.5 * 640 / 60)
            y = self.F2I(box3d[2, :], 60, 0.5 * 640 / 60)

            height = box3d[1, 0] - box3d[1, 4]
            center = o['t']
            center_x = self.F2I(center[0], 60, 0.5 * 640 / 60)  # col
            center_y = self.F2I(center[2], 60, 0.5 * 640 / 60)  # row
            if (center_x >= self.width or center_x < 0 or center_y >= self.height or center_y < 0): continue

            step_x = [i for i in range(int(x.min()), int(x.max()) + 1, 1)]
            step_z = [i for i in range(int(y.min()), int(y.max()) + 1, 1)]

            # generator center offset
            center_offset_x = (np.array(step_x) - int(center_x))
            center_offset_y = (np.array(step_z) - int(center_y))

            for i in range(len(step_x)):
                for j in range(len(step_z)):
                    channel[step_x[i], step_z[j], 0] = 1.  # category_pt
                    channel[step_x[i], step_z[j], 1] = center_offset_x[i]  # instance_x
                    channel[step_x[i], step_z[j], 2] = center_offset_y[j]  # instance_y
                    channel[step_x[i], step_z[j], 3] = 1.  # confidence_pt
                    channel[step_x[i], step_z[j], 11] = height  # height_pt
                    if o['type'] in car:
                        channel[step_x[i], step_z[j], 5:7] = 1.  # classify_pt :4-8
                    elif o['type'] in person:
                        channel[step_x[i], step_z[j], 8] = 1.
                    elif o['type'] == 'DontCare':
                        channel[step_x[i], step_z[j], 4] = 1.
                    elif o['type'] == 'Cyclist':
                        channel[step_x[i], step_z[j], 7] = 1.

        return channel


def pix2pc(in_pixel, in_size, out_range):
    res = 2.0 * out_range / in_size
    return out_range - (in_pixel + .5) * res

def F2I(val, ori, scale):
    return np.floor((ori - val) * scale)

def LogCount(count):
    log_table_ = np.arange(0, 256+1)
    log_table_ = np.log1p(1+log_table_)
    if count < len(log_table_):
        return log_table_[count]
    return np.log(1+count)

def generator_input(args, width, height, channel, range_, maxh, minh):
    """
    :param bin: which cloud points
    :param width: width of input feature
    :param height: height of input feature
    :param channel: input size of channels, 0-7 i.e, max height, top intn
    :return: input feature
    """
    bin = np.fromfile(args, np.float32).reshape([-1, 4])
    assert isinstance(width, int) and isinstance(height, int), "Wrong type for input channel map"
    channel_map = np.zeros([width, height, channel], dtype=np.float64)
    channel_map[:,:,0].fill(-5.)
    inv_res_x = 0.5 * width / range_  #length of each grid(x: meters)
    inv_res_y = 0.5 * height / range_  #length of each grid(y: meters)

    channel_map = count_data(bin, channel_map, maxh, minh, inv_res_x, inv_res_y, range_, width, height)
    for i in range(width):
        for j in range(height):
            center_x = pix2pc(i, height, range_)
            center_y = pix2pc(j, width, range_)
            channel_map[i][j][3] = np.arctan2(center_y, center_x) / (2. * np.pi) # direction data
            channel_map[i][j][6] = np.hypot(center_x, center_y) / 60.0 - 0.5     # distance data
    return channel_map


def count_data(bin, channel_map, max_height, min_height, inv_res_x, inv_res_y, range_, width, height):
    render_pi = np.zeros([640, 640], np.float64)
    #compute for vaild points
    idx = np.where(bin[:, 2] < max_height)
    bin = bin[idx]
    idx = np.where(bin[:, 2] > min_height)
    bin = bin[idx]

    for i in range(len(bin)):
        pos_x = int(F2I(bin[i, 1], range_, inv_res_x))  #col
        pos_y = int(F2I(bin[i, 0], range_, inv_res_y))  #row

        if(pos_x >= width or pos_x < 0 or pos_y >= height or pos_y < 0):continue

        pz = bin[i, 2]   #max height data
        pi = bin[i, 3] / 255.  #top intensity data
        pi_ = bin[i, 3]
        if(channel_map[pos_y,pos_x,0]<pz):
            channel_map[pos_y,pos_x,0] = pz   #max height data
            channel_map[pos_y,pos_x,4] = pi   #top intensity data
        channel_map[pos_y,pos_x,1] += pz    #mean height data
        channel_map[pos_y,pos_x,5] += pi    #mean intensity data
        channel_map[pos_y,pos_x,2] += 1.     #count data
        # print(channel_map[pos_y, pos_x, 0])

    for i in range(width):
        for j in range(height):
            if channel_map[i,j,2] <= 1e-6: channel_map[i,j,0] = 0.
            else:
                channel_map[i,j,1] /= channel_map[i,j,2]
                channel_map[i,j,5] /= channel_map[i,j,2]
                channel_map[i,j,7] = 1.
            channel_map[i,j,2] = LogCount(int(channel_map[i, j, 2]))
    return channel_map


def gt_label(label_path, width, height, channel):
    objs = parse_kitti_label(label_path)
    label = np.zeros([width, height, channel], dtype=np.float64)
    feature = get_label_channel(label, objs)
    #show_channel_label(channel, (width, height))
    return feature

def parse_kitti_label(label_file):
    lines = open(label_file).readlines()
    lines = map(lambda x: x.strip().split(), lines)
    objs = []
    for l in lines:
        o = {}
        o['type'] = l[0]
        o['truncation'] = float(l[1])
        o['occlusion'] = int(l[2])
        o['alpha'] = float(l[3])
        o['box2d'] = [float(l[4]), float(l[5]), float(l[6]), float(l[7])]
        o['h'] = float(l[8])
        o['w'] = float(l[9])
        o['l'] = float(l[10])
        o['t'] = [float(l[11]), float(l[12]), float(l[13])]
        o['yaw'] = float(l[14])
        objs.append(o)
    return objs

def compute_3d_corners(l, w, h, t, yaw):
    R = np.array([[np.cos(yaw), 0, np.sin(yaw)],
                  [0, 1, 0],
                  [-np.sin(yaw), 0, np.cos(yaw)]])
    # 3D bounding box corners
    x_corners = [l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2];
    y_corners = [0,0,0,0,-h,-h,-h,-h];
    z_corners = [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2];
    corners_3D = np.dot(R, np.array([x_corners,y_corners,z_corners]))
    corners_3D += np.array(t).reshape((3,1))
    return corners_3D

def get_label_channel(channel, obj):
    num_obj = len(obj)
    small_car = ['Car', 'Tram']
    big_car = ['Van', 'Truck']
    person = ['Pedestrian', 'Person_sitting']
    for o in obj:

        box3d = compute_3d_corners(o['l'], o['w'], o['h'], o['t'], o['yaw'])
        y = F2I(box3d[0,:], 60, 0.5*640/60)
        y = 320-(y-320)
        x = F2I(box3d[2,:], 60, 0.5*640/60)
        vertices = [list(i) for i in zip(x[:4], y[:4])]
        polygens = ComputePolygen(vertices)
        print("123: ", vertices)
        height = box3d[1,0]
        center = o['t']
        center_y = F2I(center[0], 60, 0.5*640/60)    #col
        center_y = 320 - (center_y - 320)
        center_x = F2I(center[2], 60, 0.5*640/60)    #row
        if (center_x >= 640 or center_x < 0 or center_y >= 640 or center_y < 0): continue

        step_x =[i for i in range(int(x.min()), int(x.max())+1, 1)]
        step_z =[i for i in range(int(y.min()), int(y.max())+1, 1)]
        print(step_x, step_z)


        for i in range(len(step_x)):
            for j in range(len(step_z)):
                if PointsInPolygen([step_x[i], step_z[j]], polygens, [center_x, center_y]):
                    offset = centeroffset([center_x, center_y], [step_x[i], step_z[j]])
                    channel[step_x[i], step_z[j], 0] = 1.  #category_pt
                    channel[step_x[i], step_z[j], 1] = offset[0]  #instance_x
                    channel[step_x[i], step_z[j], 2] = offset[1]  #instance_y
                    channel[step_x[i], step_z[j], 3] = 1.                   #confidence_pt
                    channel[step_x[i], step_z[j], 11] = height             #height_pt
                    if o['type'] in small_car: channel[step_x[i], step_z[j], 5] = 1.  # classify_pt :4-8
                    elif o['type'] in big_car: channel[step_x[i], step_z[j], 6] = 1.
                    elif o['type'] in person: channel[step_x[i], step_z[j], 8] = 1.
                    elif o['type'] == 'DontCare': channel[step_x[i], step_z[j], 4] = 1.
                    elif o['type'] == 'Cyclist': channel[step_x[i], step_z[j], 7] = 1.
    return channel

def getABC(po1, po2):
    A = po2[1] - po1[1]
    B = po1[0] - po2[0]
    C = A*po1[0] + B*po1[1]
    return A, B, -C

def ComputePolygen(vertices):
    assert len(vertices) == 4, "ConvexHull need four vertices, clockwise"
    points_1 = vertices[0]
    points_2 = vertices[1]
    points_3 = vertices[2]
    points_4 = vertices[3]
    A1, B1, C1 = getABC(points_1, points_2)
    A2, B2, C2 = getABC(points_2, points_3)
    A3, B3, C3 = getABC(points_3, points_4)
    A4, B4, C4 = getABC(points_4, points_1)
    line_1 = lambda x, y: x * A1 + y * B1 + C1
    line_2 = lambda x, y: x * A2 + y * B2 + C2
    line_3 = lambda x, y: x * A3 + y * B3 + C3
    line_4 = lambda x, y: x * A4 + y * B4 + C4
    polygen = [line_1, line_2, line_3, line_4]
    return polygen


def PointsInPolygen(points, polygen, center):
    b = 0
    for i, func in enumerate(polygen):
        if func(center[0], center[1]) < 0:
            a = func(points[0], points[1]) <= 0
            b += a
        else:
            a = func(points[0], points[1]) >= 0
            b += a
    if b==4:return True
    else: return False

def centeroffset(center, points):
    len_x = center[0]-points[0]
    len_y = center[1]-points[1]
    length = np.hypot(len_x, len_y)
    if length != 0:
        offset_x = len_x/length * 0.5
        offset_y = len_y/length * 0.5
    else:
        offset_x = offset_y = 0
    return [offset_x, offset_y]


def draw(path, path1):
    image = cv2.imread(path)
    objs = parse_kitti_label(path1)
    for o in objs:
        box3d = compute_3d_corners(o['l'], o['w'], o['h'], o['t'], o['yaw'])
        y = F2I(box3d[0,:], 60, 0.5*640/60)
        x = F2I(box3d[2,:], 60, 0.5*640/60)
        vertices = [i for i in zip(np.int64(x[:4]), np.int64(y[:4]))]
        for i in range(4):
            cv2.line(image, vertices[i%4], vertices[(i+1)%4], (255,0,0), 1)
    cv2.imwrite("./test.png", image)








if __name__ == "__main__":
    bin_path ="/home/bai/wrongbin/004466.bin" #"./dataset/007480.bin"
    label_path ="/home/bai/wrongbin/label_2/004466.txt" #"./dataset/007480.txt"
    # start = time.time()
    p = "/home/bai/Project/cnn_seg/testpng/003256.txt--in-1.png"
    name = os.path.basename(label_path)
    chan = generator_input(bin_path, 640, 640, 8, 60, 5, -5)
    chan = chan[np.newaxis, :]
    # show_channel_input(chan, (640, 640))
    gt = gt_label(label_path, 640, 640, 12)
    gt = gt[np.newaxis, :]
    record_confirm(chan, gt, name)
    # data = data_provider(640,640,8,12,60)
    # channel = data.generator_input(bin_path)
    # show_channel_label(gt, (640,640))
    # classif(gt)
    # print(time.time() - start)
    # draw(p, label_path)
