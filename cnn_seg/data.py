import numpy as np
import os
import argparse
import time
from render import *



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
    channel_map = np.zeros([width, height, channel])
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
    label = np.zeros([width, height, channel])
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
    car = ['Car', 'Van', 'Truck', 'Tram', 'Misc']
    person = ['Pedestrian', 'Person_sitting']
    for o in obj:
        box3d = compute_3d_corners(o['l'], o['w'], o['h'], o['t'], o['yaw'])

        x = F2I(box3d[0,:], 60, 0.5*640/60)
        y = F2I(box3d[2,:], 60, 0.5*640/60)

        height = box3d[1,0] - box3d[1,4]
        center = o['t']
        center_x = F2I(center[0], 60, 0.5*640/60)    #col
        center_y = F2I(center[2], 60, 0.5*640/60)    #row
        if (center_x >= 640 or center_x < 0 or center_y >= 640 or center_y < 0): continue

        step_x =[i for i in range(int(x.min()), int(x.max())+1, 1)]
        step_z =[i for i in range(int(y.min()), int(y.max())+1, 1)]

        #generator center offset
        center_offset_x = (np.array(step_x) - int(center_x))
        center_offset_y = (np.array(step_z) - int(center_y))


        for i in range(len(step_x)):
            for j in range(len(step_z)):
                channel[step_x[i], step_z[j], 0] = 1.  #category_pt
                channel[step_x[i], step_z[j], 1] = center_offset_x[i]  #instance_x
                channel[step_x[i], step_z[j], 2] = center_offset_y[j]  #instance_y
                channel[step_x[i], step_z[j], 3] = 1.                   #confidence_pt
                channel[step_x[i], step_z[j], 11] = height             #height_pt
                if o['type'] in car: channel[step_x[i], step_z[j], 5:7] = 1.   #classify_pt :4-8
                elif o['type'] in person: channel[step_x[i], step_z[j], 8] = 1.
                elif o['type'] == 'DontCare': channel[step_x[i], step_z[j], 4] = 1.
                elif o['type'] == 'Cyclist' : channel[step_x[i], step_z[j], 7] = 1.

    return channel

def args():
    def str2bool(v): return v.lower() in ("yes", "true", "t", "1", True)
    parser = argparse.ArgumentParser(prog="Python", description="Aim at testing cnn_seg results through VTK rendering")
    parser.add_argument("--bin-path", type=str, required=True, help="path for pcd file")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    bin_path = "./dataset/007480.bin"
    label_path = "./dataset/007480.txt"
    start = time.time()
    #chan = generator_input(bin_path, 640, 640, 8, 60, 5, -5)
    #show_channel_input(chan, (640, 640))
    gt = gt_label(label_path, 640, 640, 12)
    print(time.time() - start)