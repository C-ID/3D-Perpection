import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.misc import imread
import glob
import os
import cPickle

z_axes = 1 # bottom -> up
x_axes = 2 # left -> right
y_axes = 0 # back -> front
ref_axes = 3

zmax_func = lambda x: x[:, z_axes].max()
zmean_func = lambda x: x[:, z_axes].mean()

zmax_ref_func = lambda x: x[x[:, z_axes].argmax(), ref_axes]
refmean_func = lambda x: x[:, ref_axes].mean()

dist_func = lambda x: np.sqrt(x[:, x_axes].mean()**2 + x[:, y_axes].mean()**2) / 60.
angle_func = lambda x: np.arctan(x[:, x_axes].mean() / x[:, y_axes].mean())
counts_func = lambda x: x.shape[0] / 100.
empty_func = lambda x: float(x.shape[0] > 0)

encoding_funcs = [zmax_func, zmean_func, zmax_ref_func, refmean_func, 
         dist_func, angle_func, counts_func, empty_func]

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

def load_calibration(calib_file):
    calib = map(lambda x: x.strip().split(), open(calib_file).readlines())
    P0 = np.array(map(float,calib[0][1:])).reshape((3,4))
    P1 = np.array(map(float,calib[1][1:])).reshape((3,4))
    P2 = np.array(map(float,calib[2][1:])).reshape((3,4))
    P3 = np.array(map(float,calib[3][1:])).reshape((3,4))
    R0_rect = np.eye(4, dtype='float32')
    R0_3x3 = np.array(map(float,calib[4][1:])).reshape((3,3))
    R0_rect[:3,:3] = R0_3x3
    T_v2c = np.eye(4, dtype='float32')
    T_v2c[:3,:] = np.array(map(float,calib[5][1:])).reshape((3,4))
    T_vel_to_cam = np.dot(R0_rect, T_v2c)
    calibs = {'P0': P0, 'P1': P1, 'P2': P2,'P3': P3,
              'R0_rect': R0_rect,
              'T_v2c': T_v2c, 'T_vel_to_cam': T_vel_to_cam}
    return calibs

def get_data_paths(image_id, data_dir, db):
    #image_id = '006961'
    bin_file = '{}/{}/velodyne/{}.bin'.format(data_dir, db, image_id)
    im_file = '{}/{}/image_2/{}.png'.format(data_dir, db, image_id)
    label_file = '{}/{}/label_2/{}.txt'.format(data_dir, db, image_id)
    calib_file = '{}/{}/calib/{}.txt'.format(data_dir, db, image_id)
    return im_file, label_file, calib_file, bin_file

def project_velo2camera(vel_data, calibs):
    # vel_data_c: col 0: back -> front
    #             col 1: down -> up
    #             col 2: left -> right
    homo_vel_data = np.hstack((vel_data[:,:3],np.ones((vel_data.shape[0],1), dtype='float32')))
    vel_data_c = np.dot(homo_vel_data, calibs['T_vel_to_cam'].T)
    vel_data_c /= vel_data_c[:, -1].reshape((-1,1))
    vel_data_c = np.hstack((vel_data_c[:, :3], vel_data[:, -1].reshape((-1,1))))
    return vel_data_c

def project_velo_camera2image(vel_data_c, calibs):
    homo_vel_data_c = np.hstack((vel_data_c[:, :3], np.ones((vel_data_c.shape[0],1), dtype='float32')))
    vel_data_im = np.dot(homo_vel_data_c, calibs['P2'].T)
    vel_data_im /= vel_data_im[:, -1].reshape((-1, 1))
    vel_data_im = vel_data_im[:, :2]
    return vel_data_im

def load_data(label_file, bin_file, calib_file, velo_sampling=None, velo_positive=False):
    vel_data = np.fromstring(open(bin_file).read(), dtype='float32').reshape((-1, 4))
    if velo_sampling:
        vel_data = vel_data[::velo_sampling,:]
    #vel_data = vel_data[vel_data[:,1]<5,:]
    if velo_positive:
        vel_data = vel_data[vel_data[:,0]>0.1,:]
    #vel_data = vel_data[vel_data[:,2]>-1,:]
    annos = parse_kitti_label(label_file)
    calibs = load_calibration(calib_file)
    # convert velodyne points to camera coords
    vel_data_c = project_velo2camera(vel_data, calibs)
    # convert velodyne points to image coords
    vel_data_im = project_velo_camera2image(vel_data_c, calibs)
    return annos, calibs, vel_data_c, vel_data_im

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

def get_point_cloud_grids(data, grid_limits, cell_sizes):
    assert len(grid_limits) == 3, 'You need to specify the grid limits in X,Y,Z axes.'
    assert len(cell_sizes) == 3, 'You need to specify the cell sizes in X,Y,Z axes.'
    num_cells = [int(np.floor((l[1] - l[0]) / cs)) for l, cs in zip(grid_limits, cell_sizes)]
    start_limits = np.array([l[0] for l in grid_limits]).reshape((-1, 3))
    cell_id_quantized = ((data[:, :3] - start_limits) / np.array(cell_sizes).reshape((-1,3))).astype('int64')
    cells = {}
    for i in xrange(data.shape[0]):
        pt = data[i]
        cell_id = cell_id_quantized[i]
        if np.any(cell_id < 0) or np.any(cell_id >= np.array(num_cells)):
            continue
        cell_id_key = '_'.join(map(str, cell_id))
        if cell_id_key not in cells:
            cells[cell_id_key] = []
        cells[cell_id_key].append(pt)
    for k in cells:
        cells[k] = np.hstack(cells[k]).reshape((-1,4))
    return cells, num_cells

def get_cell_features(cells, num_cells, funcs):
    X = np.zeros(list(num_cells) + [len(funcs)], dtype='float32')
    for i, func in enumerate(funcs):
        fs = map(lambda k: [map(int, k.split('_')),func(cells[k])], cells)
        for ids, f in fs:
            X[ids[0], ids[1],ids[2], i] = f
    return X

db = 'training'
data_dir = '/home/users/benjin.zhu/data/Datasets/KITTI3D'
images = glob.glob('%s/%s/image_2/*.png' %(data_dir, db))
out_dir = '/home/users/benjin.zhu/data/Datasets/KITTI3D/preprocessed/' + db

if not os.path.exists(out_dir):
    os.makedirs(out_dir)
image_indices = map(lambda x: os.path.basename(x).split('.')[0], images)
print(type(image_indices))

grid_limits=[[-30, 30],[-2, 3],[0, 60]] # Y, Z, X
cell_sizes=[0.1,5,0.1] # vh, vd, vw, result in H' = 400, D' = 10, W' = 352

for j, idx in enumerate(image_indices):
    print(j, idx)
    im_file, label_file, calib_file, bin_file = get_data_paths(idx, data_dir, db)
    out_file = out_dir + '/' + os.path.basename(im_file).split('.')[0] + '.pkl'
    #if os.path.exists(out_file):
    #    continue
    im = imread(im_file)
    annos, calibs, vel_data_c, vel_data_im = load_data(label_file, bin_file, calib_file, 
                                                       velo_sampling=None, velo_positive=True)
    boxes = []
    for o in annos:
        print(len(annos))
        if True: 
            #o['type'] in ['Car']:
            # box3d is 3x8
            # row 0: left -> right
            # row 1: bottom -> up
            # row 2: back -> front
            box3d = compute_3d_corners(o['l'], o['w'], o['h'], o['t'], o['yaw'])
            print(box3d.shape)
            start_limits = np.array([l[0] for l in grid_limits]).reshape((-1, 3))
            cell_id_quantized = ((box3d.T - start_limits) / np.array(cell_sizes).reshape((-1,3))).astype('int64')
            proj_8pts = cell_id_quantized.T
            boxes.append(proj_8pts)
    cells, num_cells = get_point_cloud_grids(vel_data_c, grid_limits=grid_limits, cell_sizes=cell_sizes)
    X = get_cell_features(cells, num_cells, encoding_funcs)
    print(X.shape)
    print(boxes)
    
    cPickle.dump({'X': X, 'boxes': boxes}, open(out_file, 'w'), -1)
