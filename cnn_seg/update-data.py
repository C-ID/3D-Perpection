import numpy as np
import os
import time
from render import show_channel_input,show_channel_label

cfg = {
    'start_limits': [-60, -60],   #start instences for axis for x & y
    'range': 60,
    'num_cells':[640, 640],
    'inv_res_x': 0.5 * 640 / 60,
    'inv_res_y': 0.5 * 640 / 60,
    'in_channel': 8,
    'out_channel': 12,
    'width':640,
    'height':640,
    'cell_sizes': [0.5 * 640 / 60, 0.5 * 640 / 60],
    'z_range': [-5, 5]
}

def func():
    z_axes = 2
    ref_axes = 3

    zmax_func = lambda x: x[:, z_axes].max()
    zmean_func = lambda x: x[:, z_axes].mean()
    z_max_refmax_func = lambda x: x[x[:, z_axes].argmax(), ref_axes] / 255.
    refmean_func = lambda x: x[:, ref_axes].mean()
    # dist_func = lambda x: 0.
    # angle_func = lambda x: 0.
    counts_func = lambda x: LogCount(int(x.shape[0]))
    nonempty_func = lambda x: float(x.shape[0] > 0)
    # encoding_funcs = [zmax_func, zmean_func, counts_func, angle_func, z_max_refmax_func, refmean_func,
    #                   dist_func, nonempty_func]
    encoding_funcs = [zmax_func, zmean_func, counts_func, z_max_refmax_func, refmean_func, nonempty_func]
    return encoding_funcs

def pix2pc(in_pixel, in_size, out_range):
    res = 2.0 * out_range / in_size
    return out_range - (in_pixel + .5) * res


def F2I(val, ori, scale):
    return np.floor((ori - val) * scale)


def LogCount(count):
    log_table_ = np.arange(0, 256 + 1)
    log_table_ = np.log1p(1 + log_table_)
    if count < len(log_table_):
        return log_table_[count]
    return np.log(1 + count)


def get_point_cloud_grids(bin, config):
    points = np.fromfile(bin, np.float32).reshape([-1, 4])
    #constraint z axis distances among -5 to 5 meters
    idx = np.where(points[:, 2] < config['z_range'][1])
    points = points[idx]
    idx = np.where(points[:, 2] > config['z_range'][0])
    points = points[idx]

    # cell_id_quantized = np.floor((points[:, :2] - config['start_limits']) * np.array(config['cell_sizes']).reshape((-1, 2))).astype(
    #     np.int32)
    cell_id_quantized = F2I(points[:,:2], config['range'], config['cell_sizes'])
    cell_id_quantized = np.floor(cell_id_quantized).astype(np.int32)
    cells = {}
    for i in range(points.shape[0]):
        pt = points[i]
        cell_id = cell_id_quantized[i]
        if np.any(cell_id < 0) or np.any(cell_id >= np.array(config['num_cells'])):
            continue
        cell_id_key = '_'.join(map(str, cell_id))
        if cell_id_key not in cells:
            cells[cell_id_key] = []
        cells[cell_id_key].append(pt)
    for k in cells:
        cells[k] = np.vstack(cells[k])
    return cells


def get_cell_features(cells, config, functions):
    channel_map = np.zeros(list(config['num_cells']) + [len(functions) + 2], dtype='float32')
    channel_map[:, :, 0].fill(-5.)

    for i, func in enumerate(functions):
        fs = map(lambda k: [map(int, k.split('_')), func(cells[k])], cells)
        for ids, f in fs:
            ids = list(ids)
            channel_map[ids[0], ids[1], i] = f


    y_index, x_index = np.meshgrid(range(config['width']), range(config['height']))
    x_index, y_index = x_index.flatten(), y_index.flatten()

    coord = np.array([c for c in zip(x_index, y_index)])

    channel_map_angle = np.arctan2(coord[:, 1], coord[:, 0]).reshape([640,640]) / (2. * np.pi)
    channel_map_dist = np.hypot(coord[:, 0], coord[:, 1]).reshape([640, 640]) / 60.0 - 0.5
    for i in range(config['width']):
        for j in range(config['height']):
            center_x = pix2pc(i, config['height'], config['range'])
            center_y = pix2pc(j, config['width'], config['range'])
            print(center_x, center_y)
            channel_map[i, j, 3] = np.arctan2(center_y, center_x) / (2. * np.pi)  # direction data
            channel_map[i, j, 6] = np.hypot(center_x, center_y) / 60.0 - 0.5  # distance data
    # print(np.max(channel_map_angle - channel_map[:,:,3]), np.min(channel_map_angle - channel_map[:,:,3]))
    return channel_map


def gen(bin, config):
    functions = func()
    cells = get_point_cloud_grids(bin, config)
    return get_cell_features(cells, config, functions)




if __name__ == "__main__":
    range_ = [[-60, 60], [-60, 60]]# axis for X, Y, Z
    bin_path = "/home/bai/Project/cnn_seg/dataset/007480.bin"
    start = time.time()
    chan = gen(bin_path, cfg)
    print(time.time() - start)
    # show_channel_input(chan, 1)
    # show_channel_label(chan, 1)


