import json
import numpy as np
from data import data_provider
from render import *
import matplotlib.pyplot as plt




def load_json(json_file):
    with open(json_file, "rb") as f:
        file = json.load(f)

    for channel in file["feature"]:
        max_height = np.asarray(channel['channel-0'][0]['data']).reshape([640, 640, 1])
        mean_height  = np.asarray(channel['channel-1'][0]['data']).reshape([640, 640, 1])
        count_data   = np.asarray(channel['channel-2'][0]['data']).reshape([640, 640, 1])
        direction_data = np.asarray(channel['channel-3'][0]['data']).reshape([640, 640, 1])
        top_intensity_data = np.asarray(channel['channel-4'][0]['data']).reshape([640, 640, 1])
        mean_intensity_data = np.asarray(channel['channel-5'][0]['data']).reshape([640, 640, 1])
        distance_data = np.asarray(channel['channel-6'][0]['data']).reshape([640, 640, 1])
        nonempty_data = np.asarray(channel['channel-7'][0]['data']).reshape([640, 640, 1])

    feature = np.concatenate((max_height, mean_height, count_data, direction_data, top_intensity_data, mean_intensity_data, distance_data, nonempty_data), axis=2)


    return feature

def load_now(path):
    with open(path, "rb") as f:
        file = json.load(f)
    for tmp in file["feature"]:
        feature = np.asarray(tmp["data"])

    return feature.reshape([8,640,640])

def read_output_channel(path):
    with open(label, 'rb') as f: file = json.load(f)
    for channel in file['output']:
        instance_x = np.asarray(channel['instance_pt_x']).reshape([640, 640, 1])
        instance_y = np.asarray(channel['instance_pt_y']).reshape([640, 640, 1])
        category_pt = np.asarray(channel['category_pt']).reshape([640, 640, 1])
        classify_pt = np.asarray(channel['classify_pt']).reshape([5, 640, 640])
        confidence_pt = np.asarray(channel['confidence_pt']).reshape([640, 640, 1])
        height_pt = np.asarray(channel['height_pt']).reshape([640, 640, 1])

    return np.concatenate((category_pt, instance_x, instance_y, confidence_pt, height_pt), axis=2), classify_pt




def show_centeroffset(label):
    with open(label,'rb') as f: file = json.load(f)
    instance_x, instance_y = None, None
    for channel in file['output']:
        instance_x = np.asarray(channel['instance_pt_x']).reshape([640, 640])
        instance_y = np.asarray(channel['instance_pt_y']).reshape([640, 640])

    plt.figure()
    # a = plt.subplot(121)
    plt.title("real model output: center offset")
    X, Y = np.meshgrid(np.arange(0, 640, 1), np.arange(640, 0, -1))
    M = np.hypot(instance_x, instance_y)
    Q = plt.quiver(X, Y, instance_x, instance_y, M, pivot='middle', scale=0.5, scale_units='xy')
    qk = plt.quiverkey(Q, 0.9, 0.9, 2, r'$1 \frac{m}{s}$', labelpos='E',
                       coordinates='data')
    # b = plt.subplot(122)
    # png = plt.imread(png)
    # b.imshow(png)
    plt.show()


def classify_pt_show(output_path):
    with open(label,'rb') as f: file = json.load(f)
    for channel in file['output']:
        category = np.asarray(channel['classify_pt']).reshape([640, 640, 5])
    car = category[:,:,1]
    bicycle = category[:,:,4]
    return category

def contrast(c_f, now_f):
    a = now_f[0, :, :] - c_f[:, :, 0]
    b = now_f[1, :, :] - c_f[:, :, 1]
    c = now_f[2, :, :] - c_f[:, :, 2]
    d = now_f[3, :, :] - c_f[:, :, 3]
    e = now_f[4, :, :] - c_f[:, :, 4]
    f = now_f[5, :, :] - c_f[:, :, 5]
    g = now_f[6, :, :] - c_f[:, :, 6]
    h = now_f[7, :, :] - c_f[:, :, 7]
    print(a.max(), a.min(), c_f[:, :, 0].max(), now_f[0, :, :].max())
    print(b.max(), b.min(), c_f[:, :, 1].max(), now_f[1, :, :].max())
    print(c.max(), c.min(), c_f[:, :, 2].max(), now_f[2, :, :].max())
    print(d.max(), d.min(), c_f[:, :, 3].max(), now_f[3, :, :].max())
    print(e.max(), e.min(), c_f[:, :, 4].max(), now_f[4, :, :].max())
    print(f.max(), f.min(), c_f[:, :, 5].max(), now_f[5, :, :].max())
    print(g.max(), g.min(), c_f[:, :, 6].max(), now_f[6, :, :].max())
    print(h.max(), h.min(), c_f[:, :, 7].max(), now_f[7, :, :].max())





if __name__ == "__main__":
    path1 = "/home/bai/Project/3D-Perpection/feature/test/0000000010.json"
    path2 = "/home/bai/Project/3D-Perpection/feature/test/now-0000000010.json"
    bin = "./dataset/007480.bin"
    label = "/home/bai/Project/cnn_seg/dataset/output.json"
    png = "./dataset/007480.png"
    # diff between feature generator
    # c_f = load_json(label)
    # now_f = load_now(path2)
    # show_centeroffset(label)
    # out_channel, classif_pt= read_output_channel(label)
    # classif(classif_pt)
    # show_channel_input(c_f, (640,640))
    # # p_f = generator_input(bin, 640, 640, 8, 60, 5, -5)
    # data = data_provider(640,640,8,12,60)
    # p_f = data.gen(bin)
    # a = p_f[:,:,0] - c_f[:,:,0]
    # print(a.max(), a.min(), c_f[:,:,1].max(), p_f[:,:,1].max())

    #
    #diff between model output
    # show_centeroffset(label)
    # category = classify_pt_show(label)





