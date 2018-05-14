import json
import numpy as np
from data import generator_input
from render import show_channel_label
import matplotlib.pyplot as plt



def load_josn(json_file):
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


def show_centeroffset(label):
    with open(label,'rb') as f: file = json.load(f)

    instance_x, instance_y = None, None
    for channel in file['output']:
        instance_x = np.asarray(channel['instance_pt_x']).reshape([640, 640])
        instance_y = np.asarray(channel['instance_pt_y']).reshape([640, 640])

    plt.figure()
    plt.title("real model output: center offset")
    X, Y = np.meshgrid(np.arange(0, 640, 10), np.arange(0, 640, 10))
    M = np.hypot(instance_x[::10, ::10], instance_y[::10, ::10])
    Q = plt.quiver(X, Y, instance_x[::10, ::10], instance_y[::10, ::10], M, pivot='mid', units='xy')
    qk = plt.quiverkey(Q, 0.9, 0.9, 1, r'$1 \frac{m}{s}$', labelpos='E',
                       coordinates='figure')
    plt.scatter(X, Y, color='r', s=3)
    plt.show()







if __name__ == "__main__":
    path = "/home/bai/Project/cnn_seg/dataset/feature.json"
    bin = "./dataset/007480.bin"
    label = "/home/bai/Project/cnn_seg/dataset/output.json"

    #diff between feature generator
    # c_f = load_josn(path)
    # p_f = generator_input(bin, 640, 640, 8, 60, 5, -5)
    # a = p_f[:,:,7] - c_f[:,:,7]
    # print(a.max(), a.min(), c_f[:,:,7].max(), p_f[:,:,7].max())

    #diff between model output
    show_centeroffset(label)


