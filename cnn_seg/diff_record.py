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


def show_centeroffset(label, bin_path, png):
    with open(label,'rb') as f: file = json.load(f)
    chan = generator_input(bin_path, 640, 640, 8, 60, 5, -5)
    instance_x, instance_y = None, None
    for channel in file['output']:
        instance_x = np.asarray(channel['instance_pt_x']).reshape([640, 640])
        instance_y = np.asarray(channel['instance_pt_y']).reshape([640, 640])

    # a = plt.figure()
    a = plt.subplot(121)
    plt.title("real model output: center offset")
    X, Y = np.meshgrid(np.arange(0, 640, 1), np.arange(640, 0, -1))
    M = np.hypot(instance_x, instance_y)
    Q = a.quiver(X, Y, instance_x, instance_y, M, pivot='tip', units='xy')
    qk = a.quiverkey(Q, 0.9, 0.9, 2, r'$1 \frac{m}{s}$', labelpos='N',
                       coordinates='data')
    b = plt.subplot(122)
    png = plt.imread(png)
    b.imshow(png)
    plt.show()

def classify_pt_show(output_path):
    with open(label,'rb') as f: file = json.load(f)
    for channel in file['output']:
        category = np.asarray(channel['classify_pt']).reshape([640, 640, 5])
    car = category[:,:,1]
    bicycle = category[:,:,4]
    return category







if __name__ == "__main__":
    path = "/home/bai/Project/cnn_seg/dataset/feature.json"
    bin = "./dataset/007480.bin"
    label = "/home/bai/Project/cnn_seg/dataset/output.json"
    png = "./dataset/007480.png"
    #diff between feature generator
    # c_f = load_josn(path)
    # p_f = generator_input(bin, 640, 640, 8, 60, 5, -5)
    # a = p_f[:,:,2] - c_f[:,:,2]
    # print(a.max(), a.min(), c_f[:,:,2].max(), p_f[:,:,2].max())

    #
    #diff between model output
    #show_centeroffset(label, bin, png)
    category = classify_pt_show(label)



