import matplotlib.pyplot as plt
import numpy as np


def show_channel_input(channel,size):

    # plt.figure()
    # plt.title("input channel-0: top height of each grid")
    # image1 = np.zeros([640, 640, 3])
    # image1 = channel[:,:,0] * 255 / 10
    # plt.imshow(image1)

    plt.figure()
    plt.title("input channel-1: mean height of each grid")
    image2 = np.zeros([640, 640, 3])
    image2 = channel[:, :, 1] * 255 / 10
    plt.imshow(image2)

    # plt.figure()
    # plt.title("input channel-2: point num of each grid")
    # image3 = np.zeros([640, 640, 3])
    # x, y = np.where(channel[:, :, 2] > 0)
    # image3[x, y, :] = 255
    # plt.imshow(image3)
    #
    # plt.figure()
    # plt.title("input channel-4: top intensity of each grid")
    # image5 = np.zeros([640, 640])
    # # x, y = np.where(channel[:, :, 4] > 0)
    # image5[:, :] = channel[:,:,4]
    # plt.imshow(image5)
    #
    # plt.figure()
    # plt.title("input channel-5: mean intensity of each grid")
    # image6 = np.zeros([640, 640])
    # # x, y = np.where(channel[:, :, 4] > 0)
    # image6[:, :] = channel[:,:,5]
    # plt.imshow(image6)
    #
    # plt.figure()
    # plt.title("input channel-7: nonempty of each grid")
    # image7 = np.zeros([640, 640])
    # # x, y = np.where(channel[:, :, 4] > 0)
    # image7[:, :] = channel[:, :, 7]
    # plt.imshow(image7)
    #
    #
    plt.show()

def show_channel_label(channel, size):

    plt.figure()
    plt.title("label channel-(1:2): center offset")
    x = channel[:,:,1]
    y = channel[:,:,2]
    off_set_x = np.where(x != 0)
    off_set_y = np.where(y != 0)
    a = x[off_set_x]

    X, Y = np.meshgrid(np.arange(0, 1280, 2), np.arange(0, 1280, 2))
    Q = plt.quiver(X, Y, x[off_set_x], y[off_set_y], units='xy')
    qk = plt.quiverkey(Q, 0.9, 0.9, 2, r'$2 \frac{m}{s}$', labelpos='E',
                       coordinates='data')
    # X, Y = np.meshgrid(np.arange(0, 640), np.arange(0, 640))
    plt.scatter(X, Y, c='w', marker='o')

    plt.show()




