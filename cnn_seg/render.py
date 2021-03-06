import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

def show_channel_input(channel, size):

    plt.figure()
    plt.title("input channel-0: top height of each grid")
    image1 = np.zeros([640, 640, 3])
    image1 = channel[:,:,0] * 255 / 10
    plt.imshow(image1)

    plt.figure()
    plt.title("input channel-1: mean height of each grid")
    image2 = np.zeros([640, 640, 3])
    image2 = channel[:, :, 1] * 255 / 10
    plt.imshow(image2)

    plt.figure()
    plt.title("input channel-2: point num of each grid")
    image3 = np.zeros([640, 640, 3])
    x, y = np.where(channel[:, :, 2] >= 1)
    image3[x, y, :] = (255, 0, 255)
    plt.imshow(image3)

    plt.figure()
    plt.title("input channel-4: top intensity of each grid")
    image5 = np.zeros([640, 640])
    # x, y = np.where(channel[:, :, 4] > 0)
    image5[:, :] = channel[:,:,4]
    plt.imshow(image5)

    plt.figure()
    plt.title("input channel-5: mean intensity of each grid")
    image6 = np.zeros([640, 640])
    # x, y = np.where(channel[:, :, 4] > 0)
    image6[:, :] = channel[:,:,5]
    plt.imshow(image6)

    plt.figure()
    plt.title("input channel-7: nonempty of each grid")
    image7 = np.zeros([640, 640])

    image7[:, :] = channel[:, :, 7]
    plt.imshow(image7)
    plt.show()

def show_channel_label(channel, size):

    plt.figure()
    plt.title("label channel-(1:2): center offset")
    x = channel[:,:,1]
    y = channel[:,:,2]
    off_set_x = np.where(x != 0)
    off_set_y = np.where(y != 0)
    # a = x[off_set_x]
    # x = np.cos(x)
    # y = np.sin(y)
    X, Y = np.meshgrid(np.arange(0, 640, 1), np.arange(0, 640, 1))
    M = np.hypot(x, y)
    Q = plt.quiver(X, Y, x, y, M, pivot="tip", units='xy')
    qk = plt.quiverkey(Q, 0.9, 0.9, 2, r'$2 \frac{m}{s}$', labelpos='E',
                       coordinates='data')
    # X, Y = np.meshgrid(np.arange(0, 640), np.arange(0, 640))
    # plt.scatter(X, Y, c='w', marker='o')

    plt.show()
    

def record_confirm(channel, label, name):
    #top height
    top_height = channel[0,:,:,0] * 100
    cv2.imwrite(os.path.join(os.getcwd(), \
                             'testpng/{}--in-{}.png'.format(name, 1)),top_height.astype(np.uint8))
    
    
    #mean height
    mean_height = channel[0,:,:,1] * 100
    cv2.imwrite(os.path.join(os.getcwd(), \
                             'testpng/{}--in-{}.png'.format(name, 2)),mean_height.astype(np.uint8))
    
    #num points
    num_points = channel[0,:,:,2] * 100
    cv2.imwrite(os.path.join(os.getcwd(), \
                             'testpng/{}--in-{}.png'.format(name, 3)),num_points.astype(np.uint8))
    
    #top intensity
    top_intensity = channel[0,:,:,4] * 100000
    cv2.imwrite(os.path.join(os.getcwd(), \
                             'testpng/{}--in-{}.png'.format(name, 5)),top_intensity.astype(np.uint8))
    
    #mean intensity
    mean_intensity = channel[0,:,:,5] * 100000
    cv2.imwrite(os.path.join(os.getcwd(), \
                             'testpng/{}--in-{}.png'.format(name, 6)),mean_intensity.astype(np.uint8))
    
    #nonempty
    nonempty = channel[0,:,:,7] * 100
    cv2.imwrite(os.path.join(os.getcwd(), \
                             'testpng/{}--in-{}.png'.format(name, 8)),nonempty.astype(np.uint8))
    
    #label
    #objectness
    objectness = label[0,:,:,0] * 100
    cv2.imwrite(os.path.join(os.getcwd(), \
                             'testpng/{}--label-{}.png'.format(name, 1)),objectness.astype(np.uint8))
    
    #centeroffset_x
    centeroffset_x = label[0,:,:,1] * 100
    cv2.imwrite(os.path.join(os.getcwd(), \
                             'testpng/{}--label-{}.png'.format(name, 2)),centeroffset_x.astype(np.uint8))
    
    #centeroffset_y
    centeroffset_y = label[0,:,:,2] * 100
    cv2.imwrite(os.path.join(os.getcwd(), \
                             'testpng/{}--label-{}.png'.format(name, 3)),centeroffset_x.astype(np.uint8))
    
    #cofidence
    confidence = label[0,:,:,3] * 100
    cv2.imwrite(os.path.join(os.getcwd(), \
                             'testpng/{}--label-{}.png'.format(name, 4)),confidence.astype(np.uint8))
    
    #small_car
    small_car = label[0,:,:,5] * 100
    cv2.imwrite(os.path.join(os.getcwd(), \
                             'testpng/{}--label-{}.png'.format(name, 6)),small_car.astype(np.uint8))
    
    #big car
    big_car = label[0,:,:,6] * 100
    cv2.imwrite(os.path.join(os.getcwd(), \
                             'testpng/{}--label-{}.png'.format(name, 7)),big_car.astype(np.uint8))
    
    #bicycle
    bicycle = label[0,:,:,7] * 100
    cv2.imwrite(os.path.join(os.getcwd(), \
                             'testpng/{}--label-{}.png'.format(name, 8)),bicycle.astype(np.uint8))
    
    #persion
    persion = label[0,:,:,8] * 100
    cv2.imwrite(os.path.join(os.getcwd(), \
                             'testpng/{}--label-{}.png'.format(name, 9)),persion.astype(np.uint8))
    
    #height
    h = label[0,:,:,11] * 100
    cv2.imwrite(os.path.join(os.getcwd(), \
                             'testpng/{}--label-{}.png'.format(name, 12)),h.astype(np.uint8))
    
    
def split(feature, name):
    feature = feature[:, :, :, :] * 100
    for i in range(feature.shape[0]):
        for j in range(feature.shape[3]):

            cv2.imwrite(os.path.join(os.getcwd(), \
                                 'tmp/{}-batch-{}-channel-{}.png'.format(name, i, j)),feature[i,:,:,j].astype(np.uint8))
    
    
    
    
    
    
    


