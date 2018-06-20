import numpy as np

import glob
import cv2
import os
import rosbag
import sensor_msgs.point_cloud2 as pc2


def F2I(val, ori, scale):
    return np.floor((ori - val) * scale)

def LogCount(count):
    log_table_ = np.arange(0, 256+1)
    log_table_ = np.log1p(1+log_table_)
    if count < len(log_table_):
        return log_table_[count]
    return np.log(1+count)

def count_data(bin, max_height, min_height, inv_res_x, inv_res_y, range_, width, height):

    #compute for vaild points
    channel_map = np.zeros([640, 640], np.float64)
    idx = np.where(bin[:, 2] < max_height)
    bin = bin[idx]
    idx = np.where(bin[:, 2] > min_height)
    bin = bin[idx]

    for i in range(len(bin)):
        pos_x = int(F2I(bin[i, 1], range_, inv_res_x))  #col
        pos_y = int(F2I(bin[i, 0], range_, inv_res_y))  #row

        if(pos_x >= width or pos_x < 0 or pos_y >= height or pos_y < 0):continue
        channel_map[pos_y,pos_x] += 1.     #count data

    for i in range(width):
        for j in range(height):
            channel_map[i,j] = LogCount(int(channel_map[i, j]))
    return channel_map

def start_process(path):
    bag = rosbag.Bag(os.path.join(path, "2018-06-19-14-42-14_3.bag"))
    num = bag.get_message_count()
    it = bag.read_messages(['/sensor/merge/rslidar_points'])
    for a in it:
        am = a.message
        secs = str(a.message.header.stamp.secs)
        nsecs = str(a.message.header.stamp.nsecs)

        ii = pc2.read_points(am)
        pts = [a for a in ii]
        points = np.asarray(pts)


        inv_res_x = 0.5 * 640 / 60  # length of each grid(x: meters)
        inv_res_y = 0.5 * 640 / 60  # length of each grid(y: meters)
        channel = count_data(points, 5, -5, inv_res_x, inv_res_y, 60, 640, 640)
        channel *= 30
        print(os.path.join(path, "_03/image/{}.png".format(secs + '.' + nsecs.zfill(9))))
        cv2.imwrite(os.path.join(path, "_03/image/{}.png".format(secs + '.' + nsecs.zfill(9))), channel.astype(np.uint8))
        print("save done++")

if __name__ == "__main__":
    path = "/home/bai/bag"
    # pcd_id = glob.glob('%s/*.pcd' % (path))
    # for pcd in pcd_id[:1]:
    #     print(pcd)
    start_process(path=path)
        # print("process done!")

    # bag = rosbag.Bag('/home/bai/bag/2018-06-19-14-55-54_4.bag')
    # num = bag.get_message_count()
    # it = bag.read_messages(['/sensor/merge/rslidar_points'])
    # for a in it:
    #     am = a.message
    #     secs = str(a.message.header.stamp.secs)
    #     nsecs = str(a.message.header.stamp.nsecs)
    #
    #     ii = pc2.read_points(am)
    #     pts = [a for a in ii]
    #     points = np.asarray(pts)
    #     print(points.shape)
    #     print(secs + '.' + nsecs.zfill(9))