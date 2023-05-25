import pickle
import rospy
import cv2
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import CameraInfo, Image

import numpy as np
import std_msgs
import tf

from sensor_msgs.msg import PointCloud2, PointField, CameraInfo
from std_msgs.msg import Header




def get_info(tfl, stamp=None, wait=True, target_frame=None):
    hdr = std_msgs.msg.Header()
    hdr.stamp = rospy.Time.now() if stamp is None else stamp

    hdr.frame_id = 'side_camera_color_optical_frame'
    K = [907.1078491210938, 0.0, 644.1558227539062, 0.0, 906.809814453125, 367.2300720214844, 0.0, 0.0, 1.0]

    if wait:
        tfl.waitForTransform('/' + target_frame, '/' + hdr.frame_id, hdr.stamp, rospy.Duration(5.0))

    mat = tfl.asMatrix('/' + target_frame, hdr)

    return K, mat

def depth2cam(dx, dy, dz, K):
    cx = (dx - K[2]) * dz / K[0]
    cy = (dy - K[5]) * dz / K[4]

    return cx, cy, dz

def depth2world(d_img, tfl, stamp=None, wait=True, target_frame=None, A=None):
    K, mat = get_info(tfl, stamp=stamp, wait=wait, target_frame=target_frame)

    if A is not None:
        mat = A.dot(mat)

    cz = d_img.flatten()
    ix, iy = np.meshgrid(np.arange(1280), np.arange(720))
    ix, iy = ix.flatten(), iy.flatten()

    cx, cy, _ = depth2cam(ix, iy, cz, K)

    X = np.column_stack([cx, cy, cz, np.ones_like(cx)])
    X = X.dot(mat.T)

    wx, wy, wz = np.split(X[:, :3], 3, 1)
    wx, wy, wz = wx.reshape(720, 1280), wy.reshape(720, 1280), wz.reshape(720, 1280)
    w = np.dstack([wx, wy, wz])

    return w


def main():
    tfl = tf.TransformListener()
    rospy.sleep(1.0)

    import matplotlib.pyplot as plt
    time_list = []
    x_list = []
    z_list = []

    with open("../data/x_0.1y_0.2theta45_20230525_171814.pkl", "rb") as f:
        data = pickle.load(f)
        print(data["rgb"][0].shape)
        print(data["depth"][0].shape)
        print(data["camera_info"])
        print(data["delta_x"])
        print(data["delta_y"])
        print(data["delta_theta"])

    for i in range(len(data["depth"])):
        rgb = data["rgb"][i]
        cv2.imwrite("rgb"+str(i)+".png", rgb)
        d = data["depth"][i]
        # depth threshold 0.2m
        d = d/1000
        # d[d == 0] = d.max()
        cv2.imwrite("depth"+str(i)+".png", d/d.max() * 255)
        # d[d > 0.3] = 0
        xyz = depth2world(d, tfl, target_frame='ground')


        mask = np.zeros((xyz.shape[0], xyz.shape[1]))
        x_range = (0.32, 0.52)
        y_range = (-0.35, -0.25)
        z_range = (0.0, 0.2)
        for j in range(xyz.shape[0]):
            for k in range(xyz.shape[1]):
                if xyz[j, k, 0] > x_range[0] and xyz[j, k, 0] < x_range[1]:
                    if xyz[j, k, 1] > y_range[0] and xyz[j, k, 1] < y_range[1]:
                        if xyz[j, k, 2] > z_range[0] and xyz[j, k, 2] < z_range[1]:
                            mask[j, k] = 1
        xyz = xyz[d != 0].reshape(-1, 3)
        xyz = xyz[np.logical_and(xyz[:, 0] > x_range[0], xyz[:, 0] < x_range[1])]
        xyz = xyz[np.logical_and(xyz[:, 1] > y_range[0], xyz[:, 1] < y_range[1])]
        xyz = xyz[np.logical_and(xyz[:, 2] > z_range[0], xyz[:, 2] < z_range[1])]

        cv2.imwrite("mask"+str(i)+".png", mask*255)

        if xyz.shape[0] > 0:
            print(i)
            x_min = np.min(xyz[:, 0])
            z_min = np.min(xyz[:, 2])
            x_list.append(x_min)
            z_list.append(z_min)
            time_list.append(i)
            print(x_min, z_min)

    plt.plot(time_list, x_list)

    plt.show()


    #     rospy.Subscriber("/over_camera/aligned_depth_to_color/image_raw", Image, self.depth_callback)

    #     self.cv_array = None
        
    #     while not rospy.is_shutdown():
    #         if self.cv_array is not None:
    #             d = self.cv_array/1000
    #             xyz = depth2world(d, tfl, stamp=self.msg.header.stamp, target_frame='L_link_base')
    #             x_range = (0.4, 0.6)
    #             z_range = (-0.15, 0.20)
    #             xyz = xyz[d != 0].reshape(-1, 3)

    #             xyz = xyz[np.logical_and(xyz[:, 0] > x_range[0], xyz[:, 0] < x_range[1])]

    #             print(xyz)
    #             # xyz = xyz[np.logical_and(xyz[:, 2] > z_range[0], xyz[:, 2] < z_range[1])]

    #             if xyz.shape[0] > 0:
    #                 x_min, x_max = np.min(xyz[:, 0]), np.max(xyz[:, 0])
    #                 y_min, y_max = np.min(xyz[:, 1]), np.max(xyz[:, 1])
    #                 z_min, z_max = np.min(xyz[:, 2]), np.max(xyz[:, 2])
    #                 print("x", x_min, x_max)
    #                 print("y", y_min, y_max)
    #                 print("z", z_min, z_max)

    # def depth_callback(self, msg):
    #     self.msg = msg
    #     self.cv_array = CvBridge().imgmsg_to_cv2(msg, 'passthrough')


        

if __name__ == '__main__':
    rospy.init_node('process')
    main()