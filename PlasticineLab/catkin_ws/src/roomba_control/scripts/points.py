#!/usr/bin/env python3
import os
import random
from std_msgs.msg import Header
import numpy as np
import rospy
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs import point_cloud2
import open3d as o3d
import matplotlib.pyplot as plt

tmp_pcd_name = '/root/xarm/tmp/tmp_cloud.pcd'

FIELDS = [
    PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
    PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
    PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
    PointField(name='rgb', offset=12, datatype=PointField.FLOAT32, count=1),
]


class Point():
    def __init__(self):
        rospy.Subscriber('/camera/depth/color/points', PointCloud2, self.pointcloud_callback)
        self.pub = rospy.Publisher('/output', PointCloud2, queue_size=1)

    def pointcloud_callback(self, msg):
        assert isinstance(msg, PointCloud2)
        pcd = self._convert_pcl(msg)
        pcd = pcd.voxel_down_sample(voxel_size=0.0017)
        pcd = self._clip(pcd)
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=1000000, std_ratio=2.0)
        self._for_cpdeform(pcd)
        self._publish_pointcloud(pcd, msg)
    
    def _convert_pcl(self, data):
        header = '''# .PCD v0.7 - Point Cloud Data file format
                    VERSION 0.7
                    FIELDS x y z rgb
                    SIZE 4 4 4 4
                    TYPE F F F F
                    COUNT 1 1 1 1
                    WIDTH %d
                    HEIGHT %d
                    VIEWPOINT 0 0 0 1 0 0 0
                    POINTS %d
                    DATA ascii'''

        with open(tmp_pcd_name, 'w') as f:
            f.write(header % (data.width, data.height, data.width*data.height))
            f.write("\n")
            for p in point_cloud2.read_points(data, skip_nans=True):
                f.write('%f %f %f %e' % (p[0], p[1], p[2], p[3]))
                f.write("\n")
            f.write("\n")
        pcd = o3d.io.read_point_cloud(tmp_pcd_name)
        return pcd
    
    def _clip(self, pcd):
        points = np.asarray(pcd.points)
        pcd = pcd.select_by_index(np.where(points[:, 0] < 0.13)[0])
        points = np.asarray(pcd.points)
        pcd = pcd.select_by_index(np.where(points[:, 0] > -0.11)[0])
        points = np.asarray(pcd.points)
        pcd = pcd.select_by_index(np.where(points[:, 1] < 0.1)[0])
        points = np.asarray(pcd.points)
        pcd = pcd.select_by_index(np.where(points[:, 1] > -0.05)[0])
        points = np.asarray(pcd.points)
        pcd = pcd.select_by_index(np.where(points[:, 2] < 0.250)[0])
        points = np.asarray(pcd.points)
        pcd = pcd.select_by_index(np.where(points[:, 2] > 0.18)[0])
        return pcd

    def _for_cpdeform(self, pcd):
        num = 10000
        points = np.asarray(pcd.points)
        points_n = points.shape[0]
        vertical_n = num // points_n
        
        points_list = []
        for p in points:
            for i in np.linspace(p[2], 0.265, vertical_n):
                points_list.append([p[0], p[1], i])
        
        add_points = random.sample(points_list, num-len(points_list))
        points_list.extend(add_points)
        points_list = np.array(points_list)
        # points_list += 0.0005 * np.random.rand(num, 3)

        points_list = np.vstack([points_list[:, 1], points_list[:, 2], points_list[:, 0]]).T
        points_list = 0.05 * (np.array([100, -100, -100]) * points_list + np.array([7, 27, 13]))
        print(points_n, points_list.min(axis=0), points_list.max(axis=0))
        np.save('/root/xarm/diff_phys/multistage_box/v1/target_particles', points_list)
        np.save('/root/xarm/diff_phys/multistage_box/v1/state', points_list)
        

    def _publish_pointcloud(self, output_data, input_data):
        # convert pcl data format
        pc_p = np.asarray(output_data.points)
        pc_c = np.asarray(output_data.colors)
        tmp_c = np.c_[np.zeros(pc_c.shape[1])]
        tmp_c = np.floor(pc_c[:,0] * 255) * 2**16 + np.floor(pc_c[:,1] * 255) * 2**8 + np.floor(pc_c[:,2] * 255) # 16bit shift, 8bit shift, 0bit shift
        pc_pc = np.c_[pc_p, tmp_c]
        # publish point cloud
        output = point_cloud2.create_cloud(Header(frame_id=input_data.header.frame_id), FIELDS , pc_pc)
        self.pub.publish(output)

if __name__ == '__main__':
    rospy.init_node('special_node', log_level=rospy.DEBUG)
    Point()
    while not rospy.is_shutdown():
        rospy.sleep(1)