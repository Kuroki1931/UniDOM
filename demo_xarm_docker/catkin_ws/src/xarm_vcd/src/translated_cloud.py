#!/usr/bin/env python3
# coding: UTF-8
# from xmlrpc.client import Boolean
import numpy as np
import rospy
import argparse
import sensor_msgs.point_cloud2 as pc2
import tf2_ros
import tf2_py as tf2
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2
from tf2_sensor_msgs.tf2_sensor_msgs import do_transform_cloud
import time
from std_msgs.msg import Bool


class Vcd():
    def __init__(self, args):
        self.ds_cloud = None
        self.is_in_area = False
        rospy.Subscriber('/downsampled_cloud', PointCloud2, self.dscloud_cb,queue_size=1)
        self.pub = rospy.Publisher('/translated_cloud', PointCloud2, queue_size=1)
        rospy.Subscriber('is_in_area', Bool, self.callback_is_in_area)
        self.tf_buffer = tf2_ros.Buffer()

    def callback_is_in_area(self,bool):
        self.is_in_area = bool


    def dscloud_cb(self, data):
        self.ds_cloud = data

    def get_edge(self, pc):

        # tf_buffer = tf2_ros.Buffer()
        tf_buffercore = tf2_ros.BufferCore()
        self.tf_buffer = tf2_ros.Buffer() #一度tf_bufferを初期化しないとrviz上でmoveitを動かしたときに点群の座標が移動前のままになる
        tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        target_frame = 'world'
        # print(pc.header.stamp)
        # print(tf_buffercore.canTransform(target_frame, pc.header.frame_id))

        try:
            #print('pc.header.frame_id: ', pc.header.frame_id)
            trans = self.tf_buffer.lookup_transform(target_frame, pc.header.frame_id, rospy.Time(0), rospy.Duration(0.5)) #rospy.Time(0), rospy.Duration(5))#pc.header.stamp, rospy.Duration(1))
        except tf2.LookupException as ex:
            rospy.logwarn(ex)
            return
        except tf2.ExtrapolationException as ex:
            rospy.logwarn(ex)
            return

        cloud_out = do_transform_cloud(pc, trans)
        print("cloud_published to vcd")
        self.pub.publish(cloud_out)




    def main(self):
        rospy.init_node('trans_cloud')

        while not rospy.is_shutdown():
            while self.ds_cloud is None:
                rospy.sleep(1)
                print('waiting for realsense......')
            if self.is_in_area:
                current_cloud = self.ds_cloud
                self.get_edge(current_cloud)




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='trans_cloud')
    args, unknown = parser.parse_known_args()
    # args = parser.parse_args()
    vcd = Vcd(args)
    vcd.main()
