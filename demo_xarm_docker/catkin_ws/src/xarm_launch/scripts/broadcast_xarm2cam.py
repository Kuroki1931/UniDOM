#!/usr/bin/env python3
import os 
import yaml

import rospy
import rospkg

import tf

import tf2_ros
import geometry_msgs.msg

if __name__ == '__main__':
    rospy.init_node('world2realsense')

    camera_name = rospy.get_param("~CAMERA_NAME")
    
    calib_path = os.path.join(rospkg.RosPack().get_path('xarm_launch'),
            "config", "xarm2cam_" + camera_name[0:11]  + ".yml")

    with open(calib_path, "r") as f:
        data = yaml.load(f)
        calib_trans = data["xarm2cam"]

    world2realsense = tf2_ros.StaticTransformBroadcaster()
    static_tf = geometry_msgs.msg.TransformStamped()
    static_tf.header.stamp = rospy.Time.now()
    if camera_name == "eye_to_hand_camera":
        static_tf.header.frame_id = "link_base"
    elif camera_name == "eye_on_hand_camera":
        static_tf.header.frame_id = "link7"

    static_tf.child_frame_id = camera_name + "_link"
    static_tf.transform.translation.x = calib_trans[0]
    static_tf.transform.translation.y = calib_trans[1]
    static_tf.transform.translation.z = calib_trans[2]
    static_tf.transform.rotation.x = calib_trans[3]
    static_tf.transform.rotation.y = calib_trans[4]
    static_tf.transform.rotation.z = calib_trans[5]
    static_tf.transform.rotation.w = calib_trans[6]
    world2realsense.sendTransform(static_tf)

    rospy.spin()
