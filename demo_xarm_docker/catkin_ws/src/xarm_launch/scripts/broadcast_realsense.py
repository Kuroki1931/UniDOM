#!/usr/bin/env python3
import rospy

import tf

import tf2_ros
import geometry_msgs.msg

if __name__ == '__main__':
    rospy.init_node('realsense_tf')

    optical2camera = tf2_ros.StaticTransformBroadcaster()
    static_tf_realsense = geometry_msgs.msg.TransformStamped()

    camera_name = rospy.get_param("~CAMERA_NAME")
    
    static_tf_realsense.header.stamp = rospy.Time.now()
    
    static_tf_realsense.header.frame_id = camera_name + "_color_optical_frame"
    static_tf_realsense.child_frame_id = camera_name + "_link"
    # static_tf_realsense.header.frame_id = "realsense_color_optical_frame"
    # static_tf_realsense.child_frame_id = "camera_link"

    static_tf_realsense.transform.translation.x = 0.0147052564039
    static_tf_realsense.transform.translation.y = 0.00012443851978
    static_tf_realsense.transform.translation.z = 0.000286161684449
    static_tf_realsense.transform.rotation.x = 0.506091553718
    static_tf_realsense.transform.rotation.y = -0.496781819034
    static_tf_realsense.transform.rotation.z = 0.499370179605
    static_tf_realsense.transform.rotation.w = 0.4977032803

    optical2camera.sendTransform(static_tf_realsense)
    rospy.spin()

# header: 
#   seq: 0
#   stamp: 
#     secs: 0
#     nsecs:         0
#   frame_id: "camera_color_optical_frame"
# child_frame_id: "camera_link"
# transform: 
#   translation: 
#     x: 0.0147052564039
#     y: 0.00012443851978
#     z: 0.000286161684449
#   rotation: 
#     x: 0.506091553718
#     y: -0.496781819034
#     z: 0.499370179605
#     w: 0.4977032803
