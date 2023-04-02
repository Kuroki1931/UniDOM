#!/usr/bin/env python3
import os
import yaml

import rospy
import rospkg

import tf2_ros
import geometry_msgs.msg
from std_srvs.srv import Empty, EmptyResponse

class GetEndeffPos:
    def __init__(self):
        # Is node_name 'saving_calibration_result' overlap a problem or not?
        rospy.init_node("get_endeff_pos")
        

        # parent frame
        self.parent_frame_id = "world"
        
        # child frame
        self.child_frame_id = "link7"

        # tf buffer and listener
        self.tfBuffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tfBuffer)

        while not rospy.is_shutdown():
            try:
                trans = self.tfBuffer.lookup_transform(self.parent_frame_id, self.child_frame_id, rospy.Time())
                print(trans.transform.translation)
            except (tf2_ros.LookupException,
                tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as e:
                rospy.logerr(e)
        return EmptyResponse()


if __name__ == '__main__':
    saving_calibration_result = GetEndeffPos()
    rospy.spin()
