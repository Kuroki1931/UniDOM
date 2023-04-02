#!/usr/bin/env python
import os
import yaml

import rospy
import rospkg

import tf2_ros
import geometry_msgs.msg
from std_srvs.srv import Empty, EmptyResponse

class SavingCalibrationResult:
    def __init__(self):
        # Is node_name 'saving_calibration_result' overlap a problem or not?
        rospy.init_node("saving_calibration_result")

        # param
        self.parent_frame_id = rospy.get_param('~PARENT_FRAME_ID', "link_base")
        
        # get node_name
        self.node_name = rospy.get_param('node')

        # Does "eye_on_hand" realsense save the tf for xarm's end effector?? In that case, tf is static and OK!
        if self.node_name == 'eye_to_hand':
            self.child_frame_id = rospy.get_param('~CHILD_FRAME_ID', self.node_name + "_camera_color_optical_frame")
            self.save_path = os.path.join(rospkg.RosPack().get_path('xarm_launch'),
                                      "config", "xarm2cam_" + self.node_name + ".yml")
            srv_save_calibration_result = rospy.Service('/calibration/save_eye_to_hand', Empty, self.save_tf)

        elif self.node_name == 'eye_on_hand':
            self.child_frame_id = rospy.get_param('~CHILD_FRAME_ID', self.node_name + "_camera_color_optical_frame")
            self.save_path = os.path.join(rospkg.RosPack().get_path('xarm_launch'),
                                      "config", "xarm2cam_" + self.node_name + ".yml")
            srv_save_calibration_result = rospy.Service('/calibration/save_eye_on_hand', Empty, self.save_tf)
        # service
        # Is there a problem concerning of the same name of '/calibration/save' is used in eye_on_hand_save_calibration and eye_to_hand_save_calibration
        # srv_save_calibration_result = rospy.Service('/calibration/save', Empty, self.save_tf)
       
        # tf buffer and listener
        self.tfBuffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tfBuffer)
    
    def save_tf(self, req):
        try:
            trans = self.tfBuffer.lookup_transform(self.parent_frame_id, self.child_frame_id, rospy.Time())
            print(trans)

            data = {'xarm2cam':
                    [trans.transform.translation.x,
                     trans.transform.translation.y,
                     trans.transform.translation.z,
                     trans.transform.rotation.x,
                     trans.transform.rotation.y,
                     trans.transform.rotation.z,
                     trans.transform.rotation.w]
                    }
            with open(self.save_path, 'wt') as f:
                f.write(yaml.dump(data, default_flow_style=False))
        except (tf2_ros.LookupException,
                tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as e:
            rospy.logerr(e)
        return EmptyResponse()

if __name__ == '__main__':
    saving_calibration_result = SavingCalibrationResult()
    rospy.spin()
