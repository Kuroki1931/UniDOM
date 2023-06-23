#!/usr/bin/env python3
import rospy
import tf
import math
# from xarm_gripper.msg import MoveActionGoal
from moveit_commander import MoveGroupCommander
import geometry_msgs.msg
from sensor_msgs.msg import Image, JointState
from geometry_msgs.msg import Point, PointStamped, PoseStamped, WrenchStamped, Quaternion, Pose, Vector3
from xarm_msgs.srv import MoveVelocity, MoveVelocityResponse, GripperMove, GripperMoveResponse, SetInt16, SetFloat32, SetAxis, Move, ClearErr, GripperConfig, GripperState
from std_srvs.srv import Trigger, TriggerResponse
from sensor_msgs.msg import CameraInfo, Image
import cv2
from cv_bridge import CvBridge, CvBridgeError
import time
import pickle
import copy
import message_filters


class Setup:
    def __init__(self):
        self.move_vel = rospy.ServiceProxy('/L_xarm7/velo_move_joint_timed', MoveVelocity)
        self.l_gripper_control = rospy.ServiceProxy('/L_xarm7/gripper_move', GripperMove)
        self.l_gripper_state = rospy.ServiceProxy('/L_xarm7/gripper_state', GripperState)
        self.l_gripper_speed = rospy.ServiceProxy('/L_xarm7/gripper_config', GripperConfig)
        self.l_set_xarm_mode = rospy.ServiceProxy('/L_xarm7/set_mode', SetInt16)
        self.l_set_xarm_state = rospy.ServiceProxy('/L_xarm7/set_state', SetInt16)
        self.l_set_xarm_ctrl = rospy.ServiceProxy('/L_xarm7/motion_ctrl', SetAxis)
        self.l_xarm_move_line = rospy.ServiceProxy('/L_xarm7/move_line', Move)

        self.r_gripper_control = rospy.ServiceProxy('/R_xarm7/gripper_move', GripperMove)
        self.r_gripper_state = rospy.ServiceProxy('/R_xarm7/gripper_state', GripperState)
        self.r_gripper_speed = rospy.ServiceProxy('/R_xarm7/gripper_config', GripperConfig)
        self.r_set_xarm_mode = rospy.ServiceProxy('/R_xarm7/set_mode', SetInt16)
        self.r_set_xarm_state = rospy.ServiceProxy('/R_xarm7/set_state', SetInt16)
        self.r_set_xarm_ctrl = rospy.ServiceProxy('/R_xarm7/motion_ctrl', SetAxis)
        self.r_xarm_move_line = rospy.ServiceProxy('/R_xarm7/move_line', Move)

        self.l_arm = MoveGroupCommander('L_xarm7')
        self.r_arm = MoveGroupCommander('R_xarm7')

        self.l_gripper_speed(5000)
        self.r_gripper_speed(5000)
        self.tflistener = tf.TransformListener()

        self.rgb_buffer, self.depth_buffer = [], []
        self.camera_info = None
        self.bridge = CvBridge()

        rospy.sleep(1.0)

        # unnecessary to sync topic since we use only depth image to calc
        rgb_sub = message_filters.Subscriber('/side_camera/color/image_raw', Image)
        depth_sub = message_filters.Subscriber('/side_camera/aligned_depth_to_color/image_raw', Image)
        message_filters.ApproximateTimeSynchronizer([rgb_sub, depth_sub], 10, 0.01).registerCallback(self.callback_rgbd)
        # rospy.Subscriber("/side_camera/aligned_depth_to_color/image_raw", Image, self.depth_callback)
        # rospy.Subscriber("/side_camera/color/image_raw", Image, self.rgb_callback)
        msg = rospy.wait_for_message("/side_camera/aligned_depth_to_color/camera_info", CameraInfo)
        self.camera_info = msg

        rospy.loginfo("service server initialized")

    def depth_callback(self, msg):
        cv_array = self.bridge.imgmsg_to_cv2(msg, 'passthrough')
        self.depth_buffer.append(cv_array)
        if len(self.depth_buffer) > 1000:
            self.depth_buffer = []

    def rgb_callback(self, msg):
        cv_array = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        cv_array = cv2.cvtColor(cv_array, cv2.COLOR_BGR2RGB)
        self.rgb_buffer.append(cv_array)
        if len(self.rgb_buffer) > 1000:
            self.rgb_buffer = []

    def callback_rgbd(self, data1, data2):
        cv_array = self.bridge.imgmsg_to_cv2(data1, 'bgr8')
        cv_array = cv2.cvtColor(cv_array, cv2.COLOR_BGR2RGB)
        self.rgb_buffer.append(cv_array)
        print("rgb callback")

        cv_array = self.bridge.imgmsg_to_cv2(data2, 'passthrough')
        self.depth_buffer.append(cv_array)
        print("depth callback")


        
    def clear_buffer(self):
        self.rgb_buffer, self.depth_buffer = [], []
    
    def move_end_effector_by_pose(self, x, y, z, roll=3.14, pitch=0.0, yaw=0.0, target='left'):
        assert target in ['left', 'right']

        if target == 'left':
            self.l_set_xarm_ctrl(8, 1)
            self.l_set_xarm_mode(0)
            self.l_set_xarm_state(0)
            res = self.l_xarm_move_line([x*1000,y*1000,z*1000, roll, pitch, yaw], 100, 2000, 0, 0)
        else:
            self.r_set_xarm_ctrl(8, 1)
            self.r_set_xarm_mode(0)
            self.r_set_xarm_state(0)
            res = self.r_xarm_move_line([x*1000,y*1000,z*1000, roll, pitch, yaw], 100, 2000, 0, 0)


    def is_in_area(self, area_pos, target='left', threshold=0.01):
        if target == 'left':
            ee_pos = self.l_arm.get_current_pose()
        else:
            ee_pos = self.r_arm.get_current_pose()

        if abs(ee_pos.pose.position.x - area_pos[0]) < 0.01 and abs(ee_pos.pose.position.y - area_pos[1]) < 0.01 and abs(ee_pos.pose.position.z - area_pos[2]) < 0.01:
            return True
        return False

    def is_grasp(self, target='left'):
        assert target in ['left', 'right']

        if target == 'left':
            res_pos = self.l_gripper_state()
        else:
            res_pos = self.r_gripper_state()

        result = False
        if res_pos.curr_pos > 10:
            result = True
        return result
    
    def move_gripper(self, command, target='left'):
        assert command in ['open', 'close']
        assert target in ['left', 'right']

        if target == 'left':
            if command == 'open':
                self.l_gripper_control(850)
            else:
                self.l_gripper_control(0)
        else:
            if command == 'open':
                self.r_gripper_control(850)
            else:
                self.r_gripper_control(0)

    def reset(self):
        self.move_end_effector_by_pose(0.35, 0, 0.08, 1.57, 1.57, 1.57, target='right')
        self.move_end_effector_by_pose(0.45, -0.28, 0.5, target='left')

        self.move_gripper('close', target='left')
        while not rospy.is_shutdown() and not self.is_grasp(target='left'):
            self.move_gripper('open', target='left')
            rospy.sleep(1.0)
            self.move_gripper('close', target='left')
            rospy.sleep(1.0)

        self.move_gripper('open', target='right')

    def grab_and_release(self, delta_x, delta_y, delta_theta=45):
        delta_rad = delta_theta / 180.0 * math.pi
        # self.clear_buffer()
        rospy.sleep(1.0)
        self.move_gripper('open', target='right')
        self.move_end_effector_by_pose(0.43, 0, 0.08, 1.57, 1.57, 1.57, target='right')
        self.move_gripper('close', target='right')
        rospy.sleep(1.0)
        x_now, z_now = 0.43, 0.07
        self.move_end_effector_by_pose(x_now - delta_x, 0, z_now + delta_y, 0, 1.57+delta_rad, 0, target='right')
        self.move_gripper('open', target='right')
        rospy.sleep(3.0)
        save_dict = {"rgb": copy.deepcopy(self.rgb_buffer[-100:]), "depth": copy.deepcopy(self.depth_buffer[-100:]), "camera_info": self.camera_info,
                     "delta_x": delta_x, "delta_y": delta_y, "delta_theta": delta_theta}
        datetime = time.strftime("%Y%m%d_%H%M%S")
        with open("../data/x_"+str(delta_x)+"y_"+str(delta_y)+"theta"+str(delta_theta)+"_"+datetime+".pkl", "wb") as f:
            pickle.dump(save_dict, f)
            print("saved")


if __name__ == '__main__':
    rospy.init_node('service_server')
    s = Setup()