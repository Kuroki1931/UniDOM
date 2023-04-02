#!/usr/bin/env python3
# coding: UTF-8
import argparse
import math
import sys
#from xmlrpc.client import Boolean
from sensor_msgs.msg import Image
import numpy as np
import rospy
from moveit_commander import MoveGroupCommander, roscpp_initialize
import moveit_commander
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import Quaternion
import moveit_msgs
import copy
from geometry_msgs.msg import Vector3

from xarm_vcd.srv import ActionTrigger, ActionTriggerResponse
from xarm_msgs.srv import SetInt16,SetAxis

from tf import TransformListener
import geometry_msgs
import copy
import tf
import tf_conversions

import rospy
import roslib
from roslib import message


from std_msgs.msg import (
    UInt16,
)
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2, PointField

import numpy

import time
import pickle
import struct

import cv2
from cv_bridge import CvBridge

from std_msgs.msg import Bool

class Planner(object):
    def __init__(self):
        self._pub_rate = rospy.Publisher('robot/joint_state_publish_rate',UInt16, queue_size=10)
        
        self.cloud_sub = rospy.Subscriber("/translated_cloud", PointCloud2, self.callback, queue_size=1, buff_size=52428800)

        self.tf_listener_ = TransformListener()
        self.sevice_server = rospy.Service("action_service", ActionTrigger, self.callback_service)

        self.tc2 = PointCloud2()
        self.sub_mask = rospy.Subscriber("/debug_mask", Image, self.callback_mask, queue_size=1)
        self.bridge= CvBridge()
        self.is_in_area_pub = rospy.Publisher("/is_in_area", Bool, queue_size=1)
        self.trial_cur= 0

        self.xarm_motion_ctrl = rospy.ServiceProxy('/xarm/motion_ctrl', SetAxis)
        self.xarm_set_mode = rospy.ServiceProxy('xarm/set_mode', SetInt16)
        self.xarm_set_state = rospy.ServiceProxy('xarm/set_state', SetInt16)

        # control parameters
        self._rate = 10.0

        # end effectorとして指定している座標系（link:right_l6）の原点と、グリッパーの先端とのズレを修正
        self.end_effector_gripper_distance = 0.002 #実際のxarmのグリッパーの先端とlink_tcpのズレを補正
        self.pg = geometry_msgs.msg.Pose()
        self.vcd_pos = geometry_msgs.msg.Pose()
        self.vcd_vec = geometry_msgs.msg.Vector3()
        self.points = []
        self.trial_num = 100
        # the position from which realsense camera is looking at the cloth.
        self.look_at_point = [0.37, 0.0, 0.50]

        # xarmでmoveitを使うための初期化

        print("Enable Robot...")
        self.enable_servo()
        self.xarm_set_mode(1)
        self.xarm_set_state(0)

        self.xarm7 = MoveGroupCommander('xarm7')
        self.xarm_gripper = MoveGroupCommander('xarm_gripper')
        self.robot = moveit_commander.RobotCommander()
        self.xarm7.set_max_velocity_scaling_factor(1.0)
        self.xarm_gripper.set_max_velocity_scaling_factor(1.0)
        self.xarm7.set_num_planning_attempts(2)
        self.xarm7.set_planning_time(10)

        #gripperを開く
        self.drive_gripper(close=False)

        # set joint state publishing to 10Hz
        self._pub_rate.publish(self._rate)

        self.display_trajectory_publisher = rospy.Publisher('/move_group/display_planned_path',
                                               moveit_msgs.msg.DisplayTrajectory,
                                               queue_size=20)

        # # cloth occupancy in the relasense image
        self.thresh_goal = 0.10

        # wait for gripper initialize
        rospy.sleep(1)

    def callback(self, ros_point_cloud):
        xyz = np.array([[0,0,0]])
        rgb = np.array([[0,0,0]])

        gen = pc2.read_points(ros_point_cloud, skip_nans=True)
        self.pc_list = list(gen)
        # self.get_point = True

        if hasattr(self,'get_point') and self.get_point:
            pw = '/root/xarm/catkin_ws/src/xarm_vcd/list_cloth_'+str(self.trial_cur) + '.txt'
            f = open(pw, 'wb')
            list_row = self.pc_list
            pickle.dump(list_row, f)
            self.get_point=False


    def callback_pos(self, msg):
        self.vcd_pos.position.x = msg.x
        self.vcd_pos.position.y = msg.y
        self.vcd_pos.position.z = msg.z

    def callback_vec(self, msg):
        self.vcd_vec.x = msg.x
        self.vcd_vec.y = msg.y
        self.vcd_vec.z = msg.z

    def callback_mask(self, msg):
        self.debug_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

    def enable_servo(self):
        self.xarm_motion_ctrl(8,1)

    def _reset_control_modes(self):
        rate = rospy.Rate(self._rate)
        self.drive_gripper(close=False)
        rospy.sleep(0.5)
        for _ in range(10):
            if rospy.is_shutdown():
                return False
            #xarmでモードを0にする:sawyerでの設定に習い、残してはおく
            #self.xarm_set_mode(0)
            #self.xarm_set_state(0)

            self._pub_rate.publish(100)
            rate.sleep()

    def set_neutral(self):
        """
        Sets both arms back into a neutral pose.
        """
        print("Moving to neutral pose...")

        #初期位置に戻す(初期化しないと各ジョイントの角度がどんどんずれてしまうため)
        self.xarm7.set_named_target("initial_position") 
        self.xarm7.go(wait=True)
        #rospy.sleep(0.4) # wait for the arm to get stable

        pose_goal = geometry_msgs.msg.Pose()
        pose_goal.orientation = Quaternion(x=1.0, y=0.0, z=0.0, w=0.0)
        #q_orig = numpy.linalg.norm( [pose_goal.orientation.x, pose_goal.orientation.y,pose_goal.orientation.z, pose_goal.orientation.w] )
        #print(q_orig)

        pose_goal.position.x = self.look_at_point[0]
        pose_goal.position.y = self.look_at_point[1]
        pose_goal.position.z = self.look_at_point[2]

        self.xarm7.set_goal_tolerance(0.001) #1mmの精度を要求(z方向に関しては自重もあって精度が悪い)
        self.xarm7.set_pose_target(pose_goal)
        plan0th = self.xarm7.go(wait=True)
        rospy.sleep(0.2)
        self.xarm7.stop()
        self.xarm7.clear_pose_targets()
        rospy.sleep(1.5) # wait for the arm to get stable
        wpose = self.xarm7.get_current_pose().pose
        print("look pose is \n" + str(wpose.position))
        #print("look oriantation is \n" + str(wpose.orientation))

    #グリッパーの開閉
    def drive_gripper(self,close):
        joint_goal = self.xarm_gripper.get_current_joint_values()
        # gripper open close level :0 is fully open, 0.85 is closed
        # ただし、0.85よりも微妙に大きくしてもより閉じるが、大きくしすぎるとjoint limitのせいかエラーが出る
        if close:
            joint_goal[0] = 0.85
        else:
            joint_goal[0] = 0.0

        self.xarm_gripper.go(joint_goal, wait=True)
        self.xarm_gripper.stop()

    def _set_hand_down(self):
        wpose = self.xarm7.get_current_pose().pose
        pose_goal = geometry_msgs.msg.Pose()

        pose_goal.orientation = Quaternion(x=1.0, y=0.0, z=0.0, w=0.0) #self.euler2quaternion(0.0, math.pi, 0.0)
        pose_goal.position = wpose.position
        pose_goal.position.z = wpose.position.z #- 0.3
        self.xarm7.set_goal_tolerance(0.001)
        self.xarm7.set_pose_target(pose_goal)
        rospy.sleep(0.1)
        plan = self.xarm7.go(wait=True)
        rospy.sleep(0.1)
        self.xarm7.stop()
        print("hand gets down")



    def linear_moveit_vcd(self):
        dir = self.vcd_vec
        print("vcd_vec:")
        print(self.vcd_vec)
        waypoints = []
        divisions = 5
        scale = 1.0/divisions
        wpose = copy.deepcopy(self.xarm7.get_current_pose().pose)
        for i in range(divisions):
            wpose.position.x += self.vcd_vec.x * scale
            wpose.position.y += self.vcd_vec.y * scale
            wpose.position.z += self.vcd_vec.z * scale
            waypoints.append(copy.deepcopy(wpose))

        self.xarm7.set_goal_tolerance(0.001)
        (plan, fraction) = self.xarm7.compute_cartesian_path(
                            waypoints,
                            0.01,
                            0.0
                            )

        self.xarm7.execute(plan, wait=True)
        self.xarm7.stop()

    def clean_shutdown(self):
        print("\nExiting example...")
        #return to normal
        self._reset_control_modes()

        return True

    def euler2quaternion(self,euler1,euler2,euler3):
        q = tf_conversions.transformations.quaternion_from_euler(euler1, euler2, euler3)
        return Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])

    def callback_service(self, req):
        res = ActionTriggerResponse()
        self.pose_goal = geometry_msgs.msg.Pose()
        self.pose_goal.orientation = Quaternion(x=1.0, y=0.0, z=0.0, w=0.0) #gripperは下向きで固定 #self.euler2quaternion(math.pi, 0.0, math.pi)
        self.q_orig = numpy.linalg.norm( [self.pose_goal.orientation.x, self.pose_goal.orientation.y,self.pose_goal.orientation.z, self.pose_goal.orientation.w] )
        print(self.q_orig)

        pos = Vector3()
        self.vcd_vec = Vector3()
        pos.x, pos.y, pos.z = req.position_x, req.position_y, req.position_z
        print("goal pos to be arrived:")
        print(pos)

        self.vcd_vec.x, self.vcd_vec.y, self.vcd_vec.z = req.vector_x, req.vector_y, req.vector_z

        #replace here to vcd output
        self.pose_goal.position = pos
        # グリッパーがある分だけ目標値を上げる
        self.pose_goal.position.z += self.end_effector_gripper_distance


        #軌道制約用のオブジェクトを設定
        self.scene.remove_world_object()
        self._set_table()
        #self._set_upper_obs() #上の障害物を設定。xarmでは上に必要なさそうなのでコメントアウト


        self._move_pose_goal() # 布をつかむ位置まで移動

        #布をつかむ
        self.drive_gripper(close=True)

        #rospy.sleep(2) #布が開くまで待機

        self.linear_moveit_vcd() #布を掴んだまま移動

        #布を離す
        self.drive_gripper(close=False)

        print("gripper opened")
        #rospy.sleep(1) # here it need some time for collision of gripper and cloth

        self.scene.remove_world_object()
        self._set_table()
        rospy.sleep(0.1)

        self.set_neutral()
        rospy.sleep(0.1)

        # グリッパーを開く(いらないはず)
        # self.drive_gripper(close=False)
        # rospy.sleep(2.0)
        
        self.service_done = True
        print("finish service")

        return res

    # publish translated_cloud only when the gripper is designated position
    def is_in_area(self, threshold=0.07):
        ee_pos = self.xarm7.get_current_pose()
        rospy.sleep(0.1)
        if abs(ee_pos.pose.position.x - self.look_at_point[0]) < threshold and abs(ee_pos.pose.position.y - self.look_at_point[1]) < threshold and abs(ee_pos.pose.position.z - self.look_at_point[2]) < threshold:
            return self.is_in_area_pub.publish(True)
        return self.is_in_area_pub.publish(False)


    def _setup_robot(self):
        self.rate = rospy.Rate(self._rate)
        self.start = rospy.Time.now()

        self.scene = moveit_commander.PlanningSceneInterface()
        self.robot = moveit_commander.RobotCommander()

    # set obstacle upper sawyer for narrowing sawyer workspace
    def _set_upper_obs(self):
        upper = PoseStamped()
        upper.header.frame_id = self.xarm7.get_planning_frame()
        upper.pose.position.x = 1.15
        upper.pose.position.y = 0
        upper.pose.position.z = 0.95 + 0.40 #REVIEW: アクロバット軌道をしないように0.40足してみる(あまり効果なさそう)
        self.scene.add_box("upper", upper, (4, 4, 0.2))
        rospy.sleep(0.1)

    def _set_table(self):
        self.scene.remove_world_object()

        table = PoseStamped()
        table.header.frame_id = self.xarm7.get_planning_frame()
        table.pose.position.x = 1.15
        table.pose.position.y = 0
        table.pose.position.z = -0.65/2 -0.007 + 0.002# 箱の大きさと固定具の長さ0.007分下げる。ただし、机に激突しないようにすこしオフセット0.002を設ける　# REVIEW:この座標が台の高さにあっているか確認
        self.scene.add_box("table", table, (2, 2, 0.65))
        rospy.sleep(0.1)

    def _move_pose_goal(self):
        self.xarm7.set_goal_tolerance(0.001)
        #self.sawyer.set_goal_orientation_tolerance(0.01)
        self.xarm7.set_pose_target(self.pose_goal)
        plan1st = self.xarm7.go(wait=True)
        self.xarm7.stop()
        self.xarm7.clear_pose_targets()

    def _calc_value(self):
        height, width = self.debug_image.shape
        size = height * width
        count = 0
        # for i in range(height):
        #     for j in range(width):
        #         if(self.debug_image[i][j] > 0): count += 1
        # pixel_num =height*width
        # pixel_sum = np.num(self.debug_image)
        # white_pixel_num = pixel_sum/255
        # black_pixel_num = pixel_num - white_pixel_num
        # val = float(white_pixel_num)/pixel_num
        # print("reward is " + str(val))
        # if val > self.thresh_goal:
        #     print("goal achieved, robot is stopping...")
        #     self.clean_shutdown()

        rospy.sleep(2)

    def plan(self):

        self._setup_robot()

        self.scene.remove_world_object()

        self._set_table()

        self.set_neutral()
        self.drive_gripper(close=False)

        self._calc_value()

        # self.search_highest()



        self.scene.remove_world_object()

        self._set_table()

        for i in range(self.trial_num):
            print("planning", i)
            self.trial_cur = i
            self.get_point = True

            # wait for service
            self.service_done = False
            while not rospy.is_shutdown() and not self.service_done:
                self.is_in_area()

            if rospy.is_shutdown():
                break

            self._calc_value()

        #self.clean_shutdown()

def main():
    arg_fmt = argparse.RawDescriptionHelpFormatter
    parser = argparse.ArgumentParser(formatter_class=arg_fmt,
                                     description=main.__doc__)
    parser.parse_args(rospy.myargv()[1:])
    joint_state_topic = ['joint_states:=/xarm/joint_states']



    print("Initializing node... ")
    rospy.init_node("rsdk_joint_velocity_wobbler")
    moveit_commander.roscpp_initialize(joint_state_topic)
    #moveit_commander.roscpp_initialize(sys.argv)
    planner = Planner()
    rospy.on_shutdown(planner.clean_shutdown)
    planner.plan()


    print("Done.")

if __name__ == '__main__':
    main()

