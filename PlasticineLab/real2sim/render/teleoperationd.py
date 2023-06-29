#!/usr/bin/env python3
import os
import cv2
import numpy as np
import rospy
import pickle
import time
from cv_bridge import CvBridge

import moveit_commander
from trajectory_msgs.msg import JointTrajectory
from std_msgs.msg import Int8, Float32
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from sensor_msgs.msg import PointCloud2

import sys
import copy

#座標変換用のパッケージのインポート
import tf2_ros
import tf2_py as tf2
import sensor_msgs.point_cloud2 as pc2
from tf2_geometry_msgs import PointStamped
from tf2_sensor_msgs.tf2_sensor_msgs import do_transform_cloud
from geometry_msgs.msg import Quaternion
#操作点表示用のパッケージのインポート
from visualization_msgs.msg import Marker


class Robot():
    def __init__(self):
        ## initial point clouds
        #記録する点群のトピック名
        record_pc_topic_name = "/filtered_pointcloud"
        self.pc = None
        self.pc_sub = rospy.Subscriber(record_pc_topic_name, PointCloud2, self.callback, queue_size=1)
        time.sleep(1)

        ## xarm
        moveit_commander.roscpp_initialize(sys.argv)
        robot = moveit_commander.RobotCommander()
        # ロボットの情報を表示
        print("group_names:",robot.get_group_names(),"\n")
        scene = moveit_commander.PlanningSceneInterface()
        # move home position
        group_name = "R_xarm7"
        end_effector_link = "R_link7"
        move_group = moveit_commander.MoveGroupCommander(group_name)
        pub_jt = rospy.Publisher('/R_xarm7_traj_controller/command', JointTrajectory, queue_size=1)
        # ロボットを動かすためのcommanderの準備
        move_group = moveit_commander.MoveGroupCommander(group_name)
        self.xarm_gripper = moveit_commander.MoveGroupCommander('R_xarm_gripper')
        # ロボットのplanningの設定
        move_group.set_max_velocity_scaling_factor(0.001) # 速度を最大に設定, これをしないと速度制限のせいでガタガタ動く
        move_group.set_num_planning_attempts(10)
        self.xarm_gripper.set_max_velocity_scaling_factor(1.0)
        #座標変換に必要なインスタンスの生成
        tf_buffer = tf2_ros.Buffer()
        tf_listener = tf2_ros.TransformListener(tf_buffer)
        initial_arm_position = move_group.get_current_pose(end_effector_link).pose.position
        initial_orientation = move_group.get_current_pose(end_effector_link).pose.orientation
        print('initial_arm_position', initial_arm_position)

        steps = 0
        z_axis = 0
        self.real_pcds_list = []
        while not steps == 150:
            print('--------------------------------------------')
            print(steps)
            print('position', initial_arm_position.x, initial_arm_position.y, z_axis)
            steps += 1
            z_axis += 0.0015

            target_pose = move_group.get_current_pose(end_effector_link)
            target_pose.pose.position.x = initial_arm_position.x
            target_pose.pose.position.y = initial_arm_position.y
            target_pose.pose.position.z = z_axis - 0.125
            target_pose.pose.orientation = initial_orientation
   
            pose_goal = copy.deepcopy(target_pose)
            plan, _fraction = move_group.compute_cartesian_path([pose_goal.pose], 0.01, 0)
            # TODO: check duration is correct (previously rospy.Duration(1))
            plan_time = plan.joint_trajectory.points[-1].time_from_start.to_sec()
            pref_time = 1.0 / 10 * 2.0
            max_speedup = 10.0 # 10.0
            if plan_time > pref_time:
                new_plan_time = max(pref_time, plan_time / max_speedup)
                scale = new_plan_time / plan_time
                print("scaling:", scale)
                for pt in plan.joint_trajectory.points:
                    pt.time_from_start = rospy.Duration(pt.time_from_start.to_sec() * scale)
                # this speeds up the robot significantly
                plan.joint_trajectory.points = [plan.joint_trajectory.points[-1]]
                stamp = rospy.Time.now()
                pose_goal.header.stamp = stamp
                # self.pub_jt.publish(empty)
                pub_jt.publish(plan.joint_trajectory)
            
            self.real_pcds_list.append(self.pc)
            time.sleep(0.2)

    def callback(self, pointcloud):
        #PointCloud2型をnumpy配列に変換
        pc = list(pc2.read_points(pointcloud, skip_nans=True, field_names=("x", "y", "z")))
        pc = np.array(pc)
        self.pc = pc

if __name__ == '__main__':
    rospy.init_node('special_node', log_level=rospy.DEBUG)
    robot = Robot()
    fill_pcds_list = []
    bf_pcd = None
    for pcd in robot.real_pcds_list:
        if pcd.shape[0] < 1000:
            fill_pcds_list.append(pcd)
        else:
            if bf_pcd is not None:
                fill_pcds_list.append(bf_pcd)
        bf_pcd = pcd

    pcds_array = np.array(fill_pcds_list)
    rope_type = 'navy'
    output_path = f'/root/real2sim/real2sim/real_points/{rope_type}'
    os.makedirs(output_path, exist_ok=True)
    np.save(f'{output_path}/real_pcds.npy', pcds_array)
    while not rospy.is_shutdown():
        rospy.sleep(1)