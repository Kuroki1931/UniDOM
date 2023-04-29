#!/usr/bin/env python3
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

import argparse
import random
import numpy as np
import os
import torch
import sys
import copy
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from plb.envs import make
from plb.algorithms.logger import Logger

from plb.algorithms.discor.run_sac import train as train_sac
from plb.algorithms.ppo.run_ppo import train_ppo
from plb.algorithms.TD3.run_td3 import train_td3
from plb.optimizer.solver import solve_action
from plb.optimizer.solver_nn import solve_nn
from util.lang_goal import ACTION

os.environ['TI_USE_UNIFIED_MEMORY'] = '0'
os.environ['TI_DEVICE_MEMORY_FRACTION'] = '0.9'
os.environ['TI_DEVICE_MEMORY_GB'] = '4'
os.environ['TI_ENABLE_CUDA'] = '0'
os.environ['TI_ENABLE_OPENGL'] = '0'

RL_ALGOS = ['sac', 'td3', 'ppo']
DIFF_ALGOS = ['action', 'nn']


import datetime, os, cv2
import matplotlib.pyplot as plt
from PIL import Image


#座標変換用のパッケージのインポート
import tf2_ros
import tf2_py as tf2
import sensor_msgs.point_cloud2 as pc2
from tf2_geometry_msgs import PointStamped
from tf2_sensor_msgs.tf2_sensor_msgs import do_transform_cloud
from geometry_msgs.msg import Quaternion
#操作点表示用のパッケージのインポート
from visualization_msgs.msg import Marker

class TeleopCommand:
    START = 1
    STOP = 2
    REMOVE = 3
    ERROR = 4


class LoggerCommand:
    STOP = 0
    START = 1
    ERROR = 2

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_args():
    parser=argparse.ArgumentParser()
    parser.add_argument("--algo", type=str, default='action')
    parser.add_argument("--env_name", type=str, default="Move-v1")
    parser.add_argument("--path", type=str, default='./output')
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sdf_loss", type=float, default=500)
    parser.add_argument("--density_loss", type=float, default=500)
    parser.add_argument("--contact_loss", type=float, default=1)
    parser.add_argument("--soft_contact_loss", action='store_true')

    parser.add_argument("--num_steps", type=int, default=None)

    # differentiable physics parameters
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--softness", type=float, default=6666.)
    parser.add_argument("--optim", type=str, default='Adam', choices=['Adam', 'Momentum'])
    parser.add_argument("--create_grid_mass", action='store_true')

    args=parser.parse_args()

    return args

class Camera():
    def __init__(self, args):
        ## initial point clouds
        #記録する点群のトピック名
        record_pc_topic_name = "/filtered_pointcloud"
        self.pc = None
        self.pc_sub = rospy.Subscriber(record_pc_topic_name, PointCloud2, self.callback, queue_size=1)
        time.sleep(0.5)
        np.save('/root/real2sim/real2sim/points/initial_pcds.npy', self.pc) # initial position of object point clouds

        ## sim
        rospy.Subscriber(
            '/teleop/command_c1', Int8, self.command_c1_cb)
        rospy.Subscriber(
            '/teleop/command_c2', Int8, self.command_c2_cb)
        rospy.Subscriber(
            '/controller/vel_c1', Twist, self.vel_c1_cb)
        rospy.Subscriber(
            '/controller/vel_c2', Twist, self.vel_c2_cb)
        rospy.Subscriber(
            '/controller/trigger_c1', Float32, self.trigger_c1_cb)
        self.rgb_image = None
        self.running = False
        self.reset = False
        self.remove = False
        self.bridge = CvBridge()
        self.scale_x = 6
        self.scale_y = 6
        self.scale_z = 6

        ## setup
        set_random_seed(args.seed)
        self.env = make(args.env_name, nn=(args.algo=='nn'), sdf_loss=args.sdf_loss,
                                density_loss=args.density_loss, contact_loss=args.contact_loss,
                                soft_contact_loss=args.soft_contact_loss)
        self.env.seed(args.seed)
        self.env.reset()
        name = args.env_name.split('-')[0]
        if 'Chopsticks' in name:
            dummy_act = np.array([0]*7)
        else:
            dummy_act = np.array([0]*3)
        self.env.step(dummy_act)
        dummy_img = self.env.render(mode='rgb_array')
        im = plt.imshow(dummy_img)


        self.im = im
        self.action_list = []
        self.name = name

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
        move_group.set_max_velocity_scaling_factor(0.01) # 速度を最大に設定, これをしないと速度制限のせいでガタガタ動く
        move_group.set_num_planning_attempts(10)
        self.xarm_gripper.set_max_velocity_scaling_factor(1.0)

        #座標変換に必要なインスタンスの生成
        tf_buffer = tf2_ros.Buffer()
        tf_listener = tf2_ros.TransformListener(tf_buffer)

        #操作点をrvizに表示するようのPublisherを作成
        self.marker_pub = rospy.Publisher("controller_pos",Marker,queue_size=10)

        ## 初期位置に移動
        # real
        primitive_position = self.env.taichi_env.primitives[0].get_state(0)
        np.save('/root/real2sim/real2sim/points/initial_primitive.npy', primitive_position) # initial position of object point clouds
        print(primitive_position)
        #R_link_base座標系でのマニピュレーション位置の取得
        pos_local = PointStamped()
        pos_local.header.frame_id = "R_link_base"
        pos_local.header.stamp = rospy.Time()
        pos_local.point.x = primitive_position[2] - 0.1
        pos_local.point.y = primitive_position[0] - 0.5
        pos_local.point.z = primitive_position[1] - 0.14
        #R_link_base座標系からground座標系に変換
        try:
            pos_ground = tf_buffer.transform(pos_local,"ground",rospy.Duration(1))
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            rospy.logwarn("tf not found.")
        target_pose = move_group.get_current_pose(end_effector_link)
        target_pose.pose.position.x = pos_ground.point.x
        target_pose.pose.position.y = pos_ground.point.y
        target_pose.pose.position.z = pos_ground.point.z
        target_pose.pose.orientation = Quaternion(x=-1.0, y=0.0, z=0.0, w=0.0) #gripperは下向きで固定

        move_group.clear_pose_targets()
        move_group.set_pose_target(target_pose)
        move_group.go(wait=True)

        self.real_pcds_list = []
        self.sim_pcds_list = []

        while not rospy.is_shutdown():
            if self.running:
                ## sim
                pos_act = self.vel_c1
                pos_act = np.clip(pos_act,-1, 1)

                env_name = args.env_name.split('-')[0]
                order = ACTION[env_name]['order']
                direction = ACTION[env_name]['direction']
                pos_act = pos_act[order]
                pos_act = pos_act * direction

                if self.trigger_c1 > 0:
                    gripper_act = 1
                else:
                    gripper_act = 0
                act = np.append(pos_act, gripper_act)
                print(act)

                if 'Chopsticks' in self.name:
                    self.action_list.append(act)
                else:
                    act[-1] = 0
                    self.action_list.append(act)

                if 'Chopsticks' in self.name:
                    act = np.insert(act, 3, [0]*3)
                    print(f"chopsticks: {act}")
                else:
                    act = act[:3]

                self.env.step(act)
                self.render_env()
                print(len(self.action_list))

                ## real
                primitive_position = self.env.taichi_env.primitives[0].get_state(0)
                print(primitive_position)

                #R_link_base座標系でのマニピュレーション位置の取得
                pos_local = PointStamped()
                pos_local.header.frame_id = "R_link_base"
                pos_local.header.stamp = rospy.Time()
                pos_local.point.x = primitive_position[2]-0.1
                pos_local.point.y = primitive_position[0]-0.5
                pos_local.point.z = primitive_position[1]-0.14

                #R_link_base座標系からground座標系に変換
                try:
                    pos_ground = tf_buffer.transform(pos_local,"ground",rospy.Duration(1))
                except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
                    rospy.logwarn("tf not found.")

                target_pose = move_group.get_current_pose(end_effector_link)
                target_pose.pose.position.x = pos_ground.point.x
                target_pose.pose.position.y = pos_ground.point.y
                target_pose.pose.position.z = pos_ground.point.z
                target_pose.pose.orientation = Quaternion(x=-1.0, y=0.0, z=0.0, w=0.0) #gripperは下向きで固定

                #操作点をRvizに表示
                controller_marker = Marker()
                controller_marker.header.frame_id = "ground"
                controller_marker.header.stamp = rospy.Time.now()

                controller_marker.ns = ""
                controller_marker.id = 0
                controller_marker.type = 2 # 球体

                controller_marker.action = Marker.ADD

                controller_marker.pose = target_pose.pose

                controller_marker.color.r = 1.0
                controller_marker.color.g = 0.0
                controller_marker.color.b = 0.0
                controller_marker.color.a = 1.0

                controller_marker.scale.x = 0.01
                controller_marker.scale.y = 0.01
                controller_marker.scale.z = 0.01

                controller_marker.lifetime = rospy.Duration()
                self.marker_pub.publish(controller_marker)

                #グリッパーの開閉
                joint_goal = self.xarm_gripper.get_current_joint_values()
                # gripper open close level :0 is fully open, 0.85 is closed
                # ただし、0.85よりも微妙に大きくしてもより閉じるが、大きくしすぎるとjoint limitのせいかエラーが出る
                if self.trigger_c1:
                    joint_goal[0] = 0.85
                else:
                    joint_goal[0] = 0.0

                self.xarm_gripper.go(joint_goal, wait=False)

                print("target_pose.pose.position",target_pose.pose.position)        
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
                self.sim_pcds_list.append(self.env.taichi_env.simulator.get_x(0))
                print(len(self.real_pcds_list), len(self.sim_pcds_list))

            if self.remove:
                self.action_list = []
                self.real_pcds_list = []
                self.sim_pcds_list = []
                self.env.reset()
                self.running = False
                self.reset = False
                self.remove = False
                self.render_env()
                print("### Teleop Stop ###")

            if self.reset:
                np.save(f'/root/real2sim/real2sim/points/action.npy', np.array(self.action_list))
                np.save(f'/root/real2sim/real2sim/points/real_pcds.npy', np.array(self.real_pcds_list))
                np.save(f'/root/real2sim/real2sim/points/sim_pcds.npy', np.array(self.sim_pcds_list))
                self.action_list = []
                self.real_pcds_list = []
                self.sim_pcds_list = []
                self.env.reset()
                self.running = False
                self.reset = False
                self.remove = False
                self.render_env()
                print("### Teleop Stop ###")

    def callback(self, pointcloud):
        #PointCloud2型をnumpy配列に変換
        pc = list(pc2.read_points(pointcloud, skip_nans=True, field_names=("x", "y", "z")))
        pc = np.array(pc)
        self.pc = pc[:, [1, 2, 0]] + np.array([0.5, 0.14, 0.1])

    def render_env(self):
        img = self.env.render(mode='rgb_array')
        self.im.set_data(img)
        plt.pause(0.005)

    def command_c1_cb(self, msg):
        if (msg.data == TeleopCommand.START and self.running == False) or (msg.data == TeleopCommand.STOP and self.running == False):
            self.running = True
            print("### Teleop Start ###")
        elif (msg.data == TeleopCommand.STOP and self.running == True) or (msg.data == TeleopCommand.START and self.running == True):
            self.running = False
            print("### Teleop Stop ###")
        elif msg.data == TeleopCommand.REMOVE:
            self.remove = True
            print("### Remove ###")

    def command_c2_cb(self, msg):
        if msg.data == TeleopCommand.START:
            self.reset = True
            print("### Reset ###")
        elif msg.data == TeleopCommand.STOP:
            self.reset = True
            print("### Reset ###")
        elif msg.data == TeleopCommand.REMOVE:
            self.remove = True
            print("### Remove ###")

    def vel_c1_cb(self, msg):
        vel_x = - self.scale_x * msg.linear.x
        vel_y = self.scale_y * msg.linear.y
        vel_z = self.scale_z * msg.linear.z
        self.vel_c1 = np.array([vel_x, vel_y, vel_z])

    def vel_c2_cb(self, msg):
        vel_x = - self.scale_x * msg.linear.x
        vel_y = self.scale_y * msg.linear.y
        vel_z = self.scale_z * msg.linear.z
        self.vel_c2 = np.array([vel_x, vel_y, vel_z])

    def trigger_c1_cb(self, msg):
        self.trigger_c1 = np.array(msg.data)


if __name__ == '__main__':
    rospy.init_node('special_node', log_level=rospy.DEBUG)
    args = get_args()
    cam = Camera(args)

    while not rospy.is_shutdown():
        rospy.sleep(1)