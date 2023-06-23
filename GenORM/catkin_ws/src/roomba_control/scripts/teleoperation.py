#!/usr/bin/env python3

import sys
# try:
#     sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
# except:
#     pass

import pickle
import time
import rospy
from scipy.spatial.transform import Rotation as R
import numpy as np
import rospkg
import math
import json
from std_msgs.msg import Float32, Int8, Bool, Empty
from geometry_msgs.msg import PoseStamped, Twist
import os
import triad_openvr
# from std_msgs.msg import Float32MultiArray

from functools import partial

from std_msgs.msg import MultiArrayDimension, Float32MultiArray
import tf


def _numpy2multiarray(multiarray_type, np_array):
    """Convert numpy.ndarray to multiarray"""
    multiarray = multiarray_type()
    multiarray.layout.dim = [MultiArrayDimension(
        "dim%d" % i, np_array.shape[i], np_array.shape[i] * np_array.dtype.itemsize) for i in range(np_array.ndim)]
    multiarray.data = np_array.reshape(1, -1)[0].tolist()
    return multiarray


def _multiarray2numpy(pytype, dtype, multiarray):
    """Convert multiarray to numpy.ndarray"""
    dims = map(lambda x: x.size, multiarray.layout.dim)
    return np.array(multiarray.data, dtype=pytype).reshape(dims).astype(dtype)


numpy2f32multi = partial(_numpy2multiarray, Float32MultiArray)
f32multi2numpy = partial(_multiarray2numpy, float, np.float32)


class TeleopCommand:
    START = 1
    STOP = 2
    REMOVE = 3
    ERROR = 4


class LoggerCommand:
    STOP = 0
    START = 1
    ERROR = 2


def cpose2mat(cpose):
    cpose = np.array([
        [cpose[0][0], cpose[0][1], cpose[0][2], cpose[0][3]],
        [cpose[1][0], cpose[1][1], cpose[1][2], cpose[1][3]],
        [cpose[2][0], cpose[2][1], cpose[2][2], cpose[2][3]],
        [0, 0, 0, 1],
    ])
    return cpose


def cpose2matlist(cpose):
    cpose = [
        [cpose[0][0], cpose[0][1], cpose[0][2], cpose[0][3]],
        [cpose[1][0], cpose[1][1], cpose[1][2], cpose[1][3]],
        [cpose[2][0], cpose[2][1], cpose[2][2], cpose[2][3]],
        [0.0, 0.0, 0.0, 1.0],
    ]
    return cpose


class Teleop():
    def __init__(self):
        # pub sub
        base_pub_1 = rospy.Publisher(
            '/controller/base_c1', Twist, queue_size=1)
        pose_pub_1 = rospy.Publisher(
            '/controller/pose_c1', PoseStamped, queue_size=1)
        base_pub_2 = rospy.Publisher(
            '/controller/base_c2', Twist, queue_size=1)
        pose_pub_2 = rospy.Publisher(
            '/controller/pose_c2', PoseStamped, queue_size=1)
        hmdpose_pub_ = rospy.Publisher(
            '/controller/pose_hmd', Float32MultiArray, queue_size=1)
        trigger_pub_1 = rospy.Publisher(
            '/controller/trigger_c1', Float32, queue_size=1)
        trigger_pub_2 = rospy.Publisher(
            '/controller/trigger_c2', Float32, queue_size=1)
        self.rosbag_pub_ = rospy.Publisher('/logging', Int8, queue_size=1)
        gohome_pub_ = rospy.Publisher('/teleop/gohome', Empty, queue_size=1)
        command_pub_1 = rospy.Publisher(
            '/teleop/command_c1', Int8, queue_size=1)
        command_pub_2 = rospy.Publisher(
            '/teleop/command_c2', Int8, queue_size=1)
        
        vel_pub_1 = rospy.Publisher(
            '/controller/vel_c1', Twist, queue_size=1)
        vel_pub_2 = rospy.Publisher(
            '/controller/vel_c2', Twist, queue_size=1)
        # command_sub_1 = rospy.Subscriber(
        #     '/teleop/command_c1', Int8, self.command_c1_cb)
        # command_sub_2 = rospy.Subscriber(
        #     '/teleop/command_c2', Int8, self.command_c2_cb)

        # mat_pub_ = rospy.Publisher('/controller/mat_c1', Float32MultiArray, queue_size=10)

        # calibration file path
        rospack = rospkg.RosPack()
        # package_path = rospack.get_path('roomba_controller')
        # self.calibrationFileARM = package_path + '/config/calib_arm.p'
        # self.calibrationFileHMD = package_path + '/config/calib_head.p'

        self.v = triad_openvr.triad_openvr()
        self.v.print_discovered_objects()
        # self.load_calibration()

        # loop_rate = rospy.Rate(1) # 10hz
        rate = rospy.Rate(10)  # 10hz
        t = 0.0
        self.running_c1 = False
        self.running_c2 = False
        trackpad_pressed = False
        menu_pressed_c1 = False
        grip_pressed_c1 = False
        menu_pressed_c2 = False
        grip_pressed_c2 = False
        self._logging = False
        # print(self.v.devices["hmd_1"].vr)
        # print(type(self.v.devices["hmd_1"].vr))

        while not rospy.is_shutdown():
            if menu_pressed_c1 == False and self.v.devices["controller_1"].get_controller_inputs()['menu_button']:
                menu_pressed_c1 = True
                self.running_c1 = not bool(self.running_c1)
                msg = Int8()
                if self.running_c1:
                    msg.data = TeleopCommand.START
                else:
                    msg.data = TeleopCommand.STOP
                command_pub_1.publish(msg)
            elif menu_pressed_c1 == True and self.v.devices["controller_1"].get_controller_inputs()['menu_button'] == False:
                menu_pressed_c1 = False

            if menu_pressed_c2 == False and self.v.devices["controller_2"].get_controller_inputs()['menu_button']:
                menu_pressed_c2 = True
                self.running_c2 = not bool(self.running_c2)
                msg = Int8()
                if self.running_c2:
                    msg.data = TeleopCommand.START
                else:
                    msg.data = TeleopCommand.STOP
                command_pub_2.publish(msg)
            elif menu_pressed_c2 == True and self.v.devices["controller_2"].get_controller_inputs()['menu_button'] == False:
                menu_pressed_c2 = False

            if grip_pressed_c1 == False and self.v.devices["controller_1"].get_controller_inputs()['grip_button']:
                grip_pressed_c1 = True
                self.running_c1 = not bool(self.running_c1)
                msg = Int8()
                if self.running_c1:
                    msg.data = TeleopCommand.REMOVE
                else:
                    msg.data = TeleopCommand.REMOVE
                command_pub_1.publish(msg)
            elif grip_pressed_c1 == True and self.v.devices["controller_1"].get_controller_inputs()['grip_button'] == False:
                grip_pressed_c1 = False

            if grip_pressed_c2 == False and self.v.devices["controller_2"].get_controller_inputs()['grip_button']:
                grip_pressed_c2 = True
                self.running_c2 = not bool(self.running_c2)
                msg = Int8()
                if self.running_c2:
                    msg.data = TeleopCommand.REMOVE
                else:
                    msg.data = TeleopCommand.REMOVE
                command_pub_2.publish(msg)
            elif grip_pressed_c2 == True and self.v.devices["controller_2"].get_controller_inputs()['grip_button'] == False:
                grip_pressed_c2 = False

            if self.v.devices["controller_2"].get_pose_matrix() is None:
                print('none controller_2')
                continue

            cpose = cpose2mat(self.v.devices["controller_2"].get_pose_matrix())
            # print("cpose",cpose[0][1],cpose[0][2],cpose[0][3])

            # mat_msg=numpy2f32multi(cpose)
            # mat_pub_.publish(mat_msg)

            # print(f'b cpose: {cpose}')
            # cpose = self.ctrans.dot(cpose)
            # print(f'a cpose: {cpose}')
            # print(f'ctrans: {self.ctrans}')

            pos = cpose[:3, -1]  # + np.array([0, 0, -0.3])
            r = cpose[:3, :3].dot(R.from_euler(
                'xyz', [0.5 * np.pi, 0.5 * np.pi, 0]).as_matrix())
            rot = R.from_matrix(r).as_quat()

            pose = PoseStamped()
            pose.header.stamp = rospy.Time.now()
            # pose.header.frame_id = "world"
            pose.header.frame_id = "base_link"

            pose.pose.position.x = pos[0]
            pose.pose.position.y = pos[1]
            pose.pose.position.z = pos[2]
            # pose.pose.orientation.x = 0
            # pose.pose.orientation.y = 0
            # pose.pose.orientation.z = 0
            # pose.pose.orientation.w = 1
            pose.pose.orientation.x = rot[0]
            pose.pose.orientation.y = rot[1]
            pose.pose.orientation.z = rot[2]
            pose.pose.orientation.w = rot[3]

            # pose.pose.position.x = 0.4
            # pose.pose.position.y = 0.0
            # pose.pose.position.z = 0.1 + 0.1 * math.sin(2 * math.pi * t)
            # pose.pose.orientation.w = 1.0
            # print(pos)

            trigger_c1 = Float32()
            trigger_c1.data = self.v.devices["controller_1"].get_controller_inputs()[
                'trigger']

            trigger_c2 = Float32()
            trigger_c2.data = self.v.devices["controller_2"].get_controller_inputs()[
                'trigger']

            pose_pub_2.publish(pose)
            trigger_pub_1.publish(trigger_c1)
            trigger_pub_2.publish(trigger_c2)

            # For Controller #############################################################

            # For HMD ############################################################

            if self.v.devices["hmd_1"].get_pose_matrix() is None:
                print('hmd none')
                continue
            hmdpose = cpose2mat(self.v.devices["hmd_1"].get_pose_matrix())
            # hmdpose = self.hmdtrans.dot(hmdpose)

            hmd_rotmsg = Float32MultiArray()
# euler
            rot = tf.transformations.euler_from_matrix(hmdpose[:3, :3])
            rot = R.from_matrix(hmdpose[:3, :3]).as_euler('xyz')
            hmd_rotmsg.data = [rot[0], rot[1], rot[2]]

# quat
            # quat = tf.transformations.quaternion_from_matrix(hmdpose)
            # quat = R.from_matrix(hmdpose[:3, :3]).as_quat()
            # hmd_rotmsg.data = [quat[0], quat[1], quat[2], quat[3]]

            # print("hmd_rotmsg.data",hmd_rotmsg.data)
            hmdpose_pub_.publish(hmd_rotmsg)
            # For HMD ############################################################

            # For ALL ############################################################
            t += 0.1
            rate.sleep()
            # For ALL ############################################################

            # For BASE ############################################################
            base_msg = Twist()
            base_msg.linear.x = 0.0
            base_msg.angular.z = 0.0
            base_tra = self.v.devices["controller_2"].get_controller_inputs()[
                'trackpad_y']
            base_ang = self.v.devices["controller_2"].get_controller_inputs()[
                'trackpad_x']
            TRA_TH = 0.4
            ANG_TH = 0.4
            ANG_VEL = 0.9
            TRA_VEL = 0.2
            if base_tra > TRA_TH:
                base_msg.linear.x = TRA_VEL
            elif base_tra < -TRA_TH:
                base_msg.linear.x = -TRA_VEL
            else:
                base_msg.linear.x = 0.0

            if base_ang > ANG_TH:
                base_msg.angular.z = -ANG_VEL
                if base_tra < -TRA_TH:
                    base_msg.angular.z = ANG_VEL
            elif base_ang < -ANG_TH:
                base_msg.angular.z = ANG_VEL
                if base_tra < -TRA_TH:
                    base_msg.angular.z = -ANG_VEL
            else:
                base_msg.angular.z = 0.0
            trackpad_pressed = True
            base_pub_2.publish(base_msg)
            # For BASE ############################################################

            # For Velocity Publisher
            vel_tra = self.v.devices["controller_1"].get_velocity()
            # print(vel_tra[0])
            vel_msg = Twist()
            vel_msg.linear.x = -vel_tra[2]
            vel_msg.linear.y = -vel_tra[0]
            vel_msg.linear.z = vel_tra[1]
            vel_msg.angular.z = 0.0

            vel_pub_1.publish(vel_msg)

            vel_tra = self.v.devices["controller_2"].get_velocity()
            # print(vel_tra[0])
            vel_msg = Twist()
            vel_msg.linear.x = -vel_tra[2]
            vel_msg.linear.y = -vel_tra[0]
            vel_msg.linear.z = vel_tra[1]
            vel_msg.angular.z = 0.0

            vel_pub_2.publish(vel_msg)

    def exec_calibration(self):
        print('start calibration ...')
        rot_x = np.pi*0.5
        rot_y = np.pi
        rot_z = -np.pi*0.5

        rot_x_h = -np.pi*0.5
        rot_y_h = -np.pi*0.5


        self.ctrans = np.eye(4)
        self.hmdtrans = np.eye(4)

        # print('self.v.devices["controller_2"].get_pose_matrix())',
        #       self.v.devices["controller_2"].get_pose_matrix())
        # print('self.v.devices["hmd_1"].get_pose_matrix())',
        #       self.v.devices["hmd_1"].get_pose_matrix())

        if (self.v.devices["controller_2"].get_pose_matrix()) is None:
            # if (self.v.devices["controller_2"].get_pose_matrix()) or (self.v.devices["hmd_1"].get_pose_matrix())  is None:
            print('fail calibration (cannot find controller)')
            return
        if (self.v.devices["hmd_1"].get_pose_matrix()) is None:
            print('fail calibration (cannot find controller)')
            return

        cpose = cpose2mat(self.v.devices["controller_2"].get_pose_matrix())

        hmdpose = cpose2mat(self.v.devices["hmd_1"].get_pose_matrix())

        mat_x = np.array([
            [1,              0, 0, 0.0],
            [0, np.cos(rot_x), -np.sin(rot_x), 0],
            [0, np.sin(rot_x),  np.cos(rot_x), 0],
            [0,              0, 0,    1]
        ])
        mat_y = np.array([
            [np.cos(rot_y),              0, np.sin(rot_y),    0],
            [0,              1, 0,    0],
            [-np.sin(rot_y),              0, np.cos(rot_y),    0],
            [0,              0, 0,    1]
        ])
        mat_z = np.array([
            [np.cos(rot_z), -np.sin(rot_z), 0, 0.0],
            [np.sin(rot_z),  np.cos(rot_z), 0,    0],
            [0,              0, 1, 0.0],
            [0,              0, 0,    1]
        ])
        target = np.array([
            [0, 0, -1, 0],
            [-1, 0,  0, 0],
            [0, 1,  0, 0],
            [0, 0,  0, 1]
        ])


        mat_x_h = np.array([
            [1,              0, 0, 0.0],
            [0, np.cos(rot_x_h), -np.sin(rot_x_h), 0],
            [0, np.sin(rot_x_h),  np.cos(rot_x_h), 0],
            [0,              0, 0,    1]
        ])

        mat_y_h = np.array([
            [np.cos(rot_y_h),              0, np.sin(rot_y_h),    0],
            [0,              1, 0,    0],
            [-np.sin(rot_y_h),              0, np.cos(rot_y_h),    0],
            [0,              0, 0,    1]
        ])

        rot_rad = np.pi*0.0
        # Tmat = np.array([
        #     [np.cos(rot_rad), -np.sin(rot_rad), 0, 0.286],
        #     [np.sin(rot_rad),  np.cos(rot_rad), 0,    0.066],
        #     [0,              0, 1, 0.672],
        #     [0,              0, 0,    1]
        # ])

        Tmat = np.array([
            [np.cos(rot_rad), -np.sin(rot_rad), 0, 0.0],
            [np.sin(rot_rad),  np.cos(rot_rad), 0,    0.0],
            [0,              0, 1, 0.0],
            [0,              0, 0,    1]
        ])

        # rot_rad=np.pi*0.0
        # Tmat = np.array([
        #     [np.cos(rot_rad), -np.sin(rot_rad), 0, 0.672],
        #     [np.sin(rot_rad),  np.cos(rot_rad), 0,    -0.066],
        #     [            0,              0, 1, -0.286],
        #     [            0,              0, 0,    1]
        # ])

        # Tmat = np.array([
        #     [np.cos(rot_rad), -np.sin(rot_rad), 0, 0.0],
        #     [np.sin(rot_rad),  np.cos(rot_rad), 0,    0.0],
        #     [            0,              0, 1, 0.0],
        #     [            0,              0, 0,    1]
        # ])

        # print('cpose',cpose)
        # print('hmdpose',hmdpose)

        # C1 from map
        # self.ctrans = Tmat.dot(mat_y.dot(mat_z.dot(np.linalg.inv(cpose))))

        # C1 from hand_palm_link
        self.ctrans = mat_x_h.dot(mat_y_h.dot(np.linalg.inv(cpose)))

        # self.ctrans = np.linalg.inv(cpose)
        # self.ctrans = mat_z.dot(np.linalg.inv(cpose))
        # HMD
        self.hmdtrans = np.linalg.inv(hmdpose)
        # self.hmdtrans = hmdpose

        rospack = rospkg.RosPack()
        pickle.dump(self.ctrans, open(self.calibrationFileARM, 'wb'))
        pickle.dump(self.hmdtrans, open(self.calibrationFileHMD, 'wb'))

        print('completed calibration')

        # self.ctrans = Tmat.dot(np.linalg.inv(cpose))
        # self.ctrans = target.dot(np.linalg.inv(cpose))
        # self.ctrans = mat_y.dot(mat_z.dot(np.linalg.inv(cpose)))
        # self.ctrans = mat_y.dot(mat_x.dot(np.linalg.inv(cpose)))
        # self.ctrans = mat_x.dot(np.linalg.inv(cpose))
        # self.ctrans = np.linalg.inv(cpose)

        # self.ctrans = Tmat.dot(target.dot(np.linalg.inv(cpose)))
        # self.ctrans = np.linalg.inv(cpose)

        # self.ctrans[2] = [0, 1, 0, 0]

    def load_calibration(self):
        try:
            print('load calibration file ...')
            self.ctrans = pickle.load(open(self.calibrationFileARM, 'rb'))
            self.hmdtrans = pickle.load(open(self.calibrationFileHMD, 'rb'))

            print('calibration file loaded')
        except:
            print('!!!!!!!!!could not find calibration file!!!!!!!!!!!')
            exit()


def main():
    rospy.init_node('teleop_ros')
    Teleop()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
