#!/usr/bin/python3
import sys
import moveit_commander
import rospy
import time
import copy
from std_msgs.msg import Bool, Empty, Int8
from geometry_msgs.msg import PoseStamped, Pose
from moveit_msgs.msg import RobotTrajectory
from trajectory_msgs.msg import JointTrajectory
from scipy.spatial.transform import Rotation as R
from xarm_msgs.srv import SetInt16, ClearErr, Move
import numpy as np
import math


class XArmTeleopInterface():
    def __init__(self, control_dof=3, update_rate=10):
        self.enable = True
        self.group_name = 'xarm7'
        self.control_dof = control_dof
        self.update_rate = update_rate
        self.pose = None

        rospy.set_param('/xarm/wait_for_finish', True)

        moveit_commander.roscpp_initialize(sys.argv)
        rospy.init_node('xarm_teleop', anonymous=True)

        self.group = moveit_commander.MoveGroupCommander(self.group_name)
        rate = rospy.Rate(self.update_rate)

        _cp = rospy.Subscriber('/mss/m2s/pose_pub', PoseStamped, self.pose_cb)
        _tm = rospy.Subscriber('/teleop/movehome', Empty, self.movehome_cb)
        _te = rospy.Subscriber('/teleop/enable', Bool, self.enable_cb)
        self.pub_jt = rospy.Publisher('/xarm/xarm7_traj_controller/command', JointTrajectory, queue_size=1)
        self.pub_pose = rospy.Publisher('/teleop/pose_goal', PoseStamped, queue_size=1)

        counter = 0

        while not rospy.is_shutdown():
            # TODO: discard message if not fresh?
            if self.pose is not None:
                pose_goal = copy.deepcopy(self.pose)

                if self.enable:
                    counter += 1
                    # print(counter, self.enable)

                    # change rotation according to setting
                    if self.control_dof == 3:
                        # always point downward
                        pose_goal.pose.orientation.x = -1.0
                        pose_goal.pose.orientation.y = 0.0
                        pose_goal.pose.orientation.z = 0.0
                        pose_goal.pose.orientation.w = 0.0
                    elif self.control_dof == 2:
                        print("control dof is 2")
                        pose_goal.pose.position.z = 0.2
                        pose_goal.pose.orientation.x = -1.0
                        pose_goal.pose.orientation.y = 0.0
                        pose_goal.pose.orientation.z = 0.0
                        pose_goal.pose.orientation.w = 0.0
                    elif self.control_dof == 4:
                        o = self.pose.pose.orientation
                        o_euler = (R.from_quat([o.x, o.y, o.z, o.w])).as_euler('zyx')
                        o_euler[1] = 0
                        o_euler[2] = np.pi
                        o_quat = (R.from_euler('zyx', o_euler)).as_quat()
                        pose_goal.pose.orientation.x = o_quat[0]
                        pose_goal.pose.orientation.y = o_quat[1]
                        pose_goal.pose.orientation.z = o_quat[2]
                        pose_goal.pose.orientation.w = o_quat[3]
                    else:
                        # TODO: implement 4DoF & 6DoF control
                        # TODO: check if rotation is correct
                        o = self.pose.pose.orientation
                        print("o", o)
                        o_quat = (R.from_quat([o.x, o.y, o.z, o.w]) * R.from_euler('xyz', [0, 0., 0])).as_quat()
                        print("o_quat", o_quat)
                        pose_goal.pose.orientation.x = o_quat[0]
                        pose_goal.pose.orientation.y = o_quat[1]
                        pose_goal.pose.orientation.z = o_quat[2]
                        pose_goal.pose.orientation.w = o_quat[3]
                        # raise NotImplementedError

                    plan, _fraction = self.group.compute_cartesian_path([pose_goal.pose], 0.01, 0)
                    
                    # 
                    # TODO: check duration is correct (previously rospy.Duration(1))
                    plan_time = plan.joint_trajectory.points[-1].time_from_start.to_sec()
                    pref_time = 1.0 / self.update_rate * 2.0
                    max_speedup = 100.0 # 10.0

                    if plan_time > pref_time:
                        new_plan_time = max(pref_time, plan_time / max_speedup)
                        scale = new_plan_time / plan_time
                        print("scaling:", scale)

                        for pt in plan.joint_trajectory.points:
                            pt.time_from_start = rospy.Duration(pt.time_from_start.to_sec() * scale)

                        # this speeds up the robot significantly
                        plan.joint_trajectory.points = [plan.joint_trajectory.points[-1]]

                    # plan.joint_trajectory.points[-1].time_from_start = rospy.Duration()
                    # plan.joint_trajectory.points[-1].time_from_start = rospy.Duration(1. / self.update_rate)

                    stamp = rospy.Time.now()
                    pose_goal.header.stamp = stamp
                    # plan.joint_trajectory.header.stamp = stamp

                    # empty = copy.deepcopy(plan.joint_trajectory)
                    # empty.points = []

                    # self.pub_jt.publish(empty)
                    self.pub_jt.publish(plan.joint_trajectory)
                    self.pub_pose.publish(pose_goal) # [needed] y of dataset
                    # print('EE pose: ', pose_goal.pose)

            rate.sleep()

    def pose_cb(self, data):
        self.pose = data

    def enable_cb(self, data):
        if data == True:
            jt = JointTrajectory()
            jt.header.stamp = rospy.Time.now()
            jt.points = []
            self.pub_jt.publish(jt)
            self.pose = self.group.getPose()
        self.enable = data
        # print('stop interface')

    def movehome_cb(self, data):
        self.enable = False

        # pose_home = PoseStamped()
        # pose_home.pose.position.x = 0.47
        # pose_home.pose.position.y = 0.0
        # pose_home.pose.position.z = 0.52
        # pose_home.pose.orientation.x = -1.0
        # pose_home.pose.orientation.y = 0.0
        # pose_home.pose.orientation.z = 0.0
        # pose_home.pose.orientation.w = 0.0
        # self.pose = pose_home

        
        pose_home = PoseStamped()

        if self.control_dof == 2:
            pose_home.pose.position.x = 0.3
            pose_home.pose.position.y = 0.0
            pose_home.pose.position.z = 0.20
            pose_home.pose.orientation.x = -1.0
            pose_home.pose.orientation.y = 0.0
            pose_home.pose.orientation.z = 0.0
            pose_home.pose.orientation.w = 0.0
        elif self.control_dof == 3:
            pose_home.pose.position.x = 0.3
            pose_home.pose.position.y = 0.0
            pose_home.pose.position.z = 0.30 #height
            pose_home.pose.orientation.x = -1.0
            pose_home.pose.orientation.y = 0.0
            pose_home.pose.orientation.z = 0.0
            pose_home.pose.orientation.w = 0.0

        else:
            pose_home.pose.position.x = 0.47
            pose_home.pose.position.y = 0.0
            pose_home.pose.position.z = 0.52
            pose_home.pose.orientation.x = -1.0
            pose_home.pose.orientation.y = 0.0
            pose_home.pose.orientation.z = 0.0
            pose_home.pose.orientation.w = 0.0

        self.pose = pose_home

        clear_err = rospy.ServiceProxy('/xarm/clear_err', ClearErr)
        set_mode = rospy.ServiceProxy('/xarm/set_mode', SetInt16)
        set_state = rospy.ServiceProxy('/xarm/set_state', SetInt16)
        move_joint = rospy.ServiceProxy('/xarm/move_joint', Move)

        try:
            clear_err()
            set_mode(0)
            set_state(0)
            print("control_dof", self.control_dof)

            # pose = [0,0,0,1.57,0,1.57,0]
            if self.control_dof==2:
                pose = [0, math.pi*(-6.8)/180, 0, math.pi*(17.6)/180, 0, math.pi* (24.5)/180, 0 ]
            elif self.control_dof==3:
                # pose = [0, math.pi*(-6.8)/180, 0, math.pi*(17.6)/180, 0, math.pi* (24.5)/180, 0 ]
                pose = [0, math.pi*(-28.2)/180, 0, math.pi*(23.2)/180, 0, math.pi* (51.4)/180, 0 ]
            else:
                pose = [0,0,0,1.57,0,1.57,0]
            mvvelo = 0.5#1.0
            mvacc = 0.25#0.5
            mvtime = 0.8
            mvradii = 0.7
            res = move_joint(pose, mvvelo, mvacc, mvtime, mvradii)
            print(res)

            clear_err()
            set_mode(1)
            set_state(0)
        except rospy.ServiceException as e:
            print("service call failed: %s" % e)
        
        # plan, _fraction = self.group.compute_cartesian_path([pose_home.pose], 0.01, 0)
        # robot = moveit_commander.RobotCommander()
        # plan = self.group.retime_trajectory(
        #     robot.get_current_state(), 
        #     plan, 
        #     velocity_scaling_factor=0.2,                      
        #     acceleration_scaling_factor=0.2,
        #     algorithm="time_optimal_trajectory_generation")
        # self.group.execute(plan)

        jt = JointTrajectory()
        jt.header.stamp = rospy.Time.now()
        jt.points = []
        self.pub_jt.publish(jt)
        
        time.sleep(0.5)

        self.enable = True


if __name__ == '__main__':
    control_dof = rospy.get_param("/teleop/control_dof")
    print("control_dof", control_dof)
    try:
        if control_dof == 6:
            XArmTeleopInterface(control_dof=6)
        elif control_dof == 4:
            XArmTeleopInterface(control_dof=4)
        elif control_dof == 2:
            print("control dof is 2")
            XArmTeleopInterface(control_dof=2)
        else:  # control_dof == 3:
            XArmTeleopInterface()
    except rospy.ROSInterruptException:
        pass
