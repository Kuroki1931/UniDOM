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
from xarm_msgs.msg import RobotMsg
import numpy as np
import tf

class TeleopCommand:
    CARIBRATION = 1
    START = 2
    STOP = 3
    ERROR = 4

class MSSystemManager():
    def __init__(self, control_dof=3):
        self.running = False
        self.running_r = False

        self.control_dof = control_dof

        self.current_pos = [0, 0, 0]
        self.previous_goal = [0, 0, 0]

        self.current_pos_r = [0, 0, 0]
        self.previous_goal_r = [0, 0, 0]

        self.br = tf.TransformBroadcaster()
        self.br_r = tf.TransformBroadcaster()
        rospy.init_node('xarm_teleop', anonymous=True)
   
        _cp = rospy.Subscriber('/controller/pose', PoseStamped, self.pose_cb)
        self.pose_pub = rospy.Publisher('/mss/m2s/pose_pub', PoseStamped, queue_size=1)
        # self.pub_gripper = rospy.Publisher('/mss/m2s/gripper_pub', PoseStamped, queue_size=1)
        command_sub_ = rospy.Subscriber('/teleop/command', Int8, self.command_cb)
        state_sub_ = rospy.Subscriber('/L_xarm7/xarm_states', RobotMsg, self.state_cb)

        _cp_r = rospy.Subscriber('/controller_r/pose', PoseStamped, self.pose_cb_r)
        self.pose_pub_r = rospy.Publisher('/mss/m2s/pose_pub_r', PoseStamped, queue_size=1)
        command_sub_r = rospy.Subscriber('/teleop_r/command', Int8, self.command_cb_r)
        state_sub_r = rospy.Subscriber('/R_xarm7/xarm_states', RobotMsg, self.state_cb_r)

        rospy.spin()
       
    def command_cb(self, msg):
        if msg.data == TeleopCommand.START:
            print("teleopcommand:start")
            self.running = True
        elif msg.data == TeleopCommand.STOP:
            self.running = False

    def command_cb_r(self, msg):
        if msg.data == TeleopCommand.START:
            print("teleopcommand_r:start")
            self.running_r = True
        elif msg.data == TeleopCommand.STOP:
            self.running_r = False

    def state_cb(self, msg):
        self.current_pos = [msg.pose[0] * 0.001, msg.pose[1] * 0.001, msg.pose[2] * 0.001]
    
    def state_cb_r(self, msg):
        self.current_pos_r = [msg.pose[0] * 0.001, msg.pose[1] * 0.001, msg.pose[2] * 0.001]

    def pose_cb(self, data):
        pose_goal = copy.deepcopy(data)
        # dx = data.pose.position.x - self.previous_goal[0]
        # dy = data.pose.position.y - self.previous_goal[1]
        # dz = data.pose.position.z - self.previous_goal[2]
        # pose_goal.pose.position.x = self.current_pos[0] + dx
        # pose_goal.pose.position.y = self.current_pos[1] + dy
        # pose_goal.pose.position.z = self.current_pos[2] + dz

        if(self.running):
            self.pose_pub.publish(pose_goal)

        # self.previous_goal[0] = data.pose.position.x
        # self.previous_goal[1] = data.pose.position.y
        # self.previous_goal[2] = data.pose.position.z

        # [For Debug] 
        if self.control_dof == 3:
            # always point downward
            pose_goal.pose.orientation.x = -1.0
            pose_goal.pose.orientation.y = 0.0
            pose_goal.pose.orientation.z = 0.0
            pose_goal.pose.orientation.w = 0.0
        elif self.control_dof == 2:
            print("control dof is 2")
            # pose_goal.pose.position.x = 0.3
            pose_goal.pose.position.z = 0.2
            pose_goal.pose.orientation.x = -1.0
            pose_goal.pose.orientation.y = 0.0
            pose_goal.pose.orientation.z = 0.0
            pose_goal.pose.orientation.w = 0.0
        elif self.control_dof == 4:
            o = pose_goal.pose.orientation
            o_euler = (R.from_quat([o.x, o.y, o.z, o.w])).as_euler('zyx')
            print(o_euler)
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
            o = pose_goal.pose.orientation
            print("o", o)
            o_quat = (R.from_quat([o.x, o.y, o.z, o.w]) * R.from_euler('xyz', [0, 0., 0])).as_quat()
            print("o_quat", o_quat)
            pose_goal.pose.orientation.x = o_quat[0]
            pose_goal.pose.orientation.y = o_quat[1]
            pose_goal.pose.orientation.z = o_quat[2]
            pose_goal.pose.orientation.w = o_quat[3]
            # raise NotImplementedError

        self.br.sendTransform(
            (pose_goal.pose.position.x, pose_goal.pose.position.y, pose_goal.pose.position.z), 
            (pose_goal.pose.orientation.x, pose_goal.pose.orientation.y, pose_goal.pose.orientation.z, pose_goal.pose.orientation.w), 
            rospy.Time.now(),
            "reference_frame",
            "L_link_base"
            )


    def pose_cb_r(self, data):
        pose_goal_r = copy.deepcopy(data)
        # dx = data.pose.position.x - self.previous_goal[0]
        # dy = data.pose.position.y - self.previous_goal[1]
        # dz = data.pose.position.z - self.previous_goal[2]
        # pose_goal.pose.position.x = self.current_pos[0] + dx
        # pose_goal.pose.position.y = self.current_pos[1] + dy
        # pose_goal.pose.position.z = self.current_pos[2] + dz
        print("pose_cb_r", data)
        pose_goal_r.pose.position.y -= 0.5
        if(self.running_r):
            
            self.pose_pub_r.publish(pose_goal_r)

        # self.previous_goal[0] = data.pose.position.x
        # self.previous_goal[1] = data.pose.position.y
        # self.previous_goal[2] = data.pose.position.z

        # [For Debug] 
        if self.control_dof == 3:
            # always point downward
            pose_goal_r.pose.orientation.x = -1.0
            pose_goal_r.pose.orientation.y = 0.0
            pose_goal_r.pose.orientation.z = 0.0
            pose_goal_r.pose.orientation.w = 0.0
        elif self.control_dof == 2:
            print("control dof is 2")
            # pose_goal.pose.position.x = 0.3
            pose_goal_r.pose.position.z = 0.2
            pose_goal_r.pose.orientation.x = -1.0
            pose_goal_r.pose.orientation.y = 0.0
            pose_goal_r.pose.orientation.z = 0.0
            pose_goal_r.pose.orientation.w = 0.0
        elif self.control_dof == 4:
            o_r = pose_goal_r.pose.orientation
            o_euler_r = (R.from_quat([o_r.x, o_r.y, o_r.z, o_r.w])).as_euler('zyx')
            print(o_euler_r)
            o_euler_r[1] = 0
            o_euler_r[2] = np.pi
            o_quat_r = (R.from_euler('zyx', o_euler_r)).as_quat()
            pose_goal_r.pose.orientation.x = o_quat_r[0]
            pose_goal_r.pose.orientation.y = o_quat_r[1]
            pose_goal_r.pose.orientation.z = o_quat_r[2]
            pose_goal_r.pose.orientation.w = o_quat_r[3]
        else:
            # TODO: implement 4DoF & 6DoF control
            # TODO: check if rotation is correct
            o_r = pose_goal_r.pose.orientation
            print("o", o_r)
            o_quat_r = (R.from_quat([o_r.x, o_r.y, o_r.z, o_r.w]) * R.from_euler('xyz', [0, 0., 0])).as_quat()
            print("o_quat", o_quat_r)
            pose_goal_r.pose.orientation.x = o_quat_r[0]
            pose_goal_r.pose.orientation.y = o_quat_r[1]
            pose_goal_r.pose.orientation.z = o_quat_r[2]
            pose_goal_r.pose.orientation.w = o_quat_r[3]
            # raise NotImplementedError

        self.br_r.sendTransform(
            (pose_goal_r.pose.position.x, pose_goal_r.pose.position.y, pose_goal_r.pose.position.z), 
            (pose_goal_r.pose.orientation.x, pose_goal_r.pose.orientation.y, pose_goal_r.pose.orientation.z, pose_goal_r.pose.orientation.w), 
            rospy.Time.now(),
            "reference_frame_R",
            "R_link_base"
            )#"R_link_base"


if __name__ == '__main__':
    control_dof = rospy.get_param("/teleop/control_dof")
    print("control_dof", control_dof)
    try:
        if control_dof == 6:
            MSSystemManager(control_dof=6)
        elif control_dof == 4:
            MSSystemManager(control_dof=4)
        elif control_dof == 2:
            print("control dof is 2")
            MSSystemManager(control_dof=2)
        else:  # control_dof == 3:
            MSSystemManager()
    except rospy.ROSInterruptException:
        pass
