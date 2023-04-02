#!/usr/bin/python3
import sys
import moveit_commander
import rospy
import math
from std_msgs.msg import Float32
from xarm_gripper.msg import MoveActionGoal
from moveit_msgs.msg import RobotTrajectory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

class DualXArmGripperTeleopInterface():
    def __init__(self, update_rate=10):
        self.update_rate = update_rate
        self.trigger = None
        self.trigger_r = None

        rospy.init_node('xarm_gripper_teleop', anonymous=True)

        rate = rospy.Rate(self.update_rate)

        _ct = rospy.Subscriber('/controller/trigger', Float32, self.trigger_cb)
        pub_gripper = rospy.Publisher('/L_xarm7/gripper_move/goal', MoveActionGoal, queue_size=1)
        pub_jt = rospy.Publisher('/L_xarm7/gripper_trajectory_controller/command', JointTrajectory, queue_size=1)
        
        _ct_r = rospy.Subscriber('/controller_r/trigger', Float32, self.trigger_cb_r)
        pub_gripper_r = rospy.Publisher('/R_xarm7/gripper_move/goal', MoveActionGoal, queue_size=1)
        pub_jt_r = rospy.Publisher('/R_xarm7/gripper_trajectory_controller/command', JointTrajectory, queue_size=1)


        while not rospy.is_shutdown():
            # TODO: discard message if not fresh?
            if self.trigger is not None:
                gripper_goal = MoveActionGoal()
                gripper_goal.goal.target_pulse = (1.0-self.trigger.data)*850.0
                gripper_goal.goal.pulse_speed = 5000.0
                gripper_goal.header.stamp = rospy.Time.now()
                
                pub_gripper.publish(gripper_goal)
                # print('Gripper: ', gripper_goal.goal.target_pulse)

            if self.trigger_r is not None:
                gripper_goal = MoveActionGoal()
                gripper_goal.goal.target_pulse = (1.0-self.trigger_r.data)*850.0
                gripper_goal.goal.pulse_speed = 5000.0
                gripper_goal.header.stamp = rospy.Time.now()
                
                pub_gripper_r.publish(gripper_goal)
                # print('Gripper: ', gripper_goal.goal.target_pulse)

            rate.sleep()
    
    def trigger_cb(self, data):
        self.trigger = data

    def trigger_cb_r(self, data):
        self.trigger_r = data


if __name__ == '__main__':
    try:
        DualXArmGripperTeleopInterface()
    except rospy.ROSInterruptException:
        pass
