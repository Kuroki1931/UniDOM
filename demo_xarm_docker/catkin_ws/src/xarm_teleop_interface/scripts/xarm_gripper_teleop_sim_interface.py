#!/usr/bin/python3
import sys
import moveit_commander
import rospy
import math
from std_msgs.msg import Float32
from xarm_gripper.msg import MoveActionGoal
from moveit_msgs.msg import RobotTrajectory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


class XArmGripperTeleopInterface():
    def __init__(self, update_rate=10):
        self.update_rate = update_rate
        self.trigger = None

        rospy.init_node('xarm_gripper_teleop', anonymous=True)

        # group = moveit_commander.MoveGroupCommander('xarm_gripper')
        rate = rospy.Rate(self.update_rate)

        _ct = rospy.Subscriber('/controller/trigger', Float32, self.trigger_cb)
        pub_jt = rospy.Publisher('/xarm/gripper_trajectory_controller/command', JointTrajectory, queue_size=1)

        while not rospy.is_shutdown():
            # TODO: discard message if not fresh?
            if self.trigger is not None:
                jt = JointTrajectory()
                jt.header.stamp = rospy.Time.now()
                jt.joint_names = ['drive_joint']
                point = JointTrajectoryPoint()
                point.positions = [self.trigger.data]
                point.time_from_start = rospy.Duration(0.3)
                jt.points.append(point)
                pub_jt.publish(jt)

            rate.sleep()

    def trigger_cb(self, data):
        self.trigger = data


if __name__ == '__main__':
    try:
        XArmGripperTeleopInterface()
    except rospy.ROSInterruptException:
        pass
