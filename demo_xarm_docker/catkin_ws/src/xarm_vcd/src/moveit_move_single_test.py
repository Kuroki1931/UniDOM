#!/usr/bin/env python3
# coding: UTF-8
import sys
import copy
import rospy
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
import random
import yaml


# 指定したgroup_nameの情報を表示
def print_move_group_info(group_name):
    print("group_name: " + group_name)
    move_group = moveit_commander.MoveGroupCommander(group_name)
    named_target = move_group.get_named_targets()
    print("named target: ", named_target)
    for target in named_target:
        print("target: ", target)
        print("pose: ", move_group.get_named_target_values(target))
    print()

# ロボットハンドの開閉(jointを直接操作)
def move_hand(group_name, close):
    move_group = moveit_commander.MoveGroupCommander(group_name)
    joint_goal = move_group.get_current_joint_values()
    print(move_group.get_joints())
    print(joint_goal)
    if close:
        joint_goal[0] = 0.85
    else:
        joint_goal[0] = 0.0

    print(joint_goal)
    move_group.go(joint_goal, wait=True)
    move_group.stop()



# ロボットハンドの開閉(targetを利用した方法)
def close_open_hand(group_name):
    move_group = moveit_commander.MoveGroupCommander(group_name)
    #gripperを閉じる
    move_group.set_named_target("close")

    try:
        val = input("Press Enter to close gripper...\n")
    except KeyboardInterrupt:
        sys.exit()
    except:
        pass
    move_group.go(wait=True)

    #gripperを開く
    move_group.set_named_target("open")
    try:
        val = input("Press Enter to open gripper...\n")
    except KeyboardInterrupt:
        sys.exit()
    except:
        pass
    move_group.go(wait=True)

def move_single_arm(group_name, end_effector_link,target_pose):
    # ロボットを動かす
    move_group = moveit_commander.MoveGroupCommander(group_name)
    # ロボットをz方向に動かす
    move_group.set_max_velocity_scaling_factor(1.0) # 速度を最大に設定, これをしないと速度制限のせいでガタガタ動く
    move_group.set_num_planning_attempts(10)
    move_group.clear_pose_targets()
    move_group.set_pose_target(target_pose, end_effector_link)
    # ロボットの移動(plan and execute)
    move_group.go(wait=True)


if __name__ == '__main__':
    try:
        with open('/root/xarm/catkin_ws/src/xarm_vcd/config/position.yaml') as file:
            pose_list = yaml.safe_load(file)
            print("pose_list",pose_list)
    except Exception as e:
        print('Exception occurred while loading YAML...', file=sys.stderr)
        print(e, file=sys.stderr)
        sys.exit(1)

    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node("xarm_moveit_controller", anonymous=True)

    robot = moveit_commander.RobotCommander()

    # ロボットの情報を表示
    print("group_names:",robot.get_group_names(),"\n")
    scene = moveit_commander.PlanningSceneInterface()

    # ロボットを動かす
    group_name = "xarm7"
    end_effector_link = "link7"
    move_group = moveit_commander.MoveGroupCommander(group_name)
    target_pose = move_group.get_current_pose(end_effector_link)
    # print("pose_list['p_home']",pose_list['p_home'])
    # target_pose.pose.position.x= 0.4
    # target_pose.pose.position.y= 0.0
    # target_pose.pose.position.z= 0.5
    # move_single_arm(group_name,end_effector_link,target_pose)

    # for i in range(5):
    #     target_pose = move_group.get_current_pose(end_effector_link)
    #     target_poseC= random.uniform(0.3, 0.4)
    #     target_pose.pose.position.y= random.uniform(-0.2, 0.2)
    #     target_pose.pose.position.z= random.uniform(0.1, 0.6)
    #     print("target_pose",target_pose)
    #     move_single_arm(group_name,end_effector_link,target_pose)
        

    # move_hand("xarm_gripper",close=True)
    # move_hand("xarm_gripper",close=False)
