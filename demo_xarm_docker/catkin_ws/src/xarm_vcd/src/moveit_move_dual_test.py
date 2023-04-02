#!/usr/bin/env python3
# coding: UTF-8
import sys
import copy
import rospy
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg


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


moveit_commander.roscpp_initialize(sys.argv)
rospy.init_node("xarm_moveit_controller", anonymous=True)

robot = moveit_commander.RobotCommander()

# ロボットの情報を表示
print("group_names:",robot.get_group_names(),"\n")
#print("planning_frame:",robot.get_planning_frame(),"\n")

# ロボットの現在の座標を取得
#print(robot.get_current_state())

scene = moveit_commander.PlanningSceneInterface()

# ロボットを動かす
group_name = "both_xarm7"
L_end_effector_link = "L_link7"
R_end_effector_link = "R_link7"

move_group = moveit_commander.MoveGroupCommander(group_name)

L_target_pose = move_group.get_current_pose(L_end_effector_link)
R_target_pose = move_group.get_current_pose(R_end_effector_link)

# ロボットをz方向に動かす
L_target_pose.pose.position.z = 0.5
R_target_pose.pose.position.z= 0.5

move_group.set_max_velocity_scaling_factor(1.0) # 速度を最大に設定, これをしないと速度制限のせいでガタガタ動く
#move_group.set_max_acceleration_scaling_factor(1.0)

move_group.set_num_planning_attempts(10)
move_group.clear_pose_targets()
move_group.set_pose_target(L_target_pose, L_end_effector_link)
move_group.set_pose_target(R_target_pose, R_end_effector_link)

# # ロボットの移動(plan and executeを別々に実行)
# _,plan,_,_= move_group.plan()
# move_group.execute(plan,wait=True)

# ロボットの移動(plan and execute)
move_group.go(wait=True)

#gripperの動作検証
#close_open_hand("L_xarm_gripper")
close_open_hand("R_xarm_gripper")

move_hand("L_xarm_gripper",close=True)
move_hand("L_xarm_gripper",close=False)
