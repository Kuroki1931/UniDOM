import os
import sys
import random
import datetime

sys.path.insert(0, './')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import numpy as np
import torch
import pickle
import argparse
import logging
import tensorflow as tf
from tensorflow import keras
from pathlib import Path
from clip import tokenize

from tqdm import tqdm
from util.clip import build_model, load_clip
from models.cls_ssg_model import CLS_SSG_Model
from PIL import Image
from util.lang_goal import LANG_GOAL

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'pbm'))
sys.path.append('/root/fairseq/examples/MMPT')

from plb.envs import make
from plb.algorithms.logger import Logger
from plb.algorithms.discor.run_sac import train as train_sac
from plb.algorithms.ppo.run_ppo import train_ppo
from plb.algorithms.TD3.run_td3 import train_td3
from plb.optimizer.solver import solve_action
from plb.optimizer.solver_nn import solve_nn
from util.preprocess import sample_pc
# from videoclip import pooled_text

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('training')
    parser.add_argument('--use_cpu', action='store_true', default=False, help='use cpu mode')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size in training')
    parser.add_argument('--epoch', default=10000, type=int, help='number of epoch in training')
    parser.add_argument('--save_epoch', default=300, type=int, help='save epoch')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate in training')
    parser.add_argument('--num_plasticine_point', type=int, default=3072, help='Point Number of Plasticine')
    parser.add_argument('--num_primitive_point', type=int, default=1024, help='Point Number of Primitive')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer for training')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate')
    parser.add_argument('--use_normals', action='store_true', default=False, help='use normals')
    parser.add_argument('--process_data', action='store_true', default=False, help='save data offline')
    parser.add_argument('--use_uniform_sample', action='store_true', default=False, help='use uniform sampiling')
    parser.add_argument('--clip_type', type=int, default=1, help='0 -> video clip, 1 -> clip')
    parser.add_argument('--command_type', type=int, default=0, help='0 ->whole, 1 -> separate')
    parser.add_argument('--command_num', type=int, default=4, help='command num')
    
    parser.add_argument("--algo", type=str, default='action')
    parser.add_argument("--env_name", type=str, default="Rollingpin-v1")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sdf_loss", type=float, default=500)
    parser.add_argument("--density_loss", type=float, default=500)
    parser.add_argument("--contact_loss", type=float, default=1)
    parser.add_argument("--soft_contact_loss", action='store_true')
    parser.add_argument("--num_steps", type=int, default=100)
    # differentiable physics parameters
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--softness", type=float, default=6666.)
    parser.add_argument("--optim", type=str, default='Adam', choices=['Adam', 'Momentum'])
    return parser.parse_args()

tf.random.set_seed(1234)
CHECK_POINT_PATH = '/root/xarm/DMLC/policy/log/2023-03-20_14-57/2023-03-20_15-39/model/weights.ckpt'


def set_tool(tool):
    # pick up tool
    # go to the first position
    pass

def obtain_point_clouds():
    pass

def set_initial_position(tool, point_clouds):
    # calcuate positions (e.g. center)
    # move
    pass

def obtain_primitive_position(args, env_lang, tool, point_clouds):
    '''LOG'''
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    experts_dir = CHECK_POINT_PATH.split('/')[-4]

    '''CLIP'''
    clip_model, _ = load_clip("RN50", jit=False)
    clip_rn50 = build_model(clip_model.state_dict())
    clip_rn50 = clip_rn50.float()
    clip_rn50.eval()
    def clip_emb(lang):
        tokens = tokenize([lang]).numpy()
        token_tensor = torch.from_numpy(tokens)
        lang_goal_emb, lang_token_embs = clip_rn50.encode_text_with_embeddings(token_tensor)
        return lang_goal_emb[0].to('cpu').detach().numpy().copy()
    
    action_size = 3
    num_point = args.num_primitive_point + args.num_plasticine_point

    model = CLS_SSG_Model(args.batch_size, action_size)
   
    model.build([(args.batch_size, num_point, 3), (args.batch_size, num_point, 5), (args.batch_size, 1024)])
    print(model.summary())
    model.compile(
		optimizer=keras.optimizers.Adam(args.lr, clipnorm=0.1),
		loss='mean_squared_error',
		metrics='mean_squared_error',
		weighted_metrics='mean_squared_error'
	)
    model.load_weights(CHECK_POINT_PATH).expect_partial()
    
    '''env'''
    set_random_seed(args.seed)
    if tool == 'cutter':
        env = make('Rope-v1', nn=(args.algo=='nn'), sdf_loss=args.sdf_loss,
                                    density_loss=args.density_loss, contact_loss=args.contact_loss,
                                    soft_contact_loss=args.soft_contact_loss)
        env.seed(args.seed)
    else:
        pass
    env.reset()

    # test_env_name, test_lang = env_lang.split('&&')
    test_env_name, index, test_lang, test_object, test_manipulation, test_location = env_lang.split('&&')

    num_steps = LANG_GOAL[test_env_name]['num_steps']
    output_dir = f"{'/'.join(CHECK_POINT_PATH.split('/')[:-1])}/evaluation/{timestr}/{test_env_name}/{index}_{test_lang}"
    os.makedirs(output_dir, exist_ok=True)

    if args.clip_type == 0:
        pass
        # test_lang_goal_emb = pooled_text(test_lang)
    else:
        if args.command_type:
            object_lang_goal_emb = clip_emb(test_object)
            manipulation_lang_goal_emb = clip_emb(test_manipulation)
            location_lang_goal_emb = clip_emb(test_location)
            test_lang_goal_emb = np.concatenate([object_lang_goal_emb, manipulation_lang_goal_emb, location_lang_goal_emb])
        else:
            test_lang_goal_emb = clip_emb(test_lang)

    pc_encode = np.zeros((args.num_plasticine_point + args.num_primitive_point, 2))
    pc_encode[:args.num_plasticine_point, 0] = 1
    pc_encode[args.num_plasticine_point:, 1] = 1

    imgs = []
    primitive_position_list = []
    for t in range(num_steps):
        print(t, '/', num_steps)

        test_plasticine_pc, test_primtiive_pc = env.get_obs(0, t)
        primitive_centroid = np.mean(test_primtiive_pc, axis=0)
        primitive_position_list.append(primitive_centroid)
        if test_primtiive_pc.shape[0] == 0 or test_plasticine_pc.shape[0] == 0:
            env.step(np.array([0, 0, 0])) # plasticinelab bug?
            continue

        test_points = sample_pc(test_plasticine_pc, test_primtiive_pc, args.num_plasticine_point, args.num_primitive_point)
        vector = test_points - primitive_centroid
        vector_encode = np.hstack([vector, pc_encode])
        act = model.forward_pass([
            tf.cast(tf.convert_to_tensor(test_points[None]), tf.float32),
            tf.cast(tf.convert_to_tensor(vector_encode[None]), tf.float32),
            tf.cast(tf.convert_to_tensor(test_lang_goal_emb[None]), tf.float32)
        ], False, 1)
        act = act.numpy()[0]
        print(act)

        if 'Rollingpin' in test_env_name:
            act[0] *= -1
            act = act[[0, 2, 1]]
        _, _, _, loss_info = env.step(act)

        last_iou = loss_info['incremental_iou']

        if t % 5 == 0:
            print(f"Saving gif at {t} steps")
            imgs.append(Image.fromarray(env.render(mode='rgb_array')))

    imgs[0].save(f"{output_dir}/{last_iou:.4f}_{t}.gif", save_all=True, append_images=imgs[1:], loop=0)
    with open(f'{output_dir}/last_iou_{t}.txt', 'w') as f:
        f.write(str(last_iou))

    return primitive_position_list

# def control_robot(primitive_position_list):
#     pass

def return_tool(tool):
    # return tool
    # go to the first position
    pass


###################################################
# xarm code
###################################################
import sys
import copy
import rospy
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
import random
import yaml
from trajectory_msgs.msg import JointTrajectory
import time


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
    
    
def control_robot(group_name, end_effector_link,primitive_position_list):
    # ロボットを動かす
    move_group = moveit_commander.MoveGroupCommander(group_name)
    # ロボットをz方向に動かす
    move_group.set_max_velocity_scaling_factor(1.0) # 速度を最大に設定, これをしないと速度制限のせいでガタガタ動く
    move_group.set_num_planning_attempts(10)
    
    for i in range(len(primitive_position_list)-1):
        target_pose = move_group.get_current_pose(end_effector_link)
        target_pose.pose.position.x=primitive_position_list[i+1][2]-0.1
        target_pose.pose.position.y=primitive_position_list[i+1][0]-0.5
        target_pose.pose.position.z=primitive_position_list[i+1][1]+0.175+0.05
        print("target_pose.pose.position",target_pose.pose.position)
        
        move_group.clear_pose_targets()
        move_group.set_pose_target(target_pose, end_effector_link)
        # ロボットの移動(plan and execute)
        move_group.go(wait=True)
        
def control_robot_smooth(group_name, end_effector_link,primitive_position_list):
    pub_jt = rospy.Publisher('/xarm/xarm7_traj_controller/command', JointTrajectory, queue_size=1)
   
    # ロボットを動かす
    move_group = moveit_commander.MoveGroupCommander(group_name)
    # ロボットをz方向に動かす
    move_group.set_max_velocity_scaling_factor(1.0) # 速度を最大に設定, これをしないと速度制限のせいでガタガタ動く
    move_group.set_num_planning_attempts(10)
    
    for i in range(len(primitive_position_list)-1):
        target_pose = move_group.get_current_pose(end_effector_link)
        target_pose.pose.position.x=primitive_position_list[i+1][2]-0.1
        target_pose.pose.position.y=primitive_position_list[i+1][0]-0.5
        target_pose.pose.position.z=primitive_position_list[i+1][1]+0.175
        print("target_pose.pose.position",target_pose.pose.position)        
        # move_group.clear_pose_targets()
        # move_group.set_pose_target(target_pose, end_effector_link)
        # ロボットの移動(plan and execute)
        # move_group.go(wait=True)
        pose_goal = copy.deepcopy(target_pose)
        plan, _fraction = move_group.compute_cartesian_path([pose_goal.pose], 0.01, 0)
        
        # 
        # TODO: check duration is correct (previously rospy.Duration(1))
        plan_time = plan.joint_trajectory.points[-1].time_from_start.to_sec()
        pref_time = 1.0 / 10 * 2.0
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
            pub_jt.publish(plan.joint_trajectory)
            
def control_robot_smooth_test(group_name, end_effector_link,primitive_position_list):
    pub_jt = rospy.Publisher('/xarm/xarm7_traj_controller/command', JointTrajectory, queue_size=1)
   
    # ロボットを動かす
    move_group = moveit_commander.MoveGroupCommander(group_name)
    # ロボットをz方向に動かす
    move_group.set_max_velocity_scaling_factor(1.0) # 速度を最大に設定, これをしないと速度制限のせいでガタガタ動く
    move_group.set_num_planning_attempts(10)
    wpts = []
    for i in range(len(primitive_position_list)-1):
        target_pose = move_group.get_current_pose(end_effector_link)
        target_pose.pose.position.x=primitive_position_list[i+1][2]-0.1
        target_pose.pose.position.y=primitive_position_list[i+1][0]-0.5
        target_pose.pose.position.z=primitive_position_list[i+1][1]+0.175
        wpts.append( copy.deepcopy( target_pose.pose ) )
    print("wpts",wpts)

    move_group.clear_pose_targets()
    move_group.set_pose_target(target_pose, end_effector_link)
    # ロボットの移動(plan and execute)
    plan, _fraction = move_group.compute_cartesian_path(wpts, 0.01, 0)
    
    # 
    # TODO: check duration is correct (previously rospy.Duration(1))
    plan_time = plan.joint_trajectory.points[-1].time_from_start.to_sec()
    pref_time = 1.0 / 10 * 2.0
    max_speedup = 100.0 # 10.0

    if plan_time > pref_time:
        new_plan_time = max(pref_time, plan_time / max_speedup)
        scale = new_plan_time / plan_time
        print("scaling:", scale)

        for pt in plan.joint_trajectory.points:
            pt.time_from_start = rospy.Duration(pt.time_from_start.to_sec() * scale)

        # this speeds up the robot significantly
        plan.joint_trajectory.points = [plan.joint_trajectory.points[-1]]
        pub_jt.publish(plan.joint_trajectory)
        
    print("c_pose",move_group.get_current_pose(end_effector_link))
    
###################################################
# xarm code
###################################################

if __name__ == '__main__':
    args = parse_args()
    env_lang = 'Rope-v1&&0&&cut the center of the object&& && &&'
    tool = 'cutter'

    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node("xarm_moveit_controller", anonymous=True)

    robot = moveit_commander.RobotCommander()

    # ロボットの情報を表示
    print("group_names:",robot.get_group_names(),"\n")
    scene = moveit_commander.PlanningSceneInterface()

    # move home position
    group_name = "xarm7"
    end_effector_link = "link7"
    move_group = moveit_commander.MoveGroupCommander(group_name)
    target_pose = move_group.get_current_pose(end_effector_link)
    target_pose.pose.position.x= 0.4
    target_pose.pose.position.y= 0.0
    target_pose.pose.position.z= 0.5
    target_pose.pose.orientation.x=-1.0
    target_pose.pose.orientation.y=0.0
    target_pose.pose.orientation.z=0.0
    target_pose.pose.orientation.w=0.0
    move_single_arm(group_name,end_effector_link,target_pose)
    
    # calc primitive_position_list 
    primitive_position_list = obtain_primitive_position(args, env_lang, tool, None)
    print("primitive_position_list",primitive_position_list)
    
    # xarm motion generation
    control_robot(group_name,end_effector_link,primitive_position_list)
    #control_robot_smooth(group_name,end_effector_link,primitive_position_list)
    
    
    time.sleep(3)
    target_pose = move_group.get_current_pose(end_effector_link)
    print("target_pose", target_pose)
    target_pose.pose.position.x= 0.4
    target_pose.pose.position.y= 0.0
    target_pose.pose.position.z= 0.5
    target_pose.pose.orientation.x=-1.0
    target_pose.pose.orientation.y=0.0
    target_pose.pose.orientation.z=0.0
    target_pose.pose.orientation.w=0.0
    move_single_arm(group_name,end_effector_link,target_pose)
    
    return_tool(tool)
    