#!/usr/bin/env python3
import cv2
import numpy as np
import rospy
import pickle
from cv_bridge import CvBridge

from std_msgs.msg import Int8
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist

import argparse
import random
import glob
import numpy as np
import os
import torch
import sys
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
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

RL_ALGOS = ['sac', 'td3', 'ppo']
DIFF_ALGOS = ['action', 'nn']

import datetime, os, cv2
import matplotlib.pyplot as plt
from PIL import Image


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_args():
    parser=argparse.ArgumentParser()
    parser.add_argument("--algo", type=str, default='action')
    parser.add_argument("--env_name", type=str, default="Table-v1")
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


def main(args):
    set_random_seed(args.seed)
    env = make(args.env_name, nn=(args.algo=='nn'), sdf_loss=args.sdf_loss,
                            density_loss=args.density_loss, contact_loss=args.contact_loss,
                            soft_contact_loss=args.soft_contact_loss)
    env.seed(args.seed)
    target_env_name = args.env_name.split('-')[0]

    files = glob.glob('../experts/*/action/*.npy')

    for file in files:
        print(target_env_name, file)
        if target_env_name in file:
            env_name = file.split('/')[-3]
            env.reset()
            
            expert_time = file.split('/')[-1][:-4]
            action = np.load(file)

            frames = []
            for idx, act in enumerate(action):
                start_time = datetime.datetime.now()
                if env_name == 'Chopsticks':
                    act = np.insert(act, 3, [0]*3)
                else:
                    act = act[:3]
                env.step(act)
                if idx % 5 == 0:
                    img = env.render(mode='rgb_array')
                    pimg = Image.fromarray(img)
                    frames.append(pimg)
                end_time = datetime.datetime.now()
                take_time = end_time - start_time
                take_time = take_time.total_seconds()
                print('take time', take_time)
            frames[0].save(f"{'/'.join(file.split('/')[:-1])}/{expert_time}_{action.shape[0]}.gif", save_all=True, append_images=frames[1:], loop=0)

            # create dataset for bc
            env.reset()
            action_list = []
            plasticine_pc_list = []
            primitive_pc_list = []
            reward_list = []
            loss_info_list = []
            last_iou_list = []
            grid_mass_list = []

            # point_list = []
            for i in range(len(action)):
                if env_name == 'Chopsticks':
                    act = np.insert(action[i], 3, [0]*3)
                else:
                    act = action[i][:3]
                action_list.append(act)

                plasticine_pc, primtiive_pc = env.get_obs(0, i)
                plasticine_pc_list.append(plasticine_pc.tolist())
                primitive_pc_list.append(primtiive_pc.tolist())

                obs, r, done, loss_info = env.step(act)
                last_iou = loss_info['incremental_iou']
                reward_list.append(r)
                loss_info_list.append(loss_info)

                grid_mass = env.taichi_env.get_grid_mass(0)
                grid_mass_list.append(grid_mass)

            experts_output_dir = '/'.join(file.split('/')[:-2])
            if not os.path.exists(experts_output_dir):
                os.makedirs(experts_output_dir, exist_ok=True)
            
            # with open(f'{experts_output_dir}/point_list_{expert_time}.txt', mode="wb") as f:
            #     pickle.dump(point_list, f)

            print('length', i, 'r', r, 'last_iou', last_iou)
            bc_data = {
                'action': np.array(action_list),
                'rewards': np.array(reward_list),
                'env_name': env_name,
                'plasticine_pc': np.array(plasticine_pc_list),
                'primitive_pc': np.array(primitive_pc_list),
                'loss_info_list': loss_info_list
            }
            print(args.env_name)
            
            print(action.shape, np.array(reward_list).shape, np.array(plasticine_pc_list).shape, np.array(primitive_pc_list).shape)
            with open(f'{experts_output_dir}/expert_{last_iou:.4f}_{expert_time}.pickle', 'wb') as f:
                pickle.dump(bc_data, f)

            np.save(f'{experts_output_dir}/expert_{last_iou:.4f}_{expert_time}_grid_mass.npy', np.array(grid_mass_list))
            

if __name__ == '__main__':
    args = get_args()
    main(args)