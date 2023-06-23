import argparse
import random
import numpy as np
import os
import torch
import sys
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import json
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from plb.envs import make
from plb.algorithms.logger import Logger

from plb.algorithms.discor.run_sac import train as train_sac
from plb.algorithms.ppo.run_ppo import train_ppo
from plb.algorithms.TD3.run_td3 import train_td3
from plb.optimizer.solver import solve_action
from plb.optimizer.solver_nn import solve_nn

os.environ['TI_USE_UNIFIED_MEMORY'] = '0'
os.environ['TI_DEVICE_MEMORY_FRACTION'] = '0.9'
os.environ['TI_DEVICE_MEMORY_GB'] = '4'
os.environ['TI_ENABLE_CUDA'] = '0'
os.environ['TI_ENABLE_OPENGL'] = '0'


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

# Define the range of the initial state of the rope
x_range = [0.35, 0.65]
y_range = [0.4935, 0.5065]
z_range = [0, 0.013]
stick_radius = 0.015

# Calculate the initial state points of the rope
NUM_POINTS = 3000
width = [x_range[1] - x_range[0], y_range[1] - y_range[0], z_range[1] - z_range[0]]
init_pos = [(x_range[0]+x_range[1])/2, (y_range[0]+y_range[1])/2, (z_range[0]+z_range[1])/2]
rope_initial_state = (np.random.random((NUM_POINTS, 3)) * 2 - 1) * (0.5 * np.array(width)) + np.array(init_pos)
rope_length = x_range[1] - x_range[0]

def goal_state_pattern2(angle):
    angle_radians = math.radians(angle)
    pivot_point = np.array([x_range[0], (y_range[0] + y_range[1]) / 2, (z_range[0] + z_range[1]) / 2])
    rotated_rope = np.zeros((NUM_POINTS, 3))

    for i in range(NUM_POINTS):
        point = rope_initial_state[i] - pivot_point
        rotated_x = point[0] * math.cos(angle_radians) - point[1] * math.sin(angle_radians)
        rotated_y = point[0] * math.sin(angle_radians) + point[1] * math.cos(angle_radians)
        rotated_rope[i] = np.array([rotated_x, rotated_y, point[2]]) + pivot_point

    return rotated_rope


def main():
    args = get_args()
    set_random_seed(args.seed)
    env = make(args.env_name, nn=(args.algo=='nn'), sdf_loss=args.sdf_loss,
                            density_loss=args.density_loss, contact_loss=args.contact_loss,
                            soft_contact_loss=args.soft_contact_loss)
    env.seed(args.seed)
    env.reset()
    steps = 10

    base_path = '/root/ExPCP/policy/pbm/goal_state'
    os.makedirs(base_path, exist_ok=True)
    os.makedirs(f'{base_path}/goal_state2', exist_ok=True)

    index_path = f'{base_path}/goal_state2/index.txt'
    if os.path.exists(index_path):
        with open(index_path, 'r') as file:
            index_list = [int(line.strip()) for line in file]
        index = np.max(index_list)
        index += 1
    else:
        index_list = []
        index = 0

    for _ in range(steps):
        index_list.append(index)

        angle_ranges = [(-90, -30), (30, 90)]
        selected_range = random.choice(angle_ranges)
        rot_angle = random.uniform(selected_range[0], selected_range[1])
        goal_state = goal_state_pattern2(rot_angle)
        goal_state = goal_state[:, [0, 2, 1]]

        env.taichi_env.initialize()
        env.taichi_env.simulator.reset(goal_state)
        state = env.taichi_env.get_state()
        env.taichi_env.set_state(**state)
        grid_mass = env.taichi_env.get_grid_mass(0)
        np.save(f'/root/ExPCP/policy/pbm/plb/envs/assets/Move3D-v{index}', grid_mass)

        os.makedirs(f'{base_path}/goal_state2/{index}', exist_ok=True)
        random_value = {
            'rot_angle': rot_angle,
        }
        with open(f'{base_path}/goal_state2/{index}/randam_value.txt', mode="w") as f:
            json.dump(random_value, f, indent=4)
        np.save(f'{base_path}/goal_state2/{index}/Move3D-v{index}', grid_mass)

        # Saving the list to a text file
        with open(index_path, 'w') as file:
            for item in index_list:
                file.write(f"{item}\n")
        index += 1


if __name__ == '__main__':
    main()
