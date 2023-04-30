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

# Define the range of the initial state of the rope
x_range = [0.35, 0.65]
y_range = [0.4935, 0.5065]
z_range = [0, 0.013]
stick_radius = 0.06

# Calculate the initial state points of the rope
NUM_POINTS = 3000
width = [x_range[1] - x_range[0], y_range[1] - y_range[0], z_range[1] - z_range[0]]
init_pos = [(x_range[0]+x_range[1])/2, (y_range[0]+y_range[1])/2, (z_range[0]+z_range[1])/2]
rope_initial_state = (np.random.random((NUM_POINTS, 3)) * 2 - 1) * (0.5 * np.array(width)) + np.array(init_pos)
rope_length = x_range[1] - x_range[0]

def closest_tangent_point(angle):
    # stick pos
    angle_radians = math.radians(angle)
    center = [x_range[0], (y_range[0] + y_range[1]) / 2]
    radius = 0.5 * rope_length
    r = math.sqrt(random.uniform((radius**2)/5, radius**2))  # Use the square root to maintain uniform distribution
    X = center[0] + r * math.cos(angle_radians)
    Y = center[1] + r * math.sin(angle_radians)
    stick_pos = np.array([X, Y, 0])

    # tarngent pos
    radius = stick_radius + (y_range[1] - y_range[0])/2
    x = center[0]
    y = center[1]

    distance = math.sqrt((X - x)**2 + (Y - y)**2)
    theta = math.acos(radius / distance)
    theta_a = math.pi/2 - theta
    d = math.sqrt(distance**2 - radius**2)
    if angle > 0:
        theta_b = angle_radians - theta_a
        T1_x = x + d * math.cos(theta_b)
        T1_y = y + d * math.sin(theta_b)
    else:
        theta_b = angle_radians + theta_a
        T1_x = x + d * math.cos(theta_b)
        T1_y = y + d * math.sin(theta_b)

    return (T1_x, T1_y), stick_pos

def goal_state_pattern1(tangent_point, add_stick_pos):
    # rope1
    rope1_length = math.sqrt((x_range[0] - tangent_point[0])**2 + ((y_range[0] + y_range[1]) / 2 - tangent_point[1])**2)
    width = [rope1_length, y_range[1] - y_range[0], z_range[1] - z_range[0]]
    init_pos = [x_range[0]+rope1_length/2, (y_range[0]+y_range[1])/2, (z_range[0]+z_range[1])/2]
    rope1_num_points = int(NUM_POINTS/rope_length*rope1_length)
    rope1_state = (np.random.random((rope1_num_points, 3)) * 2 - 1) * (0.5 * np.array(width)) + np.array(init_pos)
    angle_radians = math.atan2(tangent_point[1] - (y_range[0] + y_range[1]) / 2, tangent_point[0] - x_range[0])

    pivot_point = np.array([x_range[0], (y_range[0] + y_range[1]) / 2, (z_range[0] + z_range[1]) / 2])
    rotated_rope1 = np.zeros((rope1_num_points, 3))

    for i in range(rope1_num_points):
        point = rope1_state[i] - pivot_point
        rotated_x = point[0] * math.cos(angle_radians) - point[1] * math.sin(angle_radians)
        rotated_y = point[0] * math.sin(angle_radians) + point[1] * math.cos(angle_radians)
        rotated_rope1[i] = np.array([rotated_x, rotated_y, point[2]]) + pivot_point
    
    # rope2
    if add_stick_pos[1] > 0.5:
        rope2_pos = add_stick_pos + np.array([0, stick_radius + (y_range[1] - y_range[0])/2, 0])
    else:
        rope2_pos = add_stick_pos - np.array([0, stick_radius + (y_range[1] - y_range[0])/2, 0])
    radius = stick_radius + (y_range[1] - y_range[0])/2
    angle1 = math.atan2(tangent_point[1] - add_stick_pos[1], tangent_point[0] - add_stick_pos[0])
    angle2 = math.atan2(rope2_pos[1] - add_stick_pos[1], rope2_pos[0] - add_stick_pos[0])
    delta_angle = angle2 - angle1

    if abs(delta_angle) > math.pi:
        delta_angle += math.copysign(2 * math.pi, -delta_angle)
    
    if delta_angle > 0 and add_stick_pos[1] < 0.5:
        delta_angle -= 2 * math.pi
    if delta_angle < 0 and add_stick_pos[1] > 0.5:
        delta_angle += 2 * math.pi
    
    arc_length = abs(radius * delta_angle)
    rope2_num_points = int(NUM_POINTS/rope_length*arc_length)
    
    points_on_arc = []
    points_around_arc = []

    for _ in range(rope2_num_points):
        t = random.uniform(0, delta_angle)
        
        x_on_arc = add_stick_pos[0] + radius * math.cos(angle1 + t)
        y_on_arc = add_stick_pos[1] + radius * math.sin(angle1 + t)
        z_on_arc = random.uniform(z_range[0], z_range[1])
        points_on_arc.append((x_on_arc, y_on_arc, z_on_arc))
        
        uniform_samples = np.random.uniform(0, 1)
        transformed_samples = uniform_samples ** 2
        A = -(y_range[1] - y_range[0])/2
        B = (y_range[1] - y_range[0])/2
        scaled_samples = transformed_samples * (B - A) + A
        r = radius + scaled_samples

        x_around_arc = add_stick_pos[0] + r * math.cos(angle1 + t)
        y_around_arc = add_stick_pos[1] + r * math.sin(angle1 + t)
        z_around_arc = random.uniform(z_range[0], z_range[1])
        points_around_arc.append((x_around_arc, y_around_arc, z_around_arc))
    rope2_points = np.array(points_around_arc)

    # # rope3
    # rest_rope_length = rope_length - rope1_length - arc_length
    # if rest_rope_length > 0:
    #     width = [rest_rope_length, y_range[1] - y_range[0], z_range[1] - z_range[0]]
    #     init_pos = [rope2_pos[0]-rest_rope_length/2, rope2_pos[1], (z_range[0]+z_range[1])/2]
    #     rope3_num_points = int(NUM_POINTS/rest_rope_length*rope1_length)
    #     rope3_state = (np.random.random((rope3_num_points, 3)) * 2 - 1) * (0.5 * np.array(width)) + np.array(init_pos)
    #     return np.concatenate([rotated_rope1, rope2_points, rope3_state])
    return np.concatenate([rotated_rope1, rope2_points])


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
    os.makedirs(f'{base_path}/goal_state1', exist_ok=True)

    index_path = f'{base_path}/goal_state1/index.txt'
    if os.path.exists(index_path):
        with open(index_path, 'r') as file:
            index_list = [int(line.strip()) for line in file]
        index = np.max(index_list)
        index += 1
    else:
        index_list = []
        index = 1

    for _ in range(steps):
        index_list.append(index)

        angle_ranges = [(40, 60)]
        selected_range = random.choice(angle_ranges)
        stick_angle = random.uniform(selected_range[0], selected_range[1])
        tangent_point, add_stick_pos = closest_tangent_point(stick_angle)
        goal_state = goal_state_pattern1(tangent_point, add_stick_pos)
        goal_state = goal_state[:, [0, 2, 1]]

        env.taichi_env.initialize()
        env.taichi_env.simulator.reset(goal_state)
        state = env.taichi_env.get_state()
        env.taichi_env.set_state(**state)
        grid_mass = env.taichi_env.get_grid_mass(0)
        grid_mass[:5, :5, :5] = 0
        np.save(f'/root/ExPCP/policy/pbm/plb/envs/assets/Move3D-v{index}', grid_mass)

        os.makedirs(f'{base_path}/goal_state1/{index}', exist_ok=True)
        random_value = {
            'add_stick_x': add_stick_pos[0],
            'add_stick_y': add_stick_pos[1],
        }
        with open(f'{base_path}/goal_state1/{index}/randam_value.txt', mode="w") as f:
            json.dump(random_value, f, indent=4)
        np.save(f'{base_path}/goal_state1/{index}/Move3D-v{index}', grid_mass)
        np.save(f'{base_path}/goal_state1/{index}/goal_state', goal_state)

        # Saving the list to a text file
        with open(index_path, 'w') as file:
            for item in index_list:
                file.write(f"{item}\n")
        index += 1


if __name__ == '__main__':
    main()
