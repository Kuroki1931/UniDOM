import argparse
import random
import numpy as np
import os
import torch
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from scipy.spatial.distance import cdist

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

RL_ALGOS = ['sac', 'td3', 'ppo']
DIFF_ALGOS = ['action', 'nn']

STEP = 70

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

def main():
    args = get_args()
    set_random_seed(args.seed)
    env = make(args.env_name, nn=(args.algo=='nn'), sdf_loss=args.sdf_loss,
                            density_loss=args.density_loss, contact_loss=args.contact_loss,
                            soft_contact_loss=args.soft_contact_loss)
    env.seed(args.seed)
    env.reset()

    path_list = [
        '/root/real2sim/real2sim/real_points/red/2023-05-21 02_15_29.987034', 
        '/root/real2sim/real2sim/real_points/white/2023-05-21 02_15_33.014992',
        '/root/real2sim/real2sim/real_points/yellow/2023-05-20 18:30:10.182365'
    ]
    
    for path in path_list:
        input_path = '/'.join(path.split('/')[:-1])
        last_state = np.load(f'{input_path}/real_pcds_modify.npy', allow_pickle=True)[-1]

        for i in range(2000,2010):
            env.reset()
            with open(f'{path}/{i}/setting.txt', 'rb') as f:
                setting = f.readlines()[0].decode('utf8').rstrip()
                setting = setting.split(',')

            E = float(setting[4])
            nu = float(setting[5])
            env.taichi_env.set_parameter(E, nu, 200)
            print('parameter', E, nu, 200)
            actions = np.array([[0, 0.2, 0]]*300)

            for act in actions:
               env.step(act)
            surface_index = env.taichi_env.surface_index.astype(bool)
            pred_last_state = env.taichi_env.simulator.get_x(0)[surface_index]

            def chamfer_distance(A, B):
                # compute distance matrix between A and B
                dist_matrix = cdist(A, B)
                # for each point in A, compute minimum distance to any point in B
                dist_A = np.min(dist_matrix, axis=1)
                # for each point in B, compute minimum distance to any point in A
                dist_B = np.min(dist_matrix, axis=0)
                # compute Chamfer distance
                chamfer_dist = np.mean(dist_A) + np.mean(dist_B)
                return chamfer_dist

            chamfer_dist = chamfer_distance(last_state, pred_last_state)

            with open(f'{path}/{i}/setting4.txt', 'w') as f:
                f.write(f'{chamfer_dist},{setting[1]},{setting[2]},{setting[3]},{setting[4]},{setting[5]},{setting[6]}')


if __name__ == '__main__':
    main()
