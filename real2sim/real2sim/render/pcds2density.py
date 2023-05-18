import argparse
import random
import numpy as np
import os
import torch
import sys
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

    output_path = '/root/real2sim/real2sim/real_points/yellow'
    
    pcds = np.load(f'{output_path}/real_pcds_modify.npy', allow_pickle=True)
    grid_mass_list = []
    default_grid_mass = env.taichi_env.simulator.get_grid_mass(0)
    default_density = default_grid_mass.sum()
    print('defualt density', default_density)
    for i in range(pcds.shape[0]):
        print(i, '----------------------------------------')
        env.taichi_env.initialize()
        pcd = pcds[i]
        env.taichi_env.simulator.reset(pcd)
        state = env.taichi_env.get_state()
        env.taichi_env.set_state(**state)
        grid_mass = env.taichi_env.get_grid_mass(0)
        x = env.taichi_env.simulator.get_x(0)
        grid_mass[:5] = 0
        density = grid_mass.sum()
        print('late', density)
        modify_grid_mass = grid_mass * default_density / density
        print('modify_grid_mass', modify_grid_mass.sum())
        grid_mass_list.append(modify_grid_mass)
    np.save(f'{output_path}/real_densities.npy', np.array(grid_mass_list))


if __name__ == '__main__':
    main()
