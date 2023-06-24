import os
import sys
import json
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

from tqdm import tqdm
from models.cls_ssg_model import MLP
from PIL import Image
from PIL import ImageDraw

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, '../../pbm'))

from plb.envs import make
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
    parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate in training')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer for training')
    
    parser.add_argument("--algo", type=str, default='action')
    parser.add_argument("--env_name", type=str, default="Torus-v1")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sdf_loss", type=float, default=500)
    parser.add_argument("--density_loss", type=float, default=500)
    parser.add_argument("--contact_loss", type=float, default=1)
    parser.add_argument("--soft_contact_loss", action='store_true')
    parser.add_argument("--num_steps", type=int, default=12)
    # differentiable physics parameters
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--softness", type=float, default=6666.)
    parser.add_argument("--optim", type=str, default='Adam', choices=['Adam', 'Momentum'])
    return parser.parse_args()

tf.random.set_seed(1234)
CHECK_POINT_PATH = '/root/ExPCP/policy/log/E_para/Torus_500_10500_0.2_0.4_200_200/2023-05-23_14-11/2023-05-23_14-16/model/best_weights.ckpt'
BASE_TASK = CHECK_POINT_PATH.split('/')[-5]
BASE_DATE = CHECK_POINT_PATH.split('/')[-4]


def test(args):
    '''LOG'''
    args = parse_args()
    
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))

    output_size = 2
    model = MLP(args.batch_size, output_size)
   
    model.build([(args.batch_size, 1), (args.batch_size, 1)])
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
    env = make(args.env_name, nn=(args.algo=='nn'), sdf_loss=args.sdf_loss,
								density_loss=args.density_loss, contact_loss=args.contact_loss,
								soft_contact_loss=args.soft_contact_loss)
    env.seed(args.seed)

    E_list = np.load(f'data/{BASE_TASK}/{BASE_DATE}/E.npy').tolist()
    Poisson_list = np.load(f'data/{BASE_TASK}/{BASE_DATE}/Poisson.npy').tolist()
    yield_stress_list = np.load(f'data/{BASE_TASK}/{BASE_DATE}/yield_stress.npy').tolist()
    goal_point_list = np.load(f'data/{BASE_TASK}/{BASE_DATE}/goal_point.npy').tolist()
    
    parameter_list = BASE_TASK.split('_')[1:]
    parameter_list = [float(parameter) for parameter in parameter_list]
    E_bottom, E_upper = parameter_list[0], parameter_list[1]
    Poisson_bottom, Poisson_upper = parameter_list[2], parameter_list[3]
    yield_stress_bottom, yield_stress_upper = parameter_list[4], parameter_list[5]

    sum_diff = 0
    for i in range(10000000, 10000050):
        env.reset()

        # set randam parameter: mu, lam, yield_stress
        np.random.seed(i)
        E = np.random.uniform(10300, 10500)
        Poisson = np.random.uniform(Poisson_bottom, Poisson_upper)
        yield_stress = np.random.uniform(yield_stress_bottom, yield_stress_upper)
        env.taichi_env.set_parameter(E, Poisson, yield_stress)
        
        lower_E = E // 1000 * 1000  # Floor division by 1000 then multiply by 1000 to get the lower bound
        upper_E = lower_E + 1000
        indices = [i for i, val in enumerate(E_list) if lower_E <= val <= upper_E]
        E_goal_point_list = np.array(goal_point_list)[indices].tolist()
        conditioned_goal_point = np.array([np.random.uniform(np.min(E_goal_point_list), np.max(E_goal_point_list))])

        output_dir = f"{'/'.join(CHECK_POINT_PATH.split('/')[:-2])}/evaluation_upper_id"
        os.makedirs(output_dir, exist_ok=True)
        
        E_value = (E - np.mean(E_list)) / np.std(E_list)
        Poisson_value = (Poisson - np.mean(Poisson_list)) / np.std(Poisson_list)
        yield_stress_value = (yield_stress - np.mean(yield_stress_list)) / np.std(yield_stress_list)
        parameters = np.array([E_value])

        release_point = model.forward_pass([
            tf.cast(tf.convert_to_tensor(conditioned_goal_point[None]), tf.float32),
            tf.cast(tf.convert_to_tensor(parameters[None]), tf.float32)
        ], False, 1)
        start_pos = release_point.numpy()[0]
        start_pos = np.array([start_pos[0], start_pos[1], 0.5])
        
        initial_primitive_pos = env.taichi_env.primitives[0].get_state(0)[:3]
        init_actions = np.linspace(initial_primitive_pos, start_pos, 200)
        init_actions = np.diff(init_actions, n=1, axis=0)
        init_actions = np.vstack([init_actions, init_actions[0][None, :]])
        action = np.concatenate([init_actions, np.array([[0, 0, 0]]*400)])
        
        initial_state = env.taichi_env.simulator.get_x(0)
        rope_bottom_index = np.argmin(initial_state[:, 1])

        try:
            best_max_x = 0
            env.taichi_env.primitives.set_softness()
            frames = []
            for idx, act in enumerate(action):
                env.step(act)
                
                if idx == 300:
                    release_point = env.taichi_env.primitives[0].get_state(0)[:3]
                    env.taichi_env.primitives.set_softness1(0)
                # if idx % 5 == 0:
                #     img = env.render(mode='rgb_array')
                #     pimg = Image.fromarray(img)
                #     I1 = ImageDraw.Draw(pimg)
                #     I1.text((5, 5), f'E{E:.2f},Poisson{Poisson:.2f},yield_stress{yield_stress:.2f}', fill=(255, 0, 0))
                #     frames.append(pimg)
                state = env.taichi_env.simulator.get_x(0)
                max_x = state[rope_bottom_index][0]
                if max_x > best_max_x:
                    best_max_x = max_x
            # frames[0].save(f'{output_dir}/{epoch}_{i}_best_x{best_max_x:.2f}_x{start_pos[0]:.2f}_y{start_pos[1]:.2f}_E{E:.2f},Poisson{Poisson:.2f},yield_stress{yield_stress:.2f}_.gif', save_all=True, append_images=frames[1:], loop=0)
            diff = conditioned_goal_point[0] - best_max_x
            print(i, diff)
            sum_diff += np.abs(diff)
            with open(f'{output_dir}/{i}.txt', 'w') as f:
                f.write(f'{diff}, {conditioned_goal_point[0]}, {best_max_x}, {start_pos[0]}, {start_pos[1]}, {E}, {Poisson}')
        except:
            print('error')
            break
        
    print('update best sum diff-----------------: %4f' % sum_diff)


if __name__ == '__main__':
	args = parse_args()
	test(args)