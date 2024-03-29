import os
import sys
import json
import random
import datetime

sys.path.insert(0, './')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

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
    parser.add_argument("--env_name", type=str, default="Pinch-v1")
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
CHECK_POINT_PATH = '/root/ExPCP/policy/log/nu/Pinch_500_10500_0.2_0.4_200_200/2023-08-08_17-42/2023-08-08_20-18/model/best_weights.ckpt'
BASE_TASK = CHECK_POINT_PATH.split('/')[-5]
BASE_DATE = CHECK_POINT_PATH.split('/')[-4]
BASE_TYPE = CHECK_POINT_PATH.split('/')[-6]


def test(args):
    '''LOG'''
    args = parse_args()
    
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))

    output_size = 1
    model = MLP(args.batch_size, output_size)
   
    model.build([(args.batch_size, 2), (args.batch_size, 1)])
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

    E_list = np.load(f'data/{BASE_TASK}/{BASE_TYPE}/{BASE_DATE}/E.npy').tolist()
    Poisson_list = np.load(f'data/{BASE_TASK}/{BASE_TYPE}/{BASE_DATE}/Poisson.npy').tolist()
    yield_stress_list = np.load(f'data/{BASE_TASK}/{BASE_TYPE}/{BASE_DATE}/yield_stress.npy').tolist()
    goal_point_list = np.load(f'data/{BASE_TASK}/{BASE_TYPE}/{BASE_DATE}/goal_point.npy').tolist()
    
    parameter_list = BASE_TASK.split('_')[1:]
    parameter_list = [float(parameter) for parameter in parameter_list]
    E_bottom, E_upper = parameter_list[0], parameter_list[1]
    Poisson_bottom, Poisson_upper = parameter_list[2], parameter_list[3]
    yield_stress_bottom, yield_stress_upper = parameter_list[4], parameter_list[5]

    sum_diff = 0
    count = 0
    for i in range(10000000, 10000050):
        env.reset()

        # set randam parameter: mu, lam, yield_stress
        np.random.seed(i)
        E = np.random.uniform(E_bottom, E_upper)
        Poisson = np.random.uniform(Poisson_bottom, Poisson_upper)
        yield_stress = np.random.uniform(yield_stress_bottom, yield_stress_upper)
        env.taichi_env.set_parameter(E, Poisson, yield_stress)
        
        lower_E = E // 500 * 500  # Floor division by 1000 then multiply by 1000 to get the lower bound
        upper_E = lower_E + 500
        indices = [i for i, val in enumerate(E_list) if lower_E <= val <= upper_E]

        E_goal_point_list = np.array(goal_point_list)[indices].tolist()
        x_values = [item[0] for item in E_goal_point_list]
        y_values = [item[1] for item in E_goal_point_list]
        # Determine the range
        min_x, max_x = min(x_values), max(x_values)
        min_y, max_y = min(y_values), max(y_values)
        # Generate a random coordinate within the range
        random_x = random.uniform(min_x, max_x)
        random_y = random.uniform(min_y, max_y)
        conditioned_goal_point = np.array([random_x, random_y])

        output_dir = f"{'/'.join(CHECK_POINT_PATH.split('/')[:-2])}/evaluation"
        os.makedirs(output_dir, exist_ok=True)
        
        E_value = (E - np.mean(E_list)) / np.std(E_list)
        Poisson_value = (Poisson - np.mean(Poisson_list)) / np.std(Poisson_list)
        yield_stress_value = (yield_stress - np.mean(yield_stress_list)) / np.std(yield_stress_list)
        parameters = np.array([Poisson])

        action_value = model.forward_pass([
            tf.cast(tf.convert_to_tensor(conditioned_goal_point[None]), tf.float32),
            tf.cast(tf.convert_to_tensor(parameters[None]), tf.float32)
        ], False, 1)
        action_value = action_value[0][0]
        
        print('parameter', action_value, E, Poisson, yield_stress)
        
        T = 5
        action = np.concatenate([np.array([[action_value, 0, 0]]*T), np.array([[0, 0, 0]]*50)])
        
        initial_state = env.taichi_env.simulator.get_x(0)
        rope_bottom_index = np.argmin(initial_state[:, 1])

        try:
            best_max_y = 0
            best_max_x = 0
            env.taichi_env.primitives.set_softness()
            frames = []
            for idx, act in enumerate(action):
                env.step(act)
                if idx % 1 == 0:
                    img = env.render(mode='rgb_array')
                    pimg = Image.fromarray(img)
                    I1 = ImageDraw.Draw(pimg)
                    I1.text((5, 5), f'E{E:.2f},Poisson{Poisson:.2f},yield_stress{yield_stress:.2f}', fill=(255, 0, 0))
                    frames.append(pimg)

                state = env.taichi_env.simulator.get_x(0)
                max_x = state[rope_bottom_index][0]
                max_y = state[rope_bottom_index][1]
                if max_y > best_max_y:
                    best_max_y = max_y
                    best_max_x = max_x
            frames[0].save(f'{output_dir}/eval_{i}_action{action_value:.3f}_best_x{best_max_x:.2f}_best_y{best_max_y:.2f}_E{E:.2f},Poisson{Poisson:.2f},yield_stress{yield_stress:.2f}_.gif', save_all=True, append_images=frames[1:], loop=0)
            diff = ((conditioned_goal_point[0] - best_max_x)**2 + (conditioned_goal_point[1] - best_max_y)**2)**(1/2)
            sum_diff += np.abs(diff)
            count += 1
            with open(f'{output_dir}/eval_{i}.txt', 'w') as f:
                f.write(f'{diff}, {conditioned_goal_point[0]}, {conditioned_goal_point[1]}, {best_max_x}, {best_max_y}, {action_value}, {E}, {Poisson}')
        except:
            print('error')
        
    mean_diff = sum_diff / count
    print('update best sum diff-----------------: %4f' % mean_diff)


if __name__ == '__main__':
	args = parse_args()
	test(args)