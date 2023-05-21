import os
import sys
import json
import random
import datetime

sys.path.insert(0, './')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import numpy as np
import torch
import pickle
import argparse
import logging
import tensorflow as tf
from tensorflow import keras
from pathlib import Path

from tqdm import tqdm
from models.cls_ssg_model import CLS_SSG_Model
from PIL import Image
from PIL import ImageDraw
from scipy.spatial.distance import cdist

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, '../../pbm'))

from plb.envs import make


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
    parser.add_argument('--num_plasticine_point', type=int, default=1000, help='Point Number of Plasticine')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer for training')
    
    parser.add_argument("--algo", type=str, default='action')
    parser.add_argument("--env_name", type=str, default="Move-v1")
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
CHECK_POINT_PATH = '/root/real2sim/real2sim/log/Move_500_10500_0.2_0.4_200_200/2023-05-20_06-24/2023-05-20_08-55/model/best_weights.ckpt'
BASE_TASK = CHECK_POINT_PATH.split('/')[-5]
BASE_DATE = CHECK_POINT_PATH.split('/')[-4]


def test(args):
    '''LOG'''
    args = parse_args()

    parameter_list = BASE_TASK.split('_')[1:]
    parameter_list = [float(parameter) for parameter in parameter_list]
    E_list = np.load(f'data/{BASE_TASK}/{BASE_DATE}/E.npy').tolist()
    Poisson_list = np.load(f'data/{BASE_TASK}/{BASE_DATE}/Poisson.npy').tolist()
    surface_index = np.load(f'data/{BASE_TASK}/{BASE_DATE}/surface_index.npy')

    parameters_size = 2
    num_point = surface_index.sum()

    model = CLS_SSG_Model(args.batch_size, parameters_size)
   
    model.build((args.batch_size, num_point, 3))
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

    now = datetime.datetime.now()
    yield_stress = 200
    action = np.array([[0, 0.6, 0]]*150)

    cd_loss = 0
    for rope_type in ['red', 'white', 'yellow']:
        env.reset()

        input_path = f'/root/real2sim/real2sim/real_points/{rope_type}'
        output_dir = f'/root/real2sim/real2sim/real_points/{rope_type}/{now}/learning'
        os.makedirs(output_dir, exist_ok=True)

        last_state = np.load(f'{input_path}/real_pcds_modify.npy', allow_pickle=True)[-1]
        if last_state.shape[0] > num_point:
            diff = last_state.shape[0] - num_point
            random_index = np.random.choice(last_state.shape[0], diff, replace=False)
            last_state = np.delete(last_state, random_index, axis=0)
        else:
            diff = num_point - last_state.shape[0]
            random_index = np.random.choice(last_state.shape[0], diff, replace=True)
            last_state = np.concatenate([last_state, last_state[random_index]], axis=0)

        pred_parameters = model.forward_pass(tf.cast(tf.convert_to_tensor(last_state[None]), tf.float32), False, 1)
        pred_parameters = pred_parameters.numpy()

        env.reset()
        pred_E = pred_parameters[0][0] * np.std(E_list) + np.mean(E_list)
        pred_Poisson = pred_parameters[0][1] * np.std(Poisson_list) + np.mean(Poisson_list)
        print('parameter', pred_E, pred_Poisson, yield_stress)
        env.taichi_env.set_parameter(pred_E, pred_Poisson, yield_stress)

        frames = []
        for idx, act in enumerate(action):
            env.step(act)
            if idx % 5 == 0:
                img = env.render(mode='rgb_array')
                pimg = Image.fromarray(img)
                I1 = ImageDraw.Draw(pimg)
                I1.text((5, 5), f'E{pred_E:.2f},Poisson{pred_Poisson:.2f},yield_stress{yield_stress:.2f}', fill=(255, 0, 0))
                frames.append(pimg)
        frames[0].save(f'{output_dir}/pred_demo.gif', save_all=True, append_images=frames[1:], loop=0)
        pred_last_state = env.taichi_env.simulator.get_x(0)

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
        cd_loss += chamfer_dist

        with open(f'{output_dir}/setting.txt', 'w') as f:
            f.write(f'{chamfer_dist}, {pred_E},{pred_Poisson},{yield_stress}')


if __name__ == '__main__':
	args = parse_args()
	test(args)
