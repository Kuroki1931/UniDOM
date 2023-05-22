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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, '../../pbm'))

from plb.envs import make
from plb.algorithms.logger import Logger
from plb.algorithms.discor.run_sac import train as train_sac
from plb.algorithms.ppo.run_ppo import train_ppo
from plb.algorithms.TD3.run_td3 import train_td3
from plb.optimizer.solver import solve_action, tell_rope_break
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
    parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate in training')
    parser.add_argument('--num_plasticine_point', type=int, default=3000, help='Point Number of Plasticine')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer for training')
    
    parser.add_argument("--algo", type=str, default='action')
    parser.add_argument("--env_name", type=str, default="Rope-v1")
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
CHECK_POINT_PATH = '/root/ExPCP/policy/log/Rope_10_500_10_500_10_500/2023-05-06_03-11/no_para/2023-05-06_11-46/model/0029_weights.ckpt'
BASE_TASK = CHECK_POINT_PATH.split('/')[-6]
BASE_DATE = CHECK_POINT_PATH.split('/')[-5]
EPOCH =int(CHECK_POINT_PATH.split('/')[-1].split('_')[0])


def test(args):
    '''LOG'''
    args = parse_args()
    
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))

    action_size = 3
    num_point = args.num_plasticine_point

    model = CLS_SSG_Model(args.batch_size, action_size)
   
    model.build([(args.batch_size, num_point, 3), (args.batch_size, num_point, 3)])
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

    success_count = 0
    for i in range(1000, 1500):
        print(i)
        version = i + 1
        test_env = args.env_name.split('-')[0]
        env.reset()

        # set randam parameter: mu, lam, yield_stress
        np.random.seed(version)
        mu = np.random.uniform(10, 500)
        lam = np.random.uniform(10, 500)
        yield_stress = np.random.uniform(10, 500)
        print('parameter', mu, lam, yield_stress)
        env.taichi_env.set_parameter(mu, lam, yield_stress)

        output_dir = f"{'/'.join(CHECK_POINT_PATH.split('/')[:-2])}/evaluation"
        os.makedirs(output_dir, exist_ok=True)

        imgs = []
        for t in range(args.num_steps):
            test_plasticine_pc = env.taichi_env.simulator.get_x(0)
            test_primtiive_pc = env.taichi_env.primitives[0].get_state(0)[:3]

            test_points = sample_pc(test_plasticine_pc, args.num_plasticine_point)
            vector = test_points - test_primtiive_pc

            act = model.forward_pass([
                tf.cast(tf.convert_to_tensor(test_points[None]), tf.float32),
                tf.cast(tf.convert_to_tensor(vector[None]), tf.float32)
            ], False, 1)
            act = act.numpy()[0]
            try:
                _, _, _, loss_info = env.step(act)
            except:
                continue
            
            if t+1 == args.num_steps:
                print(f"Saving gif at {t} steps")
                img = env.render(mode='rgb_array')
                pimg = Image.fromarray(img)
                I1 = ImageDraw.Draw(pimg)
                I1.text((5, 5), f'mu{mu:.2f},lam{lam:.2f},yield_stress{yield_stress:.2f}', fill=(255, 0, 0))
                imgs.append(pimg)

        possible = tell_rope_break(img)
        if possible:
            imgs[0].save(f"{output_dir}/{EPOCH}_{i}_break.gif", save_all=True, append_images=imgs[1:], loop=0)
            with open(f'{output_dir}/last_iou_{EPOCH}_{i}.txt', 'w') as f:
                f.write(f'0,{mu},{lam},{yield_stress}')
        else:
            rope_state = env.taichi_env.simulator.get_x(0)
            rope_length = rope_state.max(axis=0)[0] - rope_state.min(axis=0)[0]
            imgs[0].save(f"{output_dir}/{EPOCH}_{i}_{rope_length:.4f}.gif", save_all=True, append_images=imgs[1:], loop=0)
            with open(f'{output_dir}/last_iou_{EPOCH}_{i}.txt', 'w') as f:
                f.write(f'{rope_length},{mu},{lam},{yield_stress}')
            success_count += 1
    print('success_count', success_count)
    

if __name__ == '__main__':
	args = parse_args()
	test(args)
