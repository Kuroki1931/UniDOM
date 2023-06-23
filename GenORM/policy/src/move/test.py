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
    parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate in training')
    parser.add_argument('--num_plasticine_point', type=int, default=2000, help='Point Number of Plasticine')
    parser.add_argument('--num_goal_point', type=int, default=2000, help='Point Number of Primitive')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer for training')
    
    parser.add_argument("--algo", type=str, default='action')
    parser.add_argument("--env_name", type=str, default="Move-v1")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sdf_loss", type=float, default=500)
    parser.add_argument("--density_loss", type=float, default=500)
    parser.add_argument("--contact_loss", type=float, default=1)
    parser.add_argument("--soft_contact_loss", action='store_true')
    parser.add_argument("--num_steps", type=int, default=150)
    # differentiable physics parameters
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--softness", type=float, default=6666.)
    parser.add_argument("--optim", type=str, default='Adam', choices=['Adam', 'Momentum'])
    return parser.parse_args()

tf.random.set_seed(1234)
CHECK_POINT_PATH = '/root/ExPCP/policy/log/2023-05-01_09-36/no_para/2023-05-02_06-22/model/0039_weights.ckpt'


def test(args):
    '''LOG'''
    args = parse_args()
    
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))

    action_size = 3
    num_point = args.num_plasticine_point + args.num_goal_point

    model = CLS_SSG_Model(args.batch_size, action_size)
   
    model.build([(args.batch_size, num_point, 3), (args.batch_size, num_point, 5)])
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

    for i in range(500, 550):
        version = i + 1
        goal_state = np.load(f"/root/ExPCP/policy/pbm/goal_state/goal_state1/{version}/goal_state.npy")
        test_env = args.env_name.split('-')[0]
        env.reset()

        # set goal state
        env.taichi_env.initialize()
        env.taichi_env.initialize_update_target(f'envs/assets/{test_env}3D-v{version}.npy')
        env.taichi_env.loss.reset()

        # set stick pos
        state = env.taichi_env.get_state()
        with open(f'/root/ExPCP/policy/pbm/goal_state/goal_state1/{version}/randam_value.txt', mode="r") as f:
            stick_pos = json.load(f)
        state['state'][-1][0] = stick_pos['add_stick_x']
        state['state'][-1][2] = stick_pos['add_stick_y']
        env.taichi_env.set_state(**state)

        # set randam parameter: mu, lam, yield_stress
        np.random.seed(version)
        mu = np.random.uniform(500, 1500)
        lam = np.random.uniform(500, 1500)
        yield_stress = np.random.uniform(200, 1000)
        print('parameter', mu, lam, yield_stress)
        env.taichi_env.set_parameter(mu, lam, yield_stress)

        output_dir = f"{'/'.join(CHECK_POINT_PATH.split('/')[:-1])}/evaluation/{timestr}/{test_env}/{version}"
        os.makedirs(output_dir, exist_ok=True)

        pc_encode = np.zeros((args.num_plasticine_point + args.num_goal_point, 2))
        pc_encode[:args.num_plasticine_point, 0] = 1
        pc_encode[args.num_plasticine_point:, 1] = 1

        imgs = []
        for t in range(args.num_steps):
            print(t, '/', args.num_steps)

            test_plasticine_pc = env.taichi_env.simulator.get_x(0)
            test_primtiive_pc = env.taichi_env.primitives[0].get_state(0)[:3]

            test_points = sample_pc(test_plasticine_pc, goal_state, args.num_plasticine_point, args.num_goal_point)
            vector = test_points - test_primtiive_pc
            vector_encode = np.hstack([vector, pc_encode])

            act = model.forward_pass([
                tf.cast(tf.convert_to_tensor(test_points[None]), tf.float32),
                tf.cast(tf.convert_to_tensor(vector_encode[None]), tf.float32)
            ], False, 1)
            act = act.numpy()[0]
            print(act)
            try:
                _, _, _, loss_info = env.step(act)
            except:
                continue
            last_iou = loss_info['incremental_iou']
            
            if t % 100 == 0:
                print(f"Saving gif at {t} steps")
                img = env.render(mode='rgb_array')
                pimg = Image.fromarray(img)
                I1 = ImageDraw.Draw(pimg)
                I1.text((5, 5), f'mu{mu:.2f},lam{lam:.2f},yield_stress{yield_stress:.2f}', fill=(255, 0, 0))
                imgs.append(pimg)
        
        imgs[0].save(f"{output_dir}/{last_iou:.4f}_{t}.gif", save_all=True, append_images=imgs[1:], loop=0)
        with open(f'{output_dir}/last_iou_{t}.txt', 'w') as f:
            f.write(str(last_iou))
    

if __name__ == '__main__':
	args = parse_args()
	test(args)
