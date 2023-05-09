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
    parser.add_argument('--batch_size', type=int, default=64, help='batch size in training')
    parser.add_argument('--epoch', default=10000, type=int, help='number of epoch in training')
    parser.add_argument('--save_epoch', default=5, type=int, help='save epoch')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate in training')
    parser.add_argument('--num_plasticine_point', type=int, default=200, help='Point Number of Plasticine')
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
BASE_DIR = '/root/ExPCP/policy/data/Pinch_100_2000_100_2000_100_2000/2023-05-09_01-02'
BASE_TASK = BASE_DIR.split('/')[-2]
BASE_DATE = BASE_DIR.split('/')[-1]


def load_dataset(in_file, batch_size, num_point):

    assert os.path.isfile(in_file), '[error] dataset path not found'

    shuffle_buffer = 50000

    def _extract_fn(data_record):
        in_features = {
            'points': tf.io.FixedLenFeature([num_point * 3], tf.float32),
            'vector': tf.io.FixedLenFeature([num_point * 3], tf.float32),
            'action': tf.io.FixedLenFeature([3], tf.float32)
        }

        return tf.io.parse_single_example(data_record, in_features)
    
    def _preprocess_fn(sample):
        points = sample['points']
        vector = sample['vector']
        action = sample['action']

        points = tf.reshape(points, (num_point, 3))
        vector = tf.reshape(vector, (num_point, 3))

        return points, vector, action

    dataset = tf.data.TFRecordDataset(in_file)
    dataset = dataset.shuffle(shuffle_buffer)
    dataset = dataset.map(_extract_fn)
    dataset = dataset.map(_preprocess_fn)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    return dataset

def train(args):
    '''LOG'''
    args = parse_args()
    
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    exp_dir = Path('./log/')
    exp_dir.mkdir(exist_ok=True)
    exp_dir = exp_dir.joinpath(f'./{BASE_TASK}/')
    exp_dir.mkdir(exist_ok=True)
    exp_dir = exp_dir.joinpath(f'./{BASE_DATE}/')
    exp_dir.mkdir(exist_ok=True)
    exp_dir = exp_dir.joinpath(f'./no_para/')
    exp_dir.mkdir(exist_ok=True)
    exp_dir = exp_dir.joinpath(f'./{timestr}/')
    exp_dir.mkdir(exist_ok=True)
    
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (exp_dir, 'bc'))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    def log_string(str):
        logger.info(str)
        print(str)
    log_string('PARAMETER ...')
    log_string(args)

    '''env'''
    set_random_seed(args.seed)
    env = make(args.env_name, nn=(args.algo=='nn'), sdf_loss=args.sdf_loss,
								density_loss=args.density_loss, contact_loss=args.contact_loss,
								soft_contact_loss=args.soft_contact_loss)
    env.seed(args.seed)
   
    action_size = 3
    num_point = args.num_plasticine_point

    model = CLS_SSG_Model(args.batch_size, action_size)
    train_ds = load_dataset(f'data/{BASE_TASK}/{BASE_DATE}/train_experts.tfrecord', args.batch_size, num_point)
    validation_ds = load_dataset(f'data/{BASE_TASK}/{BASE_DATE}/validation_experts.tfrecord', args.batch_size, num_point)

    model.build([(args.batch_size, num_point, 3), (args.batch_size, num_point, 3)])
    print(model.summary())

    model.compile(
		optimizer=keras.optimizers.Adam(args.lr, clipnorm=0.1),
		loss='mean_squared_error',
		metrics='mean_squared_error',
		weighted_metrics='mean_squared_error'
	)
    
    parameter_list = BASE_TASK.split('_')[1:]
    parameter_list = [int(parameter) for parameter in parameter_list]
    mu_bottom, mu_upper = parameter_list[0], parameter_list[1]
    lam_bottom, lam_upper = parameter_list[2], parameter_list[3]
    yield_stress_bottom, yield_stress_upper = parameter_list[4], parameter_list[5]

    best_success_count = 0
    for epoch in range(args.epoch):
        log_string('Train epoch: %4f' % epoch)
        history = model.fit(
			train_ds,
			validation_data = validation_ds,
			validation_steps = 10,
			validation_freq = 10,
			callbacks = [
                keras.callbacks.EarlyStopping(
                    'mean_squared_error', min_delta=0.01, patience=10),
                keras.callbacks.TensorBoard(
                    f'{exp_dir}', update_freq=50),
                keras.callbacks.ModelCheckpoint(
                    f'{exp_dir}/model/{epoch:04d}_weights.ckpt', 'mean_squared_error', save_weights_only=True, save_best_only=True, save_freq='epoch')
            ],
			epochs = 1,
			verbose = 1
		)
        log_string('mean_squared_error: %4f' % history.history['loss'][0])

        success_count = 0 
        if (epoch+1) % args.save_epoch == 0 or epoch == 0:
            for i in tqdm(range(500)):
                test_env = args.env_name.split('-')[0]
                env.reset()

                # set randam parameter: mu, lam, yield_stress
                np.random.seed(epoch+i)
                mu = np.random.uniform(mu_bottom, mu_upper)
                lam = np.random.uniform(lam_bottom, lam_upper)
                yield_stress = np.random.uniform(yield_stress_bottom, yield_stress_upper)
                env.taichi_env.set_parameter(mu, lam, yield_stress)

                output_dir = exp_dir.joinpath(f'{test_env}/')
                output_dir.mkdir(exist_ok=True)

                imgs = []
                best_max_x = 0
                for t in range(args.num_steps):
                    test_plasticine_pc = env.taichi_env.simulator.get_x(0)
                    test_primtiive_pc = env.taichi_env.primitives[0].get_state(0)[:3]

                    test_points = sample_pc(test_plasticine_pc, args.num_plasticine_point)
                    vector = test_points - test_primtiive_pc

                    act = model.forward_pass([
			            tf.cast(tf.convert_to_tensor(test_points[None]), tf.float32),
			            tf.cast(tf.convert_to_tensor(vector[None]), tf.float32),
			        ], False, 1)
                    act = act.numpy()[0]
                    
                    max_x = env.taichi_env.simulator.get_x(0).max(axis=0)[0]
                    if max_x > best_max_x:
                        best_max_x = max_x

                    _, _, _, loss_info = env.step(act)

                    if t % 1 == 0:
                    # if t+1 == args.num_steps:
                        img = env.render(mode='rgb_array')
                        pimg = Image.fromarray(img)
                        I1 = ImageDraw.Draw(pimg)
                        I1.text((5, 5), f'mu{mu:.2f},lam{lam:.2f},yield_stress{yield_stress:.2f}', fill=(255, 0, 0))
                        imgs.append(pimg)
                
                # white space time
                white_time = 100
                for t in range(white_time):
                    max_x = env.taichi_env.simulator.get_x(0).max(axis=0)[0]
                    if max_x > best_max_x:
                        best_max_x = max_x
                    env.step(np.array([0, 0, 0]))
                    if t % 2 == 0:
                        img = env.render(mode='rgb_array')
                        pimg = Image.fromarray(img)
                        I1 = ImageDraw.Draw(pimg)
                        I1.text((5, 5), f'mu{mu:.2f},lam{lam:.2f},yield_stress{yield_stress:.2f}', fill=(255, 0, 0))
                        imgs.append(pimg)

                success = best_max_x > 0.55
                imgs[0].save(f"{output_dir}/{epoch}_{i}_{best_max_x:.4f}_{success}.gif", save_all=True, append_images=imgs[1:], loop=0)
                with open(f'{output_dir}/last_iou_{epoch}_{i}.txt', 'w') as f:
                    f.write(f'{best_max_x},{mu},{lam},{yield_stress}')
                success_count += int(success)
        if success_count > best_success_count:
            model.save_weights(f'{exp_dir}/model/best_weights.ckpt')
        log_string('success_count: %4f' % success_count)


if __name__ == '__main__':
	args = parse_args()
	train(args)
