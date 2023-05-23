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
    parser.add_argument('--batch_size', type=int, default=64, help='batch size in training')
    parser.add_argument('--epoch', default=300, type=int, help='number of epoch in training')
    parser.add_argument('--save_epoch', default=20, type=int, help='save epoch')
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
BASE_DIR = '/root/ExPCP/policy/data/Torus_500_10500_0.2_0.4_200_200/2023-05-23_14-11'
BASE_TASK = BASE_DIR.split('/')[-2]
BASE_DATE = BASE_DIR.split('/')[-1]


def load_dataset(in_file, batch_size):

    assert os.path.isfile(in_file), '[error] dataset path not found'

    shuffle_buffer = 50000

    def _extract_fn(data_record):
        in_features = {
            'goal_point': tf.io.FixedLenFeature([1], tf.float32),
            'parameters': tf.io.FixedLenFeature([1], tf.float32),
            'release_point': tf.io.FixedLenFeature([2], tf.float32)
        }

        return tf.io.parse_single_example(data_record, in_features)
    
    def _preprocess_fn(sample):
        goal_point = sample['goal_point']
        parameters = sample['parameters']
        release_point = sample['release_point']

        return goal_point, parameters, release_point

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
    exp_dir = exp_dir.joinpath(f'./E_para/')
    exp_dir.mkdir(exist_ok=True)
    exp_dir = exp_dir.joinpath(f'./{BASE_TASK}/')
    exp_dir.mkdir(exist_ok=True)
    exp_dir = exp_dir.joinpath(f'./{BASE_DATE}/')
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
   
    output_size = 2

    model = MLP(args.batch_size, output_size)
    train_ds = load_dataset(f'data/{BASE_TASK}/{BASE_DATE}/train_experts.tfrecord', args.batch_size)
    validation_ds = load_dataset(f'data/{BASE_TASK}/{BASE_DATE}/validation_experts.tfrecord', args.batch_size)

    model.build([(args.batch_size, 1), (args.batch_size, 1)])
    print(model.summary())
    
    E_list = np.load(f'data/{BASE_TASK}/{BASE_DATE}/E.npy').tolist()
    Poisson_list = np.load(f'data/{BASE_TASK}/{BASE_DATE}/Poisson.npy').tolist()
    yield_stress_list = np.load(f'data/{BASE_TASK}/{BASE_DATE}/yield_stress.npy').tolist()
    goal_point_list = np.load(f'data/{BASE_TASK}/{BASE_DATE}/goal_point.npy').tolist()

    model.compile(
		optimizer=keras.optimizers.Adam(args.lr, clipnorm=0.1),
		loss='mean_squared_error',
		metrics='mean_squared_error',
		weighted_metrics='mean_squared_error'
	)
    
    parameter_list = BASE_TASK.split('_')[1:]
    parameter_list = [float(parameter) for parameter in parameter_list]
    E_bottom, E_upper = parameter_list[0], parameter_list[1]
    Poisson_bottom, Poisson_upper = parameter_list[2], parameter_list[3]
    yield_stress_bottom, yield_stress_upper = parameter_list[4], parameter_list[5]

    best_sum_diff = 999999999999999999
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
                    f'{exp_dir}/model/{epoch:04d}_weights.ckpt', 'mean_squared_error', save_weights_only=True, save_best_only=True,  save_freq=10)
            ],
			epochs = 1,
			verbose = 1
		)
        log_string('mean_squared_error: %4f' % history.history['loss'][0])

        sum_diff = 0
        if (epoch+1) % args.save_epoch == 0 or epoch == 0:
            for i in tqdm(range(50000000, 50000010)):
                test_env = args.env_name.split('-')[0]
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
                conditioned_goal_point = np.array([np.random.uniform(np.min(E_goal_point_list), np.max(E_goal_point_list))])

                output_dir = exp_dir.joinpath(f'{test_env}/')
                output_dir.mkdir(exist_ok=True)
                
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
                    sum_diff += np.abs(diff)
                    with open(f'{output_dir}/{epoch}_{i}.txt', 'w') as f:
                        f.write(f'{diff}, {conditioned_goal_point[0]}, {best_max_x}, {start_pos[0]}, {start_pos[1]}, {E}, {Poisson}')
                except:
                    print('error')
                    break
                
            if sum_diff < best_sum_diff:
                best_sum_diff = sum_diff
                model.save_weights(f'{exp_dir}/model/best_weights.ckpt')
                log_string('update best sum diff-----------------: %4f' % sum_diff)
            else:
                log_string('sum diff------------------: %4f' % sum_diff)


if __name__ == '__main__':
	args = parse_args()
	train(args)
