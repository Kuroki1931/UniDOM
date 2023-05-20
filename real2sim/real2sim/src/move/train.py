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
import argparse
import logging
import tensorflow as tf
from tensorflow import keras
from pathlib import Path
from scipy.spatial.distance import cdist

from tqdm import tqdm
from models.cls_ssg_model import CLS_SSG_Model
from PIL import Image
from PIL import ImageDraw

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
    parser.add_argument('--batch_size', type=int, default=16, help='batch size in training')
    parser.add_argument('--epoch', default=10000, type=int, help='number of epoch in training')
    parser.add_argument('--save_epoch', default=100, type=int, help='save epoch')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate in training')
    parser.add_argument('--num_plasticine_point', type=int, default=3000, help='Point Number of Plasticine')
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
BASE_DIR = '/root/real2sim/real2sim/data/Move_500_10500_0.2_0.4_200_200/2023-05-20_06-24'
BASE_TASK = BASE_DIR.split('/')[-2]
BASE_DATE = BASE_DIR.split('/')[-1]


def load_dataset(in_file, batch_size, num_point):

    assert os.path.isfile(in_file), '[error] dataset path not found'

    shuffle_buffer = 50000

    def _extract_fn(data_record):
        in_features = {
            'points': tf.io.FixedLenFeature([num_point * 3], tf.float32),
            'parameters': tf.io.FixedLenFeature([2], tf.float32)
        }

        return tf.io.parse_single_example(data_record, in_features)
    
    def _preprocess_fn(sample):
        points = sample['points']
        parameters = sample['parameters']
        points = tf.reshape(points, (num_point, 3))

        return points, parameters

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

    parameters_size = 2
    surface_index = np.load(f'data/{BASE_TASK}/{BASE_DATE}/surface_index.npy')
    num_point = surface_index.sum()

    model = CLS_SSG_Model(args.batch_size, parameters_size)
    train_ds = load_dataset(f'data/{BASE_TASK}/{BASE_DATE}/train_experts.tfrecord', args.batch_size, num_point)
    validation_ds = load_dataset(f'data/{BASE_TASK}/{BASE_DATE}/validation_experts.tfrecord', args.batch_size, num_point)
    E_list = np.load(f'data/{BASE_TASK}/{BASE_DATE}/E.npy').tolist()
    Poisson_list = np.load(f'data/{BASE_TASK}/{BASE_DATE}/Poisson.npy').tolist()
    yield_stress_list = np.load(f'data/{BASE_TASK}/{BASE_DATE}/yield_stress.npy').tolist()

    model.build((args.batch_size, num_point, 3))
    print(model.summary())

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

    best_cd_loss = 0
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

        cd_loss = 0
        if (epoch+1) % args.save_epoch == 0 or epoch == 0:
            for i in tqdm(range(10000, 10005)):
                test_env = args.env_name.split('-')[0]
                env.reset()
                output_dir = exp_dir.joinpath(f'{test_env}/')
                output_dir.mkdir(exist_ok=True)

                # set randam parameter: mu, lam, yield_stress
                np.random.seed(epoch+i)
                E = np.random.uniform(E_bottom, E_upper)
                Poisson = np.random.uniform(Poisson_bottom, Poisson_upper)
                yield_stress = np.random.uniform(yield_stress_bottom, yield_stress_upper)
                env.taichi_env.set_parameter(E, Poisson, yield_stress)

                action = np.array([[0, 0.6, 0]]*150)
                frames = []
                for idx, act in enumerate(action):
                    env.step(act)
                    if idx % 5 == 0:
                        img = env.render(mode='rgb_array')
                        pimg = Image.fromarray(img)
                        I1 = ImageDraw.Draw(pimg)
                        I1.text((5, 5), f'E{E:.2f},Poisson{Poisson:.2f},yield_stress{yield_stress:.2f}', fill=(255, 0, 0))
                        frames.append(pimg)
                frames[0].save(f'{output_dir}/{epoch}_{i}_ground_truth_demo.gif', save_all=True, append_images=frames[1:], loop=0)
                last_state = env.taichi_env.simulator.get_x(0)

                pred_parameters = model.forward_pass(tf.cast(tf.convert_to_tensor(last_state[None]), tf.float32), False, 1)
                pred_parameters = pred_parameters.numpy()

                env.reset()
                pred_E = pred_parameters[0][0] * np.std(E_list) + np.mean(E_list)
                pred_Poisson = pred_parameters[0][1] * np.std(Poisson_list) + np.mean(Poisson_list)

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
                frames[0].save(f'{output_dir}/{epoch}_{i}_pred_demo.gif', save_all=True, append_images=frames[1:], loop=0)
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

                with open(f'{output_dir}/{epoch}_{i}.txt', 'w') as f:
                    f.write(f'{chamfer_dist}, {E}, {Poisson}, {yield_stress}, {pred_E},{pred_Poisson},{yield_stress}')

        if cd_loss > best_cd_loss:
            best_cd_loss = cd_loss
            model.save_weights(f'{exp_dir}/model/best_weights.ckpt')
        log_string('chamfer_distance: %4f' % cd_loss)

if __name__ == '__main__':
	args = parse_args()
	train(args)
