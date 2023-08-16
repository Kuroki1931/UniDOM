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
from models.cls_ssg_model import MLP_NO_PARA

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
    parser.add_argument('--epoch', default=200, type=int, help='number of epoch in training')
    parser.add_argument('--save_epoch', default=20, type=int, help='save epoch')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate in training')
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
CHECK_POINT_PATH = '/root/GenORM/policy/log/full/Rope_500_10500_0.2_0.4_200_200/2023-08-16_03-09/2023-08-16_03-24/model/0099_weights.ckpt'
BASE_DIR = '/root/GenORM/policy/data/Rope_500_10500_0.2_0.4_200_200/full/2023-08-16_03-13'
BASE_TASK = BASE_DIR.split('/')[-3]
BASE_TYPE = BASE_DIR.split('/')[-2]
BASE_DATE = BASE_DIR.split('/')[-1]


def load_dataset(in_file, batch_size):

    assert os.path.isfile(in_file), '[error] dataset path not found'

    shuffle_buffer = 50000

    def _extract_fn(data_record):
        in_features = {
            'height': tf.io.FixedLenFeature([1], tf.float32),
            'action': tf.io.FixedLenFeature([1], tf.float32)
        }

        return tf.io.parse_single_example(data_record, in_features)
    
    def _preprocess_fn(sample):
        height = sample['height']
        action = sample['action']

        return height, action

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
    exp_dir = exp_dir.joinpath(f'./{BASE_TYPE}/')
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
   
    output_size = 1

    model = MLP_NO_PARA(args.batch_size, output_size)
    train_ds = load_dataset(f'data/{BASE_TASK}/{BASE_TYPE}/{BASE_DATE}/train_real_experts.tfrecord', args.batch_size)
    validation_ds = load_dataset(f'data/{BASE_TASK}/{BASE_TYPE}/{BASE_DATE}/validation_real_experts.tfrecord', args.batch_size)

    model.build((args.batch_size, 1))
    print(model.summary())
    
    E_list = np.load(f'data/{BASE_TASK}/{BASE_TYPE}/{BASE_DATE}/E.npy').tolist()
    Poisson_list = np.load(f'data/{BASE_TASK}/{BASE_TYPE}/{BASE_DATE}/Poisson.npy').tolist()
    yield_stress_list = np.load(f'data/{BASE_TASK}/{BASE_TYPE}/{BASE_DATE}/yield_stress.npy').tolist()

    model.compile(
		optimizer=keras.optimizers.Adam(args.lr, clipnorm=0.1),
		loss='mean_squared_error',
		metrics='mean_squared_error',
		weighted_metrics='mean_squared_error'
	)
    model.load_weights(CHECK_POINT_PATH).expect_partial()

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
        count = 0
        if (epoch+1) % args.save_epoch == 0 or epoch == 0:
            for i in range(5):
                test_env = args.env_name.split('-')[0]
                env.reset()

                height = np.array([0.16])
                E = 584.21
                Poisson = 0.25
                yield_stress = 200
                
                E_value = (E - np.mean(E_list)) / np.std(E_list)
                Poisson_value = (Poisson - np.mean(Poisson_list)) / np.std(Poisson_list)
                yield_stress_value = (yield_stress - np.mean(yield_stress_list)) / np.std(yield_stress_list)
                parameters = np.array([E_value, Poisson_value])
                
                action_value = model.forward_pass(tf.cast(tf.convert_to_tensor(height[None]), tf.float32), False, 1)
                action_value = action_value[0][0]
                
                print('---------------', action_value)

                height = np.array([0.16])
                E = 6828.33
                Poisson = 0.26
                yield_stress = 200
                
                E_value = (E - np.mean(E_list)) / np.std(E_list)
                Poisson_value = (Poisson - np.mean(Poisson_list)) / np.std(Poisson_list)
                yield_stress_value = (yield_stress - np.mean(yield_stress_list)) / np.std(yield_stress_list)
                parameters = np.array([E_value, Poisson_value])
                
                action_value = model.forward_pass(tf.cast(tf.convert_to_tensor(height[None]), tf.float32), False, 1)
                action_value = action_value[0][0]
                
                print('---------------', action_value)
if __name__ == '__main__':
	args = parse_args()
	train(args)