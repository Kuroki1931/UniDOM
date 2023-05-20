import os
import sys
import random
import datetime

import glob
import pickle
import json

sys.path.insert(0, './')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import numpy as np
import torch
import argparse
import tensorflow as tf
from tensorflow import keras
from pathlib import Path

from util import tf_utils
from util.preprocess import sample_pc


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('training')
    parser.add_argument('--num_plasticine_point', type=int, default=2000, help='Point Number of Plasticine')
    parser.add_argument('--num_goal_point', type=int, default=2000, help='Point Number of Plasticine')
    return parser.parse_args()


def create_example(points, vector, parameters, action):

    feature = {
		'points' : tf_utils.float_list_feature(points.reshape(-1, 1)),
        'vector': tf_utils.float_list_feature(vector.reshape(-1, 1)),
		'parameters' : tf_utils.float_list_feature(parameters.reshape(-1, 1)),
		'action' : tf_utils.float_list_feature(action.reshape(-1, 1))
	}

    return tf.train.Example(features=tf.train.Features(feature=feature))

BASE_NAME = 'Table_1000_8000_1000_8000_500_500'


def main(args):
    '''CREATE DIR'''
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    exp_dir = Path('./data/')
    exp_dir.mkdir(exist_ok=True)
    exp_dir = exp_dir.joinpath(f'./{BASE_NAME}/')
    exp_dir.mkdir(exist_ok=True)
    exp_dir = exp_dir.joinpath(timestr)
    exp_dir.mkdir(exist_ok=True)

    mu_list = []
    lam_list = []
    yield_stress_list = []
    files = glob.glob(f'/root/ExPCP/policy/pbm/experts/{BASE_NAME}/*/expert*.pickle')

    with tf.io.TFRecordWriter(f'{exp_dir}/train_experts.tfrecord') as train_writer, tf.io.TFRecordWriter(f'{exp_dir}/validation_experts.tfrecord') as validation_writer:
        for path in files:
            with open(path, 'rb') as f: 
                data = pickle.load(f)         
            mu_list.append(data['mu'])
            lam_list.append(data['lam'])
            yield_stress_list.append(data['yield_stress'])
    mu_array = np.array(mu_list)
    lam_array = np.array(lam_list)
    yield_stress_array = np.array(yield_stress_list)
    np.save(f'{exp_dir}/mu.npy', mu_array)
    np.save(f'{exp_dir}/lam.npy', lam_array)
    np.save(f'{exp_dir}/yield_stress.npy', yield_stress_array)


    '''DATA LOADING'''
    file_list = []
    env_count = {}
    files = glob.glob(f'/root/ExPCP/policy/pbm/experts/{BASE_NAME}/*/expert*.pickle')
    with tf.io.TFRecordWriter(f'{exp_dir}/train_experts.tfrecord') as train_writer, tf.io.TFRecordWriter(f'{exp_dir}/validation_experts.tfrecord') as validation_writer:
        for path in files:
            with open(path, 'rb') as f: 
                data = pickle.load(f)           
            env_name = data['env_name']

            if env_name not in env_count:
                env_count[env_name] = 0

            print(path)
            env_count[env_name] += 1
            # target
            action = data['action']
            # points
            plasticine_pc = data['plasticine_pc']
            goal_state = np.load(f"/root/ExPCP/policy/pbm/goal_state/goal_state1/{env_name.split('v')[-1]}/goal_state.npy")
            pc_encode = np.zeros((args.num_plasticine_point + args.num_goal_point, 2))
            pc_encode[:args.num_plasticine_point, 0] = 1
            pc_encode[args.num_plasticine_point:, 1] = 1

            primitive_pc = data['primitive_pc']
            mu = data['mu']
            lam = data['lam']
            # yield_stress = data['yield_stress']
            mu = (mu - np.mean(mu_list)) / np.std(mu_list)
            lam = (lam - np.mean(lam_list)) / np.std(lam_list)
            # yield_stress = (yield_stress - np.mean(yield_stress_list)) / np.std(yield_stress_list)

            for i in range(action.shape[0]):
                plasticine_pc_i = plasticine_pc[i]
                primitive_center_i = primitive_pc[i]
                
                points = sample_pc(plasticine_pc_i, args.num_plasticine_point, goal_state, args.num_goal_point)
                vector = points - primitive_center_i
                vector_encode = np.hstack([vector, pc_encode])
                act = action[i]
                act[1] = 0

                # parameters = np.array([mu, lam, yield_stress])
                parameters = np.array([mu, lam])
                tf_example = create_example(points, vector_encode, parameters, act)
                if random.randint(1, 10) == 1:
                    validation_writer.write(tf_example.SerializeToString())
                else:
                    train_writer.write(tf_example.SerializeToString())    
    with open(f'{exp_dir}/file.txt', 'w') as fp:
        for item in file_list:
            # write each item on a new line
            fp.write("%s\n" % item)
    with open(f'{exp_dir}/env_count.txt', mode="w") as f:
        json.dump(env_count, f, indent=4)
    with open(f'{exp_dir}/args.txt', mode="w") as f:
        json.dump(args.__dict__, f, indent=4)


if __name__ == '__main__':
    args = parse_args()
    main(args)