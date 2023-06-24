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
    parser.add_argument('--num_plasticine_point', type=int, default=3000, help='Point Number of Plasticine')
    return parser.parse_args()


def create_example(points, parameters):

    feature = {
		'points' : tf_utils.float_list_feature(points.reshape(-1, 1)),
		'parameters' : tf_utils.float_list_feature(parameters.reshape(-1, 1)),
	}

    return tf.train.Example(features=tf.train.Features(feature=feature))

BASE_NAME = 'Move_500_10500_0.2_0.4_200_200'


def main(args):
    '''CREATE DIR'''
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    exp_dir = Path('./data/')
    exp_dir.mkdir(exist_ok=True)
    exp_dir = exp_dir.joinpath(f'./{BASE_NAME}/')
    exp_dir.mkdir(exist_ok=True)
    exp_dir = exp_dir.joinpath(timestr)
    exp_dir.mkdir(exist_ok=True)
    
    E_list = []
    Poisson_list = []
    yield_stress_list = []
    files = glob.glob(f'/root/ExPCP/policy/pbm/experts/{BASE_NAME}/*/expert*.pickle')

    with tf.io.TFRecordWriter(f'{exp_dir}/train_experts.tfrecord') as train_writer, tf.io.TFRecordWriter(f'{exp_dir}/validation_experts.tfrecord') as validation_writer:
        for path in files:
            with open(path, 'rb') as f: 
                data = pickle.load(f)           
            E_list.append(data['E'])
            Poisson_list.append(data['Poisson'])
            yield_stress_list.append(data['yield_stress'])
    E_array = np.array(E_list)
    Poisson_array = np.array(Poisson_list)
    yield_stress_array = np.array(yield_stress_list)
    np.save(f'{exp_dir}/E.npy', E_array)
    np.save(f'{exp_dir}/Poisson.npy', Poisson_array)
    np.save(f'{exp_dir}/yield_stress.npy', yield_stress_array)

    '''DATA LOADING'''
    file_list = []
    env_count = {}
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
            # points
            plasticine_pc = data['plasticine_pc'][-1]
            E = data['E']
            Poisson = data['Poisson']
            # yield_stress = data['yield_stress']
            E_value = (E - np.mean(E_list)) / np.std(E_list)
            Poisson_value = (Poisson - np.mean(Poisson_list)) / np.std(Poisson_list)

            parameters = np.array([E_value, Poisson_value])

            tf_example = create_example(plasticine_pc, parameters)
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