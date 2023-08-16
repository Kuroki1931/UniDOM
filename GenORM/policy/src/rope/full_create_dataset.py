import os
import sys
import random
import datetime

import glob
import pickle
import json

sys.path.insert(0, './')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

import numpy as np
import tensorflow as tf
from pathlib import Path

from util import tf_utils

BASE_NAME = 'Rope_500_10500_0.2_0.4_200_200'

def create_example(height, parameters, action):

    feature = {
        'parameters' : tf_utils.float_list_feature(parameters.flatten()),
		'height' : tf_utils.float_list_feature(height.flatten()),
		'action' : tf_utils.float_list_feature(action.flatten())
	}

    return tf.train.Example(features=tf.train.Features(feature=feature))


def main():
    '''CREATE DIR'''
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    exp_dir = Path('./data/')
    exp_dir.mkdir(exist_ok=True)
    exp_dir = exp_dir.joinpath(f'./{BASE_NAME}/')
    exp_dir.mkdir(exist_ok=True)
    exp_dir = exp_dir.joinpath(f'./full/')
    exp_dir.mkdir(exist_ok=True)
    exp_dir = exp_dir.joinpath(timestr)
    exp_dir.mkdir(exist_ok=True)
    
    E_list = []
    Poisson_list = []
    yield_stress_list = []
    height_list = []
    
    files = glob.glob(f'/root/GenORM/policy/pbm/experts/{BASE_NAME}/*/expert*.pickle')

  
    for path in files:
        with open(path, 'rb') as f: 
            data = pickle.load(f)           
        E_list.append(data['E'])
        Poisson_list.append(data['Poisson'])
        yield_stress_list.append(data['yield_stress'])
        height_list.append([data['height']])
    E_array = np.array(E_list)
    Poisson_array = np.array(Poisson_list)
    yield_stress_array = np.array(yield_stress_list)
    height_array = np.array(height_list)
    np.save(f'{exp_dir}/E.npy', E_array)
    np.save(f'{exp_dir}/Poisson.npy', Poisson_array)
    np.save(f'{exp_dir}/yield_stress.npy', yield_stress_array)
    np.save(f'{exp_dir}/height_point.npy', height_array)

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

            action = np.array([data['max_acxel']])
            height = np.array([data['height']]) / 2

            E = data['E']
            Poisson = data['Poisson']
            yield_stress = data['yield_stress']
            E = (E - np.mean(E_list)) / np.std(E_list)
            Poisson = (Poisson - np.mean(Poisson_list)) / np.std(Poisson_list)
            yield_stress = (yield_stress - np.mean(yield_stress_list)) / np.std(yield_stress_list)
            
            parameters = np.array([E, Poisson])
            tf_example = create_example(height, parameters, action)
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

    '''DATA LOADING'''
    file_list = []
    env_count = {}
    with tf.io.TFRecordWriter(f'{exp_dir}/train_real_experts.tfrecord') as train_writer, tf.io.TFRecordWriter(f'{exp_dir}/validation_real_experts.tfrecord') as validation_writer:
        # predifined real data
        # soft cloth
        for _ in range(1, 100):
            height = np.array([0.16])
            E = 584.21
            Poisson = 0.25
            yield_stress = 200
            E = (E - np.mean(E_list)) / np.std(E_list)
            Poisson = (Poisson - np.mean(Poisson_list)) / np.std(Poisson_list)
            yield_stress = (yield_stress - np.mean(yield_stress_list)) / np.std(yield_stress_list)
            action = 0.65 + np.random.normal(0, 0.05)
            action = np.array([action])
            
            parameters = np.array([E, Poisson])
            tf_example = create_example(height, parameters, action)
            
            if random.randint(1, 10) == 1:
                validation_writer.write(tf_example.SerializeToString())
            else:
                train_writer.write(tf_example.SerializeToString()) 
        
        # Rubber
        for _ in range(1, 100):
            height = np.array([0.16])
            E = 6828.33
            Poisson = 0.26
            yield_stress = 200
            E = (E - np.mean(E_list)) / np.std(E_list)
            Poisson = (Poisson - np.mean(Poisson_list)) / np.std(Poisson_list)
            yield_stress = (yield_stress - np.mean(yield_stress_list)) / np.std(yield_stress_list)
            action = 0.1 + np.random.normal(0, 0.05)
            action = np.array([action])
            
            parameters = np.array([E, Poisson])
            tf_example = create_example(height, parameters, action)
            
            if random.randint(1, 10) == 1:
                validation_writer.write(tf_example.SerializeToString())
            else:
                train_writer.write(tf_example.SerializeToString()) 
            


if __name__ == '__main__':
    main()