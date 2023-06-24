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

BASE_NAME = 'Torus_500_10500_0.2_0.4_200_200'
E_BOTTOM, E_UPPER = 3276.12, 8000.31
POISSON_BOTTOM, POISSON_UPPER = 0.346, 0.36


def create_example(goal_point, parameters, release_point):

    feature = {
		'goal_point' : tf_utils.float_list_feature(goal_point.reshape(-1, 1)),
		'parameters' : tf_utils.float_list_feature(parameters.reshape(-1, 1)),
		'release_point' : tf_utils.float_list_feature(release_point.reshape(-1, 1))
	}

    return tf.train.Example(features=tf.train.Features(feature=feature))


def main():
    '''CREATE DIR'''
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    exp_dir = Path('./data/')
    exp_dir.mkdir(exist_ok=True)
    exp_dir = exp_dir.joinpath(f'./Full_para_ood/')
    exp_dir.mkdir(exist_ok=True)
    exp_dir = exp_dir.joinpath(timestr)
    exp_dir.mkdir(exist_ok=True)
    
    E_list = []
    Poisson_list = []
    yield_stress_list = []
    goal_point_list = []
    
    files = glob.glob(f'/root/ExPCP/policy/pbm/experts/{BASE_NAME}/*/expert*.pickle')
    for path in files:
        with open(path, 'rb') as f: 
            data = pickle.load(f)
        E = data['E']
        Poisson = data['Poisson']
        E_list.append(E)
        Poisson_list.append(Poisson)
        yield_stress_list.append(data['yield_stress'])
        goal_point_list.append(data['max_x'])
    E_array = np.array(E_list)
    Poisson_array = np.array(Poisson_list)
    yield_stress_array = np.array(yield_stress_list)
    goal_point_array = np.array(goal_point_list)
    np.save(f'{exp_dir}/E.npy', E_array)
    np.save(f'{exp_dir}/Poisson.npy', Poisson_array)
    np.save(f'{exp_dir}/yield_stress.npy', yield_stress_array)
    np.save(f'{exp_dir}/goal_point.npy', goal_point_array)

    '''DATA LOADING'''
    file_list = []
    env_count = {}
    count = 0
    with tf.io.TFRecordWriter(f'{exp_dir}/train_experts.tfrecord') as train_writer, tf.io.TFRecordWriter(f'{exp_dir}/validation_experts.tfrecord') as validation_writer:
        for path in files:
            with open(path, 'rb') as f: 
                data = pickle.load(f)           
            env_name = data['env_name']

            if env_name not in env_count:
                env_count[env_name] = 0
            
            E = data['E']
            Poisson = data['Poisson']
            yield_stress = data['yield_stress']
                
            if E >= E_BOTTOM and E <= E_UPPER and Poisson >= POISSON_BOTTOM and Poisson <= POISSON_UPPER:
                count+=1
                print(path)
                env_count[env_name] += 1

                release_point = data['release_point'][:2]
                goal_point = data['max_x']

                E = (E - np.mean(E_list)) / np.std(E_list)
                Poisson = (Poisson - np.mean(Poisson_list)) / np.std(Poisson_list)
                yield_stress = (yield_stress - np.mean(yield_stress_list)) / np.std(yield_stress_list)

                parameters = np.array([E, Poisson])
                tf_example = create_example(goal_point, parameters, release_point)
                if random.randint(1, 10) == 1:
                    validation_writer.write(tf_example.SerializeToString())
                else:
                    train_writer.write(tf_example.SerializeToString())    
    with open(f'{exp_dir}/env_count.txt', mode="w") as f:
        json.dump(env_count, f, indent=4)
    print(count)


if __name__ == '__main__':
    main()