from collections import namedtuple
import random

import numpy as np
import open3d as o3d


Transition = namedtuple('Transition', ('points', 'lang_goal_emb', 'action'))

class ReplayBuffer(object):
    def __init__(self, capacity=1e6):
        self.capacity = capacity
        self.memory = []
        self.sample_idx = 0

    def append(self, *args):
        transition = Transition(*args)
        self.memory.append(transition)

    def sample(self, batch_size):
        data = self.memory[self.sample_idx:self.sample_idx+batch_size]
        self.sample_idx += batch_size
        return data

    def reset(self):
        self.memory = []
    
    def sample_reset(self):
        self.sample_idx = 0

    def length(self):
        return len(self.memory)

    def __len__(self):
        return len(self.memory)


def sample_pc(plasticine_pc, primitive_pc, num_plasticine_point, num_primitive_point):
    plasticine_pcd = o3d.geometry.PointCloud()
    plasticine_pcd.points = o3d.utility.Vector3dVector(plasticine_pc)
    plasticine_pcd = plasticine_pcd.voxel_down_sample(voxel_size=0.005)
    sample_plasticine_pc = np.asarray(plasticine_pcd.points)
    sample_plasticine_pc = sample_plasticine_pc[np.random.choice(sample_plasticine_pc.shape[0], num_plasticine_point), :]
    
    primitive_pcd = o3d.geometry.PointCloud()
    primitive_pcd.points = o3d.utility.Vector3dVector(primitive_pc)
    primitive_pcd = primitive_pcd.voxel_down_sample(voxel_size=0.005)
    sample_primitive_pc = np.asarray(primitive_pcd.points)
    sample_primitive_pc = sample_primitive_pc[np.random.choice(sample_primitive_pc.shape[0], num_primitive_point), :]
    
    sample_pc = np.concatenate([sample_plasticine_pc, sample_primitive_pc])
    
    return sample_pc