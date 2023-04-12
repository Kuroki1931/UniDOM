import cv2
import time
import numpy as np
from scipy.spatial.transform import Rotation
import os
import pickle


class Cloth:
    def __init__(self):
        self.base = np.load('/root/real2sim/real2sim/points/initial_pcds.npy')
        self.obj_init_pos = None
        self.obj_width = None
        self.obj_pcds = None
        
    def get_hand_position(self):
        return [str(tuple(np.array([0.5, 0.1, 0.5])))]

    def get_obj_particle(self, n_particles=3000):
        self.obj_init_pos = self.base.mean(axis=0)
        self.obj_init_pos[1] = 0.01
        self.obj_width = self.base.max(axis=0) - self.base.min(axis=0)
        self.obj_width[1] = 0.02
        self.obj_pcds = (np.random.random((n_particles, 3)) * 2 - 1) * (0.5 * self.obj_width) + np.array(self.obj_init_pos)
        return self.obj_pcds
    
    def get_surface_index(self):
        surface_index = []
        threshold = 0.01
        width = (0.3/2, 0.07/2, 0.1/2)
        init_pos = (0.5, 0.05, 0.5)
        for pcd in self.obj_pcds:
            if (init_pos[0] - width[0] + threshold >= pcd[0]) | (init_pos[0] + width[0] - threshold <= pcd[0]):
                surface_index.append(1)
            # don't use bottom side
            elif (init_pos[1] + width[1] - threshold <= pcd[1]):
                surface_index.append(1)
            elif (init_pos[2] - width[2] + threshold >= pcd[2]) | (init_pos[2] + width[2] - threshold <= pcd[2]):
                surface_index.append(1)
            else:
                surface_index.append(0)
        return surface_index

    
    
    
    
    




