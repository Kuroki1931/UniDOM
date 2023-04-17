import cv2
import time
import numpy as np
from scipy.spatial.transform import Rotation
import os
import pickle

OFFSET = 0.01
CLOTH_THICKNESS = 0.002


class Cloth:
    def __init__(self):
        self.base = np.load('/root/real2sim/real2sim/points/initial_pcds.npy', allow_pickle=True)
        self.obj_init_pos = None
        self.obj_width = None
        self.obj_pcds = None
        
    def get_hand_position(self):
        corner_position = self.obj_pcds.min(axis=0)
        return [str(tuple(np.array([corner_position[0] + 0.05, OFFSET + CLOTH_THICKNESS/2 + 0.01, corner_position[2] + 0.05])))]

    def get_obj_particle(self, n_particles=3000):
        self.obj_init_pos = self.base.mean(axis=0)
        self.obj_init_pos[1] = OFFSET
        self.obj_width = self.base.max(axis=0) - self.base.min(axis=0)
        self.obj_width[1] = CLOTH_THICKNESS
        self.obj_pcds = (np.random.random((n_particles, 3)) * 2 - 1) * (0.5 * self.obj_width) + np.array(self.obj_init_pos)
        return self.obj_pcds
    
    def get_surface_index(self):
        threshold = OFFSET
        while not abs(self.base.shape[0]-(self.obj_pcds[:, 1]>threshold).sum()) < 30:
            if (self.base.shape[0]-(self.obj_pcds[:, 1]>threshold).sum()) < 0:
                threshold += 0.00001
            else:
                threshold -= 0.00001
        return (self.obj_pcds[:, 1]>threshold).astype(int)

    
    
    
    
    




