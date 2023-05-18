import cv2
import time
import numpy as np
from scipy.spatial.transform import Rotation
import os
import pickle


class Cloth:
    def __init__(self, pcds):
        self.pcds = pcds

    def get_surface_index(self):
        threshold = 0.024
        while not abs(100-(self.pcds[:, 1]>threshold).sum()) < 30:
            if (100-(self.pcds[:, 1]>threshold).sum()) < 0:
                threshold += 0.00001
            else:
                threshold -= 0.00001
        return (self.pcds[:, 1]>threshold).astype(int)

    
    
    
    
    




