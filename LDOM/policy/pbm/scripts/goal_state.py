import numpy as np
import matplotlib.pyplot as plt

b = np.load('./plb/envs/assets/Move3D-v1.npy')
for i in range(b.shape[0]):
    b[i] = 0
for i in range(18, 49):
    b[i][:6, 29:36] = 0.0005
np.save('./plb/envs/assets/Move3D-v1.npy', b)

