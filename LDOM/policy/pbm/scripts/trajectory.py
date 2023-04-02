import numpy as np
import matplotlib.pyplot as plt

a = np.load('./output/Move-v1/output13002511/13002511.npy')

fig = plt.figure()
X1 = 0.63
Y1 = 0.3
X2 = 0.37
Y2 = 0.7

frames = []
for act in a:
    # actionの可視化
    act /= 100
    plt.quiver(X1, Y1, act[0], act[2], angles='xy',scale_units='xy',scale=1, color='g')
    plt.quiver(X2, Y2, act[3], act[5], angles='xy',scale_units='xy',scale=1, color='r')
    X1 += act[0]
    Y1 += act[2]
    X2 += act[3]
    Y2 += act[5]
