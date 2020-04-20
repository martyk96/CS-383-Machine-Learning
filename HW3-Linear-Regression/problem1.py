#!/usr/bin/env python3
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x1 = np.arange(0, 2, 0.05)
x2 = np.arange(0, 2, 0.05)

x1, x2 = np.meshgrid(x1, x2)
J = np.square(x1+x2-2)

ax.plot_surface(x1,x2,J,cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

ax.set_title('J minimized along x2=2-x1')
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('J')
plt.show()