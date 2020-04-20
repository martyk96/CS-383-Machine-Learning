#!/usr/bin/env python3
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

# setup
x = np.array([[0],[0]])
J = 0
it =0
x1 = [0]
x2 = [0]
J = [(x1[0]+x2[0]-2)**2]
n = 0.1
dJdx = lambda x1,x2: 2*(x1+x2-2)

# run until J converges
while it==0 or abs(J[it] - J[it-1]) > 2**-32:
    x1.append(x1[it] - n * dJdx(x1[it],x2[it]))
    x2.append(x2[it] - n * dJdx(x1[it],x2[it]))

    it +=1
    J.append((x1[it]+x2[it]-2)**2)
    

# plotting
fig = plt.figure()
fig.suptitle('x1, x2, J vs Iterations for Gradient Descent Method',fontweight='bold')
ax1 = fig.add_subplot(131)
ax1.plot(list(range(it+1)),x1,list(range(it+1)),x1,'o',ms=3)
ax1.set_xlabel('iterations')
ax1.set_ylabel('x1')

ax2 = fig.add_subplot(132)
ax2.plot(list(range(it+1)),x2,list(range(it+1)),x2,'o',ms=3)
ax2.set_xlabel('iterations')
ax2.set_ylabel('x2')

ax3 = fig.add_subplot(133)
ax3.plot(list(range(it+1)),J,list(range(it+1)),J,'o',ms=3)
ax3.set_xlabel('iterations')
ax3.set_ylabel('J')

plt.show()

