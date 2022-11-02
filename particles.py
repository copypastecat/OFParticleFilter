from statistics import mean
from sampling import *
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

N = 100 #number of generated particles in each estimation

xp1, w1 = generateparticles1(N=N)
xp2, w2 = generateparticles2(N=N)
xp3, w3 = generateparticles3(N=N)
xp4, w4 = generateparticles4(N=N)

plt.figure(1)
plt.stem(xp1, w1)
plt.title('Method 1')
plt.xlabel('X')
plt.ylabel('w')
plt.figure(2)
plt.stem(xp2, w2)
plt.title('Method 2')
plt.xlabel('X')
plt.ylabel('w')
plt.figure(3)
plt.stem(xp3, w3)
plt.title('Method 3')
plt.xlabel('X')
plt.ylabel('w')
plt.figure(4)
plt.stem(xp4, w4)
plt.title('Method 4')
plt.xlabel('X')
plt.ylabel('w')
plt.show()