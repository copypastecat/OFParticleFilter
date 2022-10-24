from statistics import mean
from sampling import *
import numpy as np
import seaborn as sns

M = 10 #number of estimations
N = 1000 #number of generated particles in each estimation

means1 = np.zeros(M)
means2 = np.zeros(M)
means3 = np.zeros(M)
means4 = np.zeros(M)
vars1 = np.zeros(M)
vars2 = np.zeros(M)
vars3 = np.zeros(M)
vars4 = np.zeros(M)

for i in range(M):
    xp1, w1 = generateparticles1(N=N)
    xp2, w2 = generateparticles2(N=N)
    xp3, w3 = generateparticles3(N=N)
    xp4, w4 = generateparticles4(N=N)

    #estimate the mean using Monte-Carlo
    mean1 = np.sum(xp1*w1)
    mean2 = np.sum(xp2*w2)
    mean3 = np.sum(xp3*w3)
    mean4 = np.sum(xp4*w4)

    var1 = np.sum(((xp1-mean1)**2)*w1)
    var2 = np.sum(((xp2-mean2)**2)*w2)
    var3 = np.sum(((xp3-mean3)**2)*w3)
    var4 = np.sum(((xp4-mean4)**2)*w4)
    

    means1[i] = mean1
    means2[i] = mean2
    means3[i] = mean3
    means4[i] = mean4

    vars1[i] = var1
    vars2[i] = var2
    vars3[i] = var3
    vars4[i] = var4


mean_mean1 = means1.mean()
mean_mean2 = means2.mean()
mean_mean3 = means3.mean()
mean_mean4 = means4.mean()

mean_var1 = means1.var()
mean_var2 = means2.var()
mean_var3 = means3.var()
mean_var4 = means4.var()

var_mean1 =vars1.mean()
var_mean2 =vars2.mean()
var_mean3 =vars3.mean()
var_mean4 =vars4.mean()

var_var1 = vars1.var()
var_var2 = vars2.var()
var_var3 = vars3.var()
var_var4 = vars4.var()


print("Generator 1: Mean %.4f with estimator variance %.4f and variance %.4f with estimator variance %.4f" %(mean_mean1, mean_var1, var_mean1, var_var1))
print("Generator 2: Mean %.4f with estimator variance %.4f and variance %.4f with estimator variance %.4f" %(mean_mean2, mean_var2, var_mean2, var_var2))
print("Generator 3: Mean %.4f with estimator variance %.4f and variance %.4f with estimator variance %.4f" %(mean_mean3, mean_var3, var_mean3, var_var3))
print("Generator 4: Mean %.4f with estimator variance %.4f and variance %.4f with estimator variance %.4f" %(mean_mean4, mean_var4, var_mean4, var_var4))

sns.kdeplot(xp1,weights=np.array(w1))