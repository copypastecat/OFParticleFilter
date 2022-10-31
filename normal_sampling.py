from hashlib import new
from numpy.random import default_rng
import numpy as np
from sampling import generateparticles1, generateparticles2, generateparticles3, generateparticles4
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

M = 10 #number of estimations
N = 10000 #number of generated particles in each estimation

means1 = np.zeros(M)
means2 = np.zeros(M)
means3 = np.zeros(M)
means4 = np.zeros(M)
squares1 = np.zeros(M)
squares2 = np.zeros(M)
squares3 = np.zeros(M)
squares4 = np.zeros(M)
probabilities1 = np.zeros(M)
probabilities2 = np.zeros(M)
probabilities3 = np.zeros(M)
probabilities4 = np.zeros(M)

def generate_new_particles(particles, weights):
    random_generator = default_rng()

    # Resample so that the weights are uniform
    new_particles = np.empty_like(particles)
    new_weights = np.empty_like(weights)

    for index in range(len(particles)):
        new_particles[index] = random_generator.normal(particles[index]**2, 10)
        new_weights[index] = weights[index] / np.exp((-1/2) * (new_particles[index] - particles[index]**2)**2/100)
        new_weights[index] *= 1/20 if new_particles[index] - particles[index]**2 >= -10 and new_particles[index] - particles[index]**2 <= 10 else 0
    
    new_weights /= new_weights.sum()

    return new_particles, new_weights

def cdf(value, particles, weights):
    probability = 0
    for i in range(len(particles)):
        if particles[i] < value:
            probability += weights[i]
    
    return probability

for i in range(M):
    xp1, w1 = generateparticles1(N=N)
    xp2, w2 = generateparticles2(N=N)
    xp3, w3 = generateparticles3(N=N)
    xp4, w4 = generateparticles4(N=N)

    xp1, w1 = generate_new_particles(xp1, w1)
    xp2, w2 = generate_new_particles(xp2, w2)
    xp3, w3 = generate_new_particles(xp3, w3)
    xp4, w4 = generate_new_particles(xp4, w4)

    means1[i] = np.sum(xp1*w1)
    means2[i] = np.sum(xp2*w2)
    means3[i] = np.sum(xp3*w3)
    means4[i] = np.sum(xp4*w4)

    squares1[i] = np.sum(np.square(xp1)*w1)
    squares2[i] = np.sum(np.square(xp2)*w2)
    squares3[i] = np.sum(np.square(xp3)*w3)
    squares4[i] = np.sum(np.square(xp4)*w4)

    probabilities1[i] = cdf(60, xp1, w1)
    probabilities2[i] = cdf(60, xp2, w2)
    probabilities3[i] = cdf(60, xp3, w3)
    probabilities4[i] = cdf(60, xp4, w4)


mean_mean1 = means1.mean()
mean_mean2 = means2.mean()
mean_mean3 = means3.mean()
mean_mean4 = means4.mean()

mean_var1 = means1.var()
mean_var2 = means2.var()
mean_var3 = means3.var()
mean_var4 = means4.var()

square_mean1 = squares1.mean()
square_mean2 = squares2.mean()
square_mean3 = squares3.mean()
square_mean4 = squares4.mean()

square_var1 = squares1.var()
square_var2 = squares2.var()
square_var3 = squares3.var()
square_var4 = squares4.var()

probability_mean1 = probabilities1.mean()
probability_mean2 = probabilities2.mean()
probability_mean3 = probabilities3.mean()
probability_mean4 = probabilities4.mean()

probability_var1 = probabilities1.var()
probability_var2 = probabilities2.var()
probability_var3 = probabilities3.var()
probability_var4 = probabilities4.var()


print("Generator 1: Mean %.4f with estimator variance %.4f, square %.4f with estimator variance %.4f and probability of being lesser than 60 %.4f with estimator variance %.4f" %(mean_mean1, mean_var1, square_mean1, square_var1, probability_mean1, probability_var1))
print("Generator 2: Mean %.4f with estimator variance %.4f, square %.4f with estimator variance %.4f and probability of being lesser than 60 %.4f with estimator variance %.4f" %(mean_mean2, mean_var2, square_mean2, square_var2, probability_mean2, probability_var2))
print("Generator 3: Mean %.4f with estimator variance %.4f, square %.4f with estimator variance %.4f and probability of being lesser than 60 %.4f with estimator variance %.4f" %(mean_mean3, mean_var3, square_mean3, square_var3, probability_mean3, probability_var3))
print("Generator 4: Mean %.4f with estimator variance %.4f, square %.4f with estimator variance %.4f and probability of being lesser than 60 %.4f with estimator variance %.4f" %(mean_mean4, mean_var4, square_mean4, square_var4, probability_mean4, probability_var4))

df1 = pd.DataFrame(np.array([xp1,xp2,xp3,xp4,w1,w2,w3,w4]).T,columns=["x1","x2","x3","x4","w1","w2","w3","w4"])
long_df1 = pd.melt(df1)
print(long_df1.head())
sns.kdeplot(data=df1,x="x1",weights="w1")
sns.kdeplot(data=df1,x="x2",weights="w2")
sns.kdeplot(data=df1,x="x3",weights="w3")
sns.kdeplot(data=df1,x="x4",weights="w4")
plt.legend(["method 1","method 2", "method 3", "method 4"])
plt.xlim([-10,19])
plt.show()