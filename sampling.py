# Example functions to generate particles from a specific distribution. (Which?)
# FEM3200/EQ2801 Optimal Filtering. KTH Royal Institute of Technology
# October 2022, Mats Bengtsson <matben@kth.se>

import numpy as np
from scipy.stats import norm
xmax = 9

def generateparticles1(N):
    """Return particles xp and weights w as NumPy vectors"""
    xp = xmax*np.sqrt(np.random.default_rng().uniform(0,1,N))
    w = np.full(N,1/N)
    return xp, w


def generateparticles2(N):
    """Return particles xp and weights w as NumPy vectors"""
    xp = xmax*np.random.default_rng().uniform(0,1,N)
    w = xp/xp.sum()
    return xp, w


def generateparticles3(N):
    """Return particles xp and weights w as NumPy vectors"""
    my_lambda=1/5
    xp = -np.log(np.random.default_rng().uniform(0,1,N))/my_lambda
    w_unnormalized=xp*np.exp(my_lambda*xp)/my_lambda
    w_unnormalized[xp>xmax]=0
    w = w_unnormalized/w_unnormalized.sum()
    return xp, w


def generateparticles4(N):
    """Return particles xp and weights w as NumPy vectors"""
    sigma = 7
    M=1.1*2/xmax/norm.pdf(xmax,0,sigma)
    xp=np.random.default_rng().normal(0,sigma,N)
    u=np.random.default_rng().uniform(0,1,N)
    rejected = (xp<0) | (xp>xmax) | (u >= (2*xp/xmax**2 / norm.pdf(xp,0,sigma)/M))
    while np.any(rejected):
        Nrej=rejected.sum()
        xp_new=np.random.default_rng().normal(0,sigma,Nrej)
        u=np.random.default_rng().uniform(0,1,Nrej)
        xp[rejected]=xp_new;
        rejected_new = (xp_new<0) | (xp_new>xmax) | (u >= (2*xp_new/xmax**2 / 
norm.pdf(xp_new,0,sigma)/M))
        rejected[rejected]=rejected_new
    w = np.full(N,1/N)
    return xp, w
