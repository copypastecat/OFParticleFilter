import numpy as np

def p_W(w: float):
    if(abs(w)<=1):
        return abs(w)
    else:
        return 0


def measurement_update(xp: np.array,wp: np.array,y:float):
    particles = np.array([xp,wp])
    if(y != 0):
       ratio = y*np.ones(len(xp))/xp
       l = np.zeros(len(ratio))
       for i,r in enumerate(ratio): 
           l[i] = p_W(r)
       wp_y = wp*l
       wp_y = wp_y/np.sum(wp_y)
    else:
        #if y=0, z has to be 0 since w is 0 with probability 0 
        wp_y = np.zeros(len(wp))
        wp_y[particles[0]==0] = 1/len(particles[0][particles[0]==0])
    return wp_y