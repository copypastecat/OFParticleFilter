from measurement_update import measurement_update
import numpy as np
import matplotlib.pyplot as plt

xp = np.linspace(0,100,200)
wp = np.ones(len(xp))*(1/len(xp))
y = 60

#plt.stem(xp,measurement_update(xp,wp,y))
#plt.show()

wp_y = measurement_update(xp,wp,60)
mean_estimate = np.sum(xp*wp_y)

wp_y2 = measurement_update(xp,wp,-20)
wp_y2[xp > 50] = 0
p50 = np.sum(wp_y2)
print("E[Z|Y=60] = %.4f" %mean_estimate)
print("P[Z < 50|Y=-20] = %.4f" %p50)

