from scipy import interpolate as interp
import matplotlib.pyplot as plt
import numpy as np
import math
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import PolynomialFeatures

%matplotlib qt



def getAngle(P,x,y):
    """
    Given the weigths P of the fit, returns the prediced angles for the position
    (x,y) of the center of the bounding box
    """
    return P[6]*x**3 + P[7]*x**2*y + P[8]*x*y**2 + P[9]*y**3 + P[3]*x**2 + P[4]*x*y + P[5]*y**2 + P[1]*x + P[2]*y + P[0]
    # return P[1]*x + P[2]*y + P[0]


#%% read the training dataset and compute angles

is_fake = True

f = open('detections_faked.txt' if is_fake else 'detections_real.txt')
det = []
detections = []

for line in f.readlines():
    splited = line.split(' ' if is_fake else ', ')
    if '\n' in splited:
        splited.remove('\n')
    det.append(np.array(splited).astype(float))
det = np.array(det)

detections = det[:, :2]
positions = det[:, -2:]
angles = np.tan(positions[:,1] / positions[:, 0]) * 180 / np.pi

#%% 

poly = PolynomialFeatures(degree = 3)
qs = poly.fit_transform(detections).T
z = angles
A = (qs @ qs.T)

n = qs.shape[1] # 59
A = np.zeros((qs.shape[0], qs.shape[0])) # shape = (10, 10)
b = np.zeros((qs.shape[0], 1)) # shape = (1, 10)

for i in range(n):
    qi = qs[:,i].reshape(-1, 1)
    ai = qi @ qi.T # shape = (10, 10)
    A += ai
    b += angles[i] * qi

#%% 
P = np.linalg.solve(A,b)

x = np.arange(0, 1280, 1e-1)
y = np.arange(0, 720, 1e-1)
x,y = np.meshgrid(x, y)
z = getAngle(P, x, y)

fig = plt.figure()
ax = fig.gca(projection='3d')

surf = ax.plot_surface(x, y, z, linewidth=0, antialiased=False)
plt.show()
