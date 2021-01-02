from scipy import interpolate as interp
import matplotlib.pyplot as plt
import numpy as np
import math
from mpl_toolkits.mplot3d import Axes3D



def getSum(x,y,n,m):
    return sum(x**n * y**m)

def getSumZ(x,y,z,n,m,l):
    return sum(x**n * y**m * z**l)

def getAngle(P,x,y):
    return P[0]*x**3 + P[1]*x**2*y + P[2]*x*y**2 + P[3]*y**3 + P[4]*x**2 + P[5]*x*y + P[6]*y**2 + P[7]*x + P[8]*y + P[9]

# create detections list

f = open('_detection.txt')
det = []
detections = []

for line in f.readlines():
    for el in line.split(', '):
        if el[len(el)-1] == '\n':
            el.replace('\n','')
        det.append(float(el))

for i in range(0,int(len(det)/4)):
    detections.append((det[4*i],det[4*i+1],det[4*i+2],det[4*i+3]))

detections = np.array(detections)
f.close()


# create positions list

f = open('positions.txt')
pos = []
positions = []

for line in f.readlines():
    for el in line.split(' '):
        el.replace('\n','')
        pos.append(float(el))

for i in range(0,int(len(pos)/2)):
    positions.append((pos[2*i],pos[2*i+1]))

positions = np.array(positions)
angles = np.tan(positions[:,1]/positions[:,0])*180/np.pi
f.close()


# INTERPOLATION

# dist =  interp.interp2d(np.transpose(detections[:,0]), np.transpose(detections[:,1]), np.transpose(positions[:,0]), kind = 'linear')
#
# # plot the interpolated data
# x = np.arange(0, 1280, 1e-1)
# y = np.arange(0, 720, 1e-1)
# z = dist(x, y)
# x,y = np.meshgrid(x, y)
#
#
# fig = plt.figure()
# ax = fig.gca(projection='3d')
#
# surf = ax.plot_surface(x, y, z, linewidth=0, antialiased=False)
# plt.show()



# LEAST SQUARES PARABOLIC FUNCTION

x = detections[:,0] # horizontal axis
y = detections[:,1] # vertical axis
z = angles

x6 = getSum(x,y,6,0)
x5y1 = getSum(x,y,5,1)
x4y2 = getSum(x,y,4,2)
x3y3 = getSum(x,y,3,3)
x2y4 = getSum(x,y,2,4)
x1y5 = getSum(x,y,1,5)
y6 = getSum(x,y,0,6)
x5 = getSum(x,y,5,0)
x4y1 = getSum(x,y,4,1)
x3y2 = getSum(x,y,3,2)
x2y3 = getSum(x,y,2,3)
x1y4 = getSum(x,y,1,4)
y5 = getSum(x,y,0,5)
x4 = getSum(x,y,4,0)
x3y1 = getSum(x,y,3,1)
x2y2 = getSum(x,y,2,2)
x1y3 = getSum(x,y,1,3)
y4 = getSum(x,y,0,4)
x3 = getSum(x,y,3,0)
x2y1 = getSum(x,y,2,1)
x1y2 = getSum(x,y,1,2)
y3 = getSum(x,y,0,3)
x2 = getSum(x,y,2,0)
x1y1 = getSum(x,y,1,1)
y2 = getSum(x,y,0,2)
x1 = getSum(x,y,1,0)
y1 = getSum(x,y,0,1)
one = getSum(x,y,0,0)

A = np.array([[x6,   x5y1, x4y2, x3y3, x5,   x4y1, x3y2, x4,   x3y1, x3],
              [x5y1, x4y2, x3y3, x2y4, x4y1, x3y2, x2y3, x3y1, x2y2, x2y1],
              [x4y2, x3y3, x2y4, x1y5, x3y2, x2y3, x1y4, x2y2, x1y3, x1y2],
              [x3y3, x2y4, x1y5, y6,   x2y3, x1y4, y5,   x1y3, y4,   y3],
              [x5,   x4y1, x3y2, x2y3, x4,   x3y1, x2y2, x3,   x2y1, x2],
              [x4y1, x3y2, x2y3, x1y4, x3y1, x2y2, x1y3, x2y1, x1y2, x1y1],
              [x3y2, x2y3, x1y4, y5,   x2y2, x1y3, y4,   x1y2, y3,   y2],
              [x4,   x3y1, x2y2, x1y3, x3,   x2y1, x1y2, x2,   x1y1, x1],
              [x3y1, x2y2, x1y3, y4,   x2y1, x1y2, y3,   x1y1, y2,   y1],
              [x3,   x2y1, x1y2, y3,   x2,   x1y1, y2,   x1,   y1,   one]])

zx3 = getSumZ(x,y,z,3,0,1)
zx2y1 = getSumZ(x,y,z,2,1,1)
zx1y2 = getSumZ(x,y,z,1,2,1)
zy3 = getSumZ(x,y,z,0,3,1)
zx2 = getSumZ(x,y,z,2,0,1)
zxy = getSumZ(x,y,z,1,1,1)
zy2 = getSumZ(x,y,z,0,2,1)
zx = getSumZ(x,y,z,1,0,1)
zy = getSumZ(x,y,z,0,1,1)
z1 = getSumZ(x,y,z,0,0,1)

b = np.array([[zx3],[zx2y1],[zx1y2],[zy3],[zx2],[zxy],[zy2],[zx],[zy],[z1]])

P = np.linalg.solve(A,b)

print('the coefficient of the funtion are:',P)


x = np.arange(0, 1280, 1e-1)
y = np.arange(0, 720, 1e-1)
x,y = np.meshgrid(x, y)
z = getAngle(P, x, y)

fig = plt.figure()
ax = fig.gca(projection='3d')

surf = ax.plot_surface(x, y, z, linewidth=0, antialiased=False)
plt.show()
