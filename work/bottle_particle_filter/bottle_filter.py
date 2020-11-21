import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import imageio

# TODO
# - add noise to sampling
# - add random new particles to sampling (and remove others)
# - function to determine if particle is in robot range of view
# - add real simulation with several timesteps

#%% Class definition

sigma = 1
sigma_noise = 0.5
range_of_vision = np.pi / 3 # [degs] each side of the robot

def is_in_range(robot_pose, particles):
    """
    Returns the list of 'particles' that are in front of the robot
    - particles is a (k,2) np.array 

    First it transform coordinates in the robot frame
    Then it compare the tan to a threshold value
    """
    theta = robot_pose[2]
    threshold = np.tan(range_of_vision)
    # rotation matrix to be in robot coordinates
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    particles_r = particles - robot_pose[:2].reshape(1,2)
    particles_r = particles_r @ R
    tans = particles_r[:, 1] / particles_r[:, 0]
    return (particles_r[:,0] > 0) * (np.abs(tans) < threshold)

class BottleFilter:
    def __init__(self, k = 10, W = 25, gif = False):
        self.W = W
        self.particles = np.random.uniform(size=(k,2), low = 0, high = W)
        # for drawing GIF images
        self.gif = gif
        if gif:
            self.images = []

    def measurements_update(self, measurements, robot_pose):
        """
        This function updates the particles from the given measurements and the robot position.
        - measurements are the estimate of the bottle position

        So far, it assumes position estimates have same variance.
        """
        # get indices of particle to be resampled (in front of the robot)
        idcs = np.where(is_in_range(robot_pose, self.particles))[0]
        ps = self.particles[idcs]
        # shape of 'delta' = ( M, k_filtered, 2 )
        deltas = measurements.reshape(len(measurements), 1, 2) - ps.reshape(1, len(ps), 2)
        # compute distances betweeen particles and the measurements
        d2 = (deltas * deltas).sum(axis=2)
        # compute the new weights for each particles
        omegas = np.exp(-0.5 * d2 / sigma / sigma).sum(axis=0)
        # as it is a distribution, it must be normalized
        omegas = omegas / np.sum(omegas)
        # resample based on the new distribution
        self.particles[idcs] = ps[np.random.choice(len(ps), p = omegas, size = len(idcs))] + np.random.normal(loc=0, scale=sigma_noise, size=ps.shape)


    def plot_world(self, robot_pose, measurements = [], bottles_positions = []):
        fig = plt.figure()
        # plot the particles
        plt.plot(self.particles[:,0], self.particles[:,1], 'x')
        # plot the robot and the range of vision
        plt.plot(robot_pose[0], robot_pose[1], 'o')
        theta = robot_pose[2]
        plt.arrow(robot_pose[0], robot_pose[1], 2*np.cos(theta), 2*np.sin(theta), shape='full', head_width=1)
        plt.arrow(robot_pose[0], robot_pose[1],
                6*np.cos(theta+range_of_vision), 6*np.sin(theta+range_of_vision), color = 'orange')
        plt.arrow(robot_pose[0], robot_pose[1],
                6*np.cos(theta-range_of_vision), 6*np.sin(theta-range_of_vision), color = 'orange')
        # plot the measurements and the real bottle position
        if len(measurements):
            plt.plot(measurements[:,0], measurements[:,1], 'or')
        if len(bottles_positions):
            plt.plot(bottles_positions[:,0], bottles_positions[:,1], 'Dy')
        plt.xlim([0, self.W])
        plt.ylim([0, self.W])

        # for creating a gif
        if self.gif:
            fig.canvas.draw()       # draw the canvas, cache the renderer
            image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
            image  = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            self.images.append(image)

    def save_gif(self, name):
        # kwargs_write = {'fps':1.0, 'quantizer':'nq'}
        imageio.mimsave(name, self.images, fps=4)



#%% Simulation environment

# list of successive position of the robot
robot_positions = np.array([
    [5,5,0],
    [6,5,0],
    [7,5,0],
    [8,5,0],
    [9,5,0],
    [9,6,0],
    ])

# list of the TRUE positions of bottles
bottles_positions = np.array([
    [20,5],
    [10,10],
    [20,15],
    ])


def get_measurements(robot_pose, bottles_positions):
    """
    Given the robot position and the true positions of the bottles, 
    it will simulate some measurements with guassian error added to this
    """
    bottles_in_range = bottles_positions[is_in_range(robot_pose, bottles_positions)]
    measurements = bottles_in_range + np.random.normal(loc=0, scale = sigma, size=bottles_in_range.shape)
    return measurements

# Successively move the robot and update the map ! 
bottle_filter = BottleFilter(k = 500, W = 25, gif=True)
bottle_filter.plot_world(robot_pose = robot_positions[0], measurements=[], bottles_positions=bottles_positions)
for robot_pose in robot_positions:
    print("Robot pos: {}".format(robot_pose))
    measurements = get_measurements(robot_pose, bottles_positions)
    bottle_filter.measurements_update(robot_pose = robot_pose, measurements = measurements)
    bottle_filter.plot_world(robot_pose = robot_pose, measurements=measurements, bottles_positions=bottles_positions)
bottle_filter.save_gif("./animation.gif")
















