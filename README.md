# Robottle Python Utils 

This contain most of the python work done for the Robottle project, including
- the [Robottle Python package](/robottle_utils)
- a work folder containing itself several attemps / scrips
    - [Bottle Particle filter](/work/bottle_particle_filter) which was implemented but still not integrated
    - [SLAM](/work/slam) which contains developing scripts for LIDAR and MAP analysis
    - [CSI-Camera](/work/csi-camera) which contains code to take picture with the csi-camera of the jetson nano
    
You will find other readme which explains part of this code.    
    
## Robottle Python Package

This package exists to make the [ROS code](https://github.com/arthurBricq/ros_robottle) easier.

We keep the ROS code to do the administratif work, and the python package will do most of the computation.

### Install

Run the line `sudo python3 setup.py install` (sudo is important, ROS uses super user python version, not any other one...)





