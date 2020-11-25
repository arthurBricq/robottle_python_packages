# Map Analysis

We want to analyse the output of the SLAM in order to **take control decisions**. 

## Task 1: find the zones in the map

### a) find the rotated rectangle that fits the map 

Here are the steps to do that: 
- binary threshold
- median filter
- contour detection
- contour merging
- min area fits for a rectangle around resulting unique contour

It is expressed in the function 'get_bounding_rect' from robottle_utils.map_utils

Todo
- make it robust 
- detect when it doesn't work 

### b) extract the zone from the rectangle 

