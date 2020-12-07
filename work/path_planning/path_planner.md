# Path Planning 

We want to use RRT star algorithm for optimal path planning with a sampling based method. The code was found online and is of great quality


## Understanding the RRT Star Algorithm 

### Some definitions

- A tree
    - A vertex / node: a point in the configuration space
    - An edge: a line between 2 points


### Parameters of the code
- *goal_sample_rate* (int [0 100]): whenever a new node is sampled, it can be either drawn randomly, or it can be the end position point. The higher it is, the more often the 'goal position' will be sampled. 
- *expand_dis*: it is the **steering factor** that is applied between the randomly select point and its nearest neighbord. The higher it is, the faster the graph is going to expend
- *path_resolution*: ?

## Interfacing the API with our Code

**What the API expects**
- all positions are in a 2D grid (x,y) 
- the 'check_collisition' method is the only thing which must be interfaced. It simply must retuns 'true' if a node is inside an obstacle or not. 










