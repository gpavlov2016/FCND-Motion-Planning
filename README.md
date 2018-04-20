# FCND - 3D Motion Planning
![Quad Image](./misc/enroute.png)

This project is a continuation of the Backyard Flyer project where that executes a simple square shaped flight path. This project demonstrated planning and executing a path through an urban environment.

## To run this project on your local machine, follow these instructions:
### Step 1: Download the Simulator
This is a new simulator environment!  

Download the Motion-Planning simulator for this project that's appropriate for your operating system from the [simulator releases respository](https://github.com/udacity/FCND-Simulator-Releases/releases).

### Step 2: Set up your Python Environment
If you haven't already, set up your Python environment and get all the relevant packages installed using Anaconda following instructions in [this repository](https://github.com/udacity/FCND-Term1-Starter-Kit)

### Step 3: Clone this Repository
```
sh
git clone https://github.com/gpavlov2016/FCND-Motion-Planning
```

### Step 4: Test setup
```
sh
source activate fcnd # if you haven't already sourced your Python environment, do so now.
python backyard_flyer_solution.py
```
The quad should take off, fly a square pattern and land, just as in the previous project. If everything functions as expected then you are ready to start work on this project. 
```
sh
source activate fcnd # if you haven't already sourced your Python environment, do so now.
python motion_planning.py
```

## Implementation Details

### Functionality of motion_planning.py and backyard_flyer_solution.py:
Both scripts have similar structure implementing an event driven drone flight control module. The code uses a library called `udacidrone` that publishes events to registered functions. There are three types of callbacks - local_position_callback, velocity_callback and state_callback. In both cases the user class implements a state machine with the following states - MANUAL, ARMING, TAKEOFF, WAYPOINT, LANDING and DISARMING. Those states help to execute a path that was planned fore the drone.
The scripts differ in the plan executed in each one:
backyard_flyer_solution.py - contains basic code to takeoff, fly in a square and land.
motion_planning.py - contains a flight planning phase avoiding obstacles before the flight execution. It reads a map of obstacles, creates a free space graph using [Voronoi method](https://en.wikipedia.org/wiki/Voronoi_diagram) and finds a route from start to goal positions avoiding obstacles using [A* (A-star) algorithm](https://en.wikipedia.org/wiki/A*_search_algorithm). 

### Map
The obstacles map is given in colliders.csv file and contains two pieces of information:
- Global position in [Geodetic coordinates](https://en.wikipedia.org/wiki/Geodetic_datum)
- Location of obstacles and their size in local coordinates relative to map center

### Coordinate conversion
The Geodetic coordinates are converted to local coordinates using python [utm module](https://pypi.org/project/utm/) which uses [Cartesian coordinates to express the location](https://en.wikipedia.org/wiki/Universal_Transverse_Mercator_coordinate_system). The conversion requires setting a reference position which is called in the code global_home and is set using the method `set_home_position()` to the global Geodetic position that was read from colliders.csv
The drone position is updated in underlying udacidrone class and is available in a variable called `global_position`. This position is in Geodetic coordinates and is converted to local Cartesian coordinates using the function `global_to_local()` from planning_utils.py (which uses python module utm).   

### Space modeling
The space is modeled as a 2D grid such as grid(0,0) corresponds to the most negative offset in the obstacle map in local coordinates. This offset is calculated in the function `create_grid_and_edges()`. We also assume that `global_home` (extracted from first line of colliders.csv)is also at the same point - grid(0,0).

### Start position
The start position on the grid is the drone current position, but the latter is given in local coordinates, which can be negative numbers whereas grid coordinates must be positive integers. Therefore we convert the local position to grid position by adding the (nort_offset, east_offset) which correspond to grid(0,0).

### Goal position
The goal position is specified in global coordinates, in this case I chose the (lat, lon) coordinates of the address "275 Sacramento St" in Google maps which is within the map range. The position is then converted to local coordinates using `global_to_local()` and from there to grid coordinates by adding the offsets.

### Path search
The algorithm used to find the path is A* (A-star) on a graph. The edges of the graph are connections between free space paths based on Voronoi calculations.

### Path prunning
The path that the A* algorithm finds might contain points that can be removed by connecting the non-adjustent points together. This is done by the `prune_path()` method in planning_utils.py. This method runs over all path points and for each i and i+2 points checks if the direct connection between them collides with an obstacle. If there is no collision the path point in the middle (i+1) can be removed. Collision check is done using the [bresenham method](https://en.wikipedia.org/wiki/Bresenham%27s_line_algorithm) and based on the obstacle grid that we already created in previous steps. With the example selected goal location, for example, the number of path points reduces from 31 to 5.

