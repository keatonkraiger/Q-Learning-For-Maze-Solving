# Classical Q Learning for Maze Environment

Spring 2023
TAs: Keaton Kraiger & Shimian Zhang

``` bash
  /\ ___ /\
 (  o   o  ) Wecome to PRML Project 4, Classical Q Learning!
  \  >#<  /
  /       \  
 /         \       ^
|           |     //
 \         /    //
  ///  ///   --
```

We provide the code for training an agent using Q-learning on various mazes. You will be tasked with trying to train an agent using various mazes. Parameters that can change include the size (3x3, 5x5, 10x10, and 100x100). We recommend first trying to train the agent on a maze (say 10x10) and observing performance. Then, you will create a maze of your own design and train the agent on that maze. This maze will need to have some regularity to it (e.g. a repeating wall pattern or entrance/exits). You will need to report back on the performance of the two agents and see if one performs better (if you want to plot something like reward you will need to add functionality for it). Have fun and good luck!

## Installation

We recommend creating a virtual environment for this project (specifically, we recommend using anaconda). Create a new environment and install the required packages using the following commands:

``` bash
conda create -n p4 python=3.8
conda activate p4
pip install -r requirements.txt
pip install -e .
```

## Running the code

To train the q-learning agent, you will be running the 'q_learning.py' file. The configuration of the agent, map, and training are defined in the 'config.cfg' file. Adjust the config file to your liking to train the agent.

## Quick Start

To train the agent on a 5x5 maze, adjust the config file with the following settings:

``` bash
env_name = maze-sample-5x5-v0


## Environment

This code directly uses the environment from the [gym-maze](https://github.com/MattChanTK/gym-maze).

The enviornment is a a simple 2D maze environment where an agent (blue dot) finds its way from the top left corner (blue square) to the goal at the bottom right corner (red square). 
The objective is to find the shortest path from the start to the goal.

<kbd>![Simple 2D maze environment](http://i.giphy.com/Ar3aKxkAAh3y0.gif)</kbd>

#### Action space

The agent may only choose to go up, down, right, or left ("N", "S", "E", "W"). If the way is blocked, it will remain at the same the location. 

#### Observation space

The observation space is the (x, y) coordinate of the agent. The top left cell is (0, 0).

#### Reward

A reward of 1 is given when the agent reaches the goal. For every step in the maze, the agent recieves a reward of -0.1/(number of cells).

#### End condition

The maze is reset when the agent reaches the goal. 

### Maze Versions

#### Pre-generated mazes

* 3 cells x 3 cells: _MazeEnvSample3x3_
* 5 cells x 5 cells: _MazeEnvSample5x5_
* 10 cells x 10 cells: _MazeEnvSample10x10_
* 100 cells x 100 cells: _MazeEnvSample100x100_

#### Randomly generated mazes (same maze every epoch)

* 3 cells x 3 cells: _MazeEnvRandom3x3_
* 5 cells x 5 cells: _MazeEnvRandom5x5_
* 10 cells x 10 cells: _MazeEnvRandom10x10_
* 100 cells x 100 cells: _MazeEnvRandom100x100_

#### Randomly generated mazes with portals and loops

With loops, it means that there will be more than one possible path.
The agent can also teleport from a portal to another portal of the same colour. 

* 10 cells x 10 cells: _MazeEnvRandom10x10Plus_
* 20 cells x 20 cells: _MazeEnvRandom20x20Plus_
* 30 cells x 30 cells: _MazeEnvRandom30x30Plus_

### Maze Generation Table

When loading a maze from a predefined maze npy file, the following table is used to determine the maze layout where the number corresponds to the locations of the walls. At each cell, walls can be north, south, east, or west of the cell.
| Number  | Wall removed|
| ------- | ----------- |
| 1       | North       |
| 2       | East        |
| 4       | South       |
| 8       | West        |

## Creating your own maze

Mazes are created by npy files defining the maze layout. Maze layouts will need to by sized 3x3, 5x5, 10x10, or 100x100 and subsequently stored in the maze/maze_env/maze_samples folder.
To create a map, the code first "creates" walls bordering each cell by making N+1 horizontal and N+1 vertical lines. The read in file defining the maze layout will be used to determine which walls are
"torn down". Below is an example of a 3x3
Initial maze layout with all walls intact:

``` bash
+--+--+--+
|  |  |  |
+--+--+--+
|  |  |  | = np.zeros((3,3)) [maze array]
+--+--+--+
|  |  |  |
+--+--+--+
We can then define which walls to remove
maze[0,0] = 2 (remove east wall of cell (0,0))
+--+--+--+                  +--+--+--+
|     |  |                  |     |  |
+--+--+--+                  +--+  +--+
|  |  |  | , maze[0,1]=4... |  |  |  |, etc.
+--+--+--+                  +--+--+--+  
|  |  |  |                  |  |  |  |
+--+--+--+                  +--+--+--+
```

A complete maze layout definition can be found in `map_generator.py`. You may either create the maze
directly in the `map_generator.py` file or create a new file csv file that defines the maze layout and load it in the `map_generator.py` file as an argument. Give the saved npy file a
name that won't conflict with any existing maze files.

Feel free to add code to the map generator to make the maze generation process easier. For example, for a larger maze (say 100x100) you wouldn't want to do it by hand. You could try looping over the maze to make regular patterns in the maze walls. 

## Adding the map to the environment

* Once the map is created you will need to move it to `maze/maze_env/maze_samples` and then register the map `maze/__init__.py` file. Follow the existing examples.
* You will also need to add or modify a class for the map in `maze/maze_env/maze_env.py` file. Follow the existing examples. Make sure the `entry_point` you register is the as the class name you create.

Note that you will need to run the map_generator.py file to generate the sample map and move it into the correct directory before trying to it in the environment.

## Note on the code

Please note that currently the recording of the agent's performance is not working (setting cfg.ENABLE_RECORDING to True will cause issues).
If you want to try and fix it please feel free, its pretty close to working we just didn't have the time.
