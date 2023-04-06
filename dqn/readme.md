# Q Learning for Maze Environment

Spring 2023
TAs: Keaton Kraiger & Shimian Zhang

``` bash
    ,-. __ .-,
  --;`. '   `.'
   / (  ^__^  ) Welcome to Project 4, Deep Q Learning!
  ;   `(_`'_)' \
  '  ` .`--'_,  ;
~~`-..._)))(((.'
```

We provide the code for training an agent using deep Q-learning on simple mazes. You will be tasked with trying to train an agent using various mazes.

We provide three preset maps, defined as CSVs for you to work with. You will need to create all other maps. We recommend you first try experimenting with the preset maps to get an idea for performance.

Then, you will create mazes that will need to have some regularity to it (e.g. a repeating wall pattern or entrance/exits). You will need to report back on the performance of the different mazes and see if one performs better. Have fun and good luck!

## Installation

We recommend you first go to the q-learning directory's readme to see instructions on setting up an environment.

## Quick Start

After installing the required packages, you can run the code by running the following command

``` bash
python deep_q_learning.py --maze_csv maze_samples/maze-sample-5x5.csv --config_file config.cfg
```

To run with your GPU, set the `device` variable to `cpu` in the `config.cfg` file.

## File and Code Strucutre

We implement the DQN in the `deep_q_learning.py` file. Utility functions such as those for plotting are in the `utils.py` file. All preset mazes are in the `maze_samples` directory. 

The code uses parameters that are defined in the config.cfg file. Adjust the config file to your liking to train the agent.

## Maze Generation

The DQN agent is trained in a slightly different environment than the Q-learning agent. In this code, the maze structure is more simple and less expressive. To define mazes, you no longer specify which walls will be torn down. Instead, you simply specify which cells contain walls (or blockage) and which don't. Defining a maze is relatively simple. Mazes are defined as below
| Character | Meaning |
|-----------|---------|
| `0`       | Empty cell |
| `1`       | Wall |
Your agent will always start at (0,0) and the end location will always be at (n-1, n-1) where n is the size of the maze.

In the code, mazes are read in from a csv that describes the mazes cells. The csv should be have n rows and n columns.

## Creating Your Own Maze

To create a maze you will need to make a csv file that describes the maze. The csv should be have n rows and n columns and the file should be the name of the maze. Below is an example csv file that defines a 5x5 maze
  
  ``` csv
  0,0,0,0,0
  0,1,1,1,0
  0,1,0,1,0
  0,1,0,0,0
  0,0,0,0,0
  ```
You may then save the csv file and give it a meaningful name. For example we name the above maze `maze-sample-5x5.csv`. We also recommend you place the maze in the `maze_samples` directory.

## Running the Code

The `deep_q_learning.py` expects two arguments. The first is the `maze_csv` that gives the maze csv file path and `config_file` which gives the config file path. 

Below is an example of how to run the code

``` bash
python deep_q_learning.py --maze_csv maze_samples/maze-sample-5x5.csv --config_file config.cfg
```

You may need to modify the `config.cfg` depending on the experiment you are running. Some relevant parameters are
* `max_episodes` - The maximum number of episodes to train the agent for.
* `consecutive_solved` - The number of consecutive episodes the agent must solve the maze for to be considered solved.
* `window_plot_len` - The window using for plotting moving averages.
* `solved_threshold` - The threshold for the number of steps the agent must take to be considered a good solution. 
* `test_interval` - The number of episodes between testing the agent.
