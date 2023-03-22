# Q Learning for Maze Environment

``` bash
  /\ ___ /\
 (  o   o  ) Wecome to PRML Project 4, Deep Q Learning!
  \  >#<  /
  /       \  
 /         \       ^
|           |     //
 \         /    //
  ///  ///   --
```

We provide the code for training an agent using deep Q-learning on simple mazes. You will be tasked with trying to train an agent using various mazes.
Unlike in the q-learning code, there are no preset maps (except the one in the code) for you to work with. You will need to create all maps in the code. We recommend you first try making "regular" maps (or you could even do it algorithmically) to test the performance.
Then, you will create mazes that will need to have some regularity to it (e.g. a repeating wall pattern or entrance/exits). You will need to report back on the performance of the different mazes and see if one performs better. Have fun and good luck!

## Installation

We recommend you first go to the q-learning directory's readme to see instructions on setting up an environment.

## Running the code

To train the DQN, you simply need to run the 'dqn.py' file. The maze layout and hyperparameters are defined in the dqn file itself. For creating a maze you can modify the 'get_new_maze' function.
