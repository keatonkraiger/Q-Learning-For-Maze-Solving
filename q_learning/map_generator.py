import sys
import numpy as np

# Check if a csv file is passed as an argument. If so load it, otherwise create it in numpy
if len(sys.argv) > 1:
    maze_data = np.loadtxt(sys.argv[1], delimiter=',', dtype=int)
    assert maze_data.shape[0] == maze_data.shape[1], "Map must be square"
else:
    maze_data = np.zeros((3, 3), dtype=int)
    # Recall the map starts with ALL walls up (perimeter and interior) and at each cell
    # we define which walls get taken down (1=North, 2=East, 4=South, 8=West). The agent
    # starts at the top left corner (0,0) and the goal is at the bottom right corner (n-1,n-1)
    maze_data[0, 0] = 2
    maze_data[1, 0] = 4
    maze_data[1, 1] = 2
    maze_data[1, 2] = 1
    maze_data[0, 2] = 2
    maze_data[2, 0] = 4
    maze_data[2, 2] = 1
    '''
    # Here is the maze we just created
    +--+--+--+
    |     |  | 
    +--+  +  +
    |  |     | 
    +--+  +  +
    |     |  |
    +--+--+--+
    '''

# Note that the code expects it a certain way so we pre-process it here.
maze_data = np.swapaxes(maze_data, 0, 1)
maze_data = maze_data.T

save_map_name = 'maze2d_regular_3x3.npy'
np.save(save_map_name, maze_data)
