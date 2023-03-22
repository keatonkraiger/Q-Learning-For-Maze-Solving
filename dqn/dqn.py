import time
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # Force CPU, comment to use GPU
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, PReLU
from tensorflow.keras.optimizers import SGD , Adam, RMSprop
import matplotlib.pyplot as plt

# Define actions
actions = {'U': (-1, 0), 'D': (1, 0), 'L': (0, -1), 'R': (0, 1)}
action_arr = ['U', 'D', 'L', 'R']

# Function for printing the maze if you need it
def print_maze(maze):
    print("██████████")
    for row in maze:
        print("█", end='')
        for col in row:
            if (col == 0):
                print(' ', end='')
            elif (col == 1):
                print('█', end ='')
            elif (col == 2):
                print('O', end='')
        print("█")
    print("██████████")

# Return a clean copy of the maze
def get_new_maze():
    maze = [[2, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 1],
            [0, 0, 0, 0, 1, 0, 0, 0]]
    return maze

# Update the state of the maze and agent
def do_move(maze, current_location, end_location, move_idx, allowed_states, width):
    new_maze = maze
    y, x = current_location
    direction = actions[action_arr[move_idx]]
    pos_idx = get_position_index(current_location, width)
    if (allowed_states[pos_idx][move_idx] == 1):
        new_x = x + direction[1]
        new_y = y + direction[0]
        #new_maze[y][x] = 0
        #new_maze[new_y][new_x] = 2
        new_location = (new_y, new_x)
    else:
        new_location = current_location
    # Get reward
    if (new_location == end_location):
        current_reward = 0
    else:
        current_reward = -1
    return (new_maze, new_location, current_reward)

def get_position_index(loc, width):
    return loc[0] * width + loc[1]

# Try an iteration of the maze
def tryMaze(max_moves):
    num_moves = 0
    start_location = (0, 0)
    end_location = (7, 7)
    current_location = start_location
    x = []
    y = []
    x.append(current_location[1] + 0.5)
    y.append(current_location[0] + 0.5)
    exploit_moves = np.argmax(model.predict(np.identity(64)), axis=1)
    maze = get_new_maze()
    while (current_location != end_location and num_moves < max_moves):
        # Choose next action
        next_move = exploit_moves[get_position_index(current_location, width)]
        # Update the state
        maze, current_location, _ = do_move(maze, current_location, end_location, next_move, allowed_states, width)
        x.append(current_location[1] + 0.5)
        y.append(current_location[0] + 0.5)
        num_moves += 1
    return num_moves, x, y

# Set up important variables
start_location = (0, 0)
end_location = (7, 7)
width = 8
height = 8
alpha = 0.95
random_factor = 0.8
drop_rate = 0.995
max_moves = width * height * 2
test_interval = 5
maze = get_new_maze()
allowed_states = np.zeros((width * height, 4))
start_time = time.time()

# Set up neural network
model = Sequential()
model.add(Dense(width * height, input_shape=(width * height,)))
model.add(Dense(width * height, activation='relu'))
model.add(Dense(len(action_arr), activation='linear'))
model.compile(optimizer='adam', loss='mse')

# Construct allowed states
for y in range(len(maze)):
    for x in range(len(maze[y])):
        pos_idx = get_position_index((y, x), width)
        if (maze[y][x] == 1):
            continue
        for action_idx in range(len(action_arr)):
            action = action_arr[action_idx]
            new_x = x + actions[action][1]
            new_y = y + actions[action][0]
            if (new_x < 0 or new_x > width-1 or new_y < 0 or new_y > height-1):
                continue
            if maze[new_y][new_x] == 0 or maze[new_y][new_x] == 2:
                allowed_states[pos_idx][action_idx] = 1


# The training happens here
move_counts = []
consecutive_good_win = 0
for iteration in range (5000):
    # Reset important variables
    moves = 0
    current_location = start_location
    state_history = []
    maze = get_new_maze()
    print("ITERATION {}".format(iteration+1), end = ' ')
    # Prefetch exploits
    exploit_moves = np.argmax(model.predict(np.identity(64)), axis=1)
    # Do a run through the maze
    while (current_location != end_location):
        # Choose next action
        next_move = None
        # Explore
        if (np.random.random() < random_factor):
            next_move = np.random.choice([0, 1, 2, 3])
        # Exploit
        else:
            next_move = exploit_moves[get_position_index(current_location, width)]
        # Update the state
        prev_location = current_location
        maze, current_location, current_reward = do_move(maze, current_location, end_location, next_move, allowed_states, width)
        moves += 1
        # Update state history
        if (moves > max_moves or current_location == end_location):
            run_end = True
        else:
            run_end = False
        state_history.append((prev_location, next_move, current_reward, current_location, run_end))
        # If robot takes too long, just end the run
        if (current_location == end_location):
            print("- TRAINING WIN in {} moves!".format(moves), end = ' ')
        if (moves > max_moves):
            current_location = end_location

    # DO DNN LEARNING
    data_size = len(state_history)
    input_mat = np.zeros((data_size, width * height))
    current_loc_mat = np.zeros((data_size, width * height))
    target_mat = np.zeros((data_size, len(action_arr)))
    for i in range(data_size):
        prev_loc, _, _, curr_loc, _ = state_history[i]
        prev_loc_idx = get_position_index(prev_loc, width)
        curr_loc_idx = get_position_index(curr_loc, width)  
        input_mat[i][prev_loc_idx] = 1
        current_loc_mat[i][curr_loc_idx] = 1

    target_mat = model.predict(input_mat)
    Qval = np.amax(model.predict(current_loc_mat), axis=1)
    for i in range(data_size):
        prev_loc, move, reward, curr_loc, run_end = state_history[i]
        if (run_end == True):
            target_mat[i][move] = reward
        else:
            target_mat[i][move] = reward + alpha * Qval[i]
    h = model.fit(input_mat, target_mat, epochs=4, batch_size=64, verbose=0)
    # Lower the random factor
    random_factor *= drop_rate
    print("- Total Runtime: {:0.3f} minutes".format((time.time() - start_time) / 60))

    # Test and visualize the model
    if ((iteration+1) % test_interval == 0):
        test_val, test_x, test_y = tryMaze(max_moves)
        if test_val == max_moves:
            print("TEST FAIL")
            consecutive_good_win = 0
        else:
            print("TEST WIN ({} moves)".format(test_val))
            if (test_val < 25):
                consecutive_good_win += 1
            else:
                consecutive_good_win = 0

        # Store number of moves for plotting
        move_counts.append(test_val)

    # If the agent can win with few moves 3 times in a row, consider it to be trained
    if (consecutive_good_win >= 3):
        # Set up everything for rendering
        fig = plt.figure(1)
        #plt.get_current_fig_manager().window.state('zoomed')
        fig.show()
        fig.canvas.draw()
        xedges = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        yedges = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        heatmap, _, _ = np.histogram2d(test_x, test_y, bins=(xedges, yedges))
        extent = [0, 8, 0, 8]
        plt.clf()
        plt.imshow(heatmap.T, extent=extent, origin='lower')
        plt.xlabel("Agent x-coordinate", fontsize=20)
        plt.ylabel("Agent y-coordinate", fontsize=20)
        plt.gca().invert_yaxis()
        plt.gca().xaxis.tick_top()
        plt.gca().xaxis.set_label_position('top')
        cb = plt.colorbar(ticks=[0, heatmap.max()])
        cb.ax.set_yticklabels(['Cell never entered by agent', 'Cell frequently entered by agent'])
        cb.ax.tick_params(labelsize=16)
        fig.canvas.draw()
        plt.pause(1)
        

fig2 = plt.figure(2)
plt.semilogy(move_counts, 'b', linewidth=0.5)
plt.xlabel("Iteration Number", fontsize=16)
plt.ylabel("Number of moves taken", fontsize=16)
plt.show()