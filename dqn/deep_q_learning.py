import argparse
import time
import os
import numpy as np
import configparser
import csv
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, PReLU
from tensorflow.keras.optimizers import SGD , Adam, RMSprop
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.patches as patches

from utils import *

# Define actions
actions = {'U': (-1, 0), 'D': (1, 0), 'L': (0, -1), 'R': (0, 1)}
action_arr = ['U', 'D', 'L', 'R']

def read_maze_from_csv(filename='maze.csv'):
    """
    Read the maze from a csv file
    Args:
        filename (str): The name of the csv file
    Returns:
        maze (list): The maze
    """
    with open(filename, newline='') as csvfile:
        maze_data = list(csv.reader(csvfile))
        maze = [[int(cell) for cell in row] for row in maze_data]
    return maze

def get_position_index(loc, width):
    """
    Get the index of the position in the Q-table
    Args:
        loc (tuple): The location
        width (int): The width of the maze
    Returns:
        index (int): The index of the position in the Q-table
    """
    return loc[0] * width + loc[1]

def try_maze(max_moves, model, allowed_states):
    """
    Try to solve the maze
    Args:
        max_moves (int): The maximum number of moves
        model (keras model): The model to use
        allowed_states (np array): The allowed states
    Returns:
        locations (list): The locations visited
    """
    maze = read_maze_from_csv(MAZE_CSV)
    num_moves = 0
    start_location = (0, 0)
    height, width = len(maze), len(maze[0])
    end_location = (height - 1, width - 1)
    current_location = start_location
    x, y = [] , []
    x.append(current_location[1])
    y.append(current_location[0])
    num_states = height * width 
    exploit_moves = np.argmax(model.predict(np.identity(num_states)), axis=1)
    while (current_location != end_location and num_moves < max_moves):
        # Choose next action
        next_move = exploit_moves[get_position_index(current_location, width)]
        # Update the state
        maze, current_location, _ = do_move(maze, current_location, end_location, next_move, 
                                                allowed_states, width)

        x.append(current_location[1])
        y.append(current_location[0])
        num_moves += 1
    locations = list(zip(x, y))
    return num_moves, locations


def do_move(maze, current_location, end_location, move_idx, allowed_states, width):
    """
    Do a move in the maze
    Args:
        maze (list): The maze
        current_location (tuple): The current location
        end_location (tuple): The end location
        move_idx (int): The index of the move
        allowed_states (np array): The allowed states
        width (int): The width of the maze
    Returns:
        new_maze (list): The new maze
        new_location (tuple): The new location
    """
    new_maze = maze
    y, x = current_location
    direction = actions[action_arr[move_idx]]
    pos_idx = get_position_index(current_location, width)
    if (allowed_states[pos_idx][move_idx] == 1):
        new_x = x + direction[1]
        new_y = y + direction[0]
        new_location = (new_y, new_x)
    else:
        new_location = current_location
    # Get reward
    if (new_location == end_location):
        current_reward = 0
    else:
        current_reward = -1
    return (new_maze, new_location, current_reward)

def main(cfg):
    # Read in the cfg file and the maze csv file

    # Hyperparameters
    alpha = cfg.alpha
    random_factor = cfg.random_factor
    drop_rate = cfg.drop_factor
    test_interval = cfg.test_interval

    # Set up the environment variables
    maze = read_maze_from_csv(MAZE_CSV)
    plot_maze(maze, MAZE_CSV, display_fig=False, save_fig=True)
    height, width = len(maze), len(maze[0])
    end_location = (height - 1, width - 1)
    max_steps = height * width * 25
    max_episodes = cfg.max_episodes
    num_states = height * width
    start_location = (0, 0)
    end_location = (height - 1, width - 1)
    allowed_states = np.zeros((width * height, 4))
    start_time = time.time()

    # Set up the model
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

    step_counts = []
    ep_steps = []
    ep_rewards = []
    consecutive_good_win = 0
    visited_cells = []
    for iteration in range (cfg.max_episodes):
        # Reset important variables
        steps = 0
        cumulative_reward = 0
        current_location = start_location
        state_history = []
        maze = read_maze_from_csv(MAZE_CSV)
        print("ITERATION {}".format(iteration+1), end = ' ')
        # Prefetch exploits
        exploit_moves = np.argmax(model.predict(np.identity(num_states)), axis=1)
        # Do a run through the maze
        while (current_location != end_location):
            next_step = None

            # Exploration
            if (np.random.random() < random_factor):
                next_step = np.random.choice([0, 1, 2, 3])
            # Exploit
            else:
                next_step = exploit_moves[get_position_index(current_location, width)]

            # Update the state
            prev_location = current_location
            maze, current_location, current_reward = do_move(maze, current_location, end_location, next_step, allowed_states, width)
            steps += 1
            cumulative_reward += current_reward
            visited_cells.append(current_location)

            # Update state history
            if (steps > max_steps or current_location == end_location):
                run_end = True
            else:
                run_end = False
            state_history.append((prev_location, next_step, current_reward, current_location, run_end))

            if (current_location == end_location):
                print("- TRAINING WIN in {} moves!".format(steps), end = ' ')
            if (steps> max_steps):
                current_location = end_location
            
        ep_steps.append(steps)
        ep_rewards.append(cumulative_reward)

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
        h = model.fit(input_mat, target_mat, epochs=4, batch_size=256, verbose=0)
        # Lower the random factor
        random_factor *= drop_rate
        print("- Total Runtime: {:0.3f} minutes".format((time.time() - start_time) / 60))
        if (iteration % 250 == 0):
            print(f"{iteration} iterations completed")

        # Test and visualize the model
        if ((iteration+1) % test_interval == 0):
            num_moves, locations = try_maze(max_steps, model, allowed_states)
            # Apend the visited cells and keep the (x, y) format
            if num_moves == max_steps:
                print("TEST FAIL")
                consecutive_good_win = 0
            else:
                print("TEST WIN ({} moves)".format(num_moves))
                if (num_moves < cfg.solved_threshold):
                    consecutive_good_win += 1
                else:
                    consecutive_good_win = 0

            step_counts.append(num_moves)

        # if consecutively winning we break
        if (consecutive_good_win >= cfg.consecutive_solved):
            # Set up everything for rendering
            x_test = [cell[1] for cell in visited_cells]
            y_test = [cell[0] for cell in visited_cells]
            plot_visitation_frequency(x_test, y_test, (height, width), MAZE_CSV)
            break

    # Plot the steps and rewards
    ep_steps = np.array(ep_steps)
    maze_name = MAZE_CSV.split('.')[0].split('/')[-1]
    file_name = f"imgs/dqn_{maze_name}_steps.png"
    plot_moving_average_with_std(ep_steps, window_size=cfg.window_plot_len, data_name="Steps", file_save=file_name)

    ep_rewards = np.array(ep_rewards)
    file_name = f"imgs/dqn_{maze_name}_rewards.png"
    plot_moving_average_with_std(ep_rewards, window_size=cfg.window_plot_len, data_name="Rewards", file_save=file_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Q-learning maze solver")
    parser.add_argument("--maze_csv", type=str, default="maze-sample-10x10.csv", help="The path to the maze CSV file")
    parser.add_argument("--config_file", type=str, default='config.cfg', help="The path to the config file")
    args = parser.parse_args()
    cfg = CFG(args.config_file)

    # Move to the device 
    if cfg.device == 'cpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = -1
    global MAZE_CSV
    MAZE_CSV = args.maze_csv
    if not os.path.exists("imgs"):
        os.mkdir("imgs")
    main(cfg)