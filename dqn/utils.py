import configparser
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

class CFG:
    def __init__(self, config_file):
        self.config = configparser.ConfigParser()
        self.config.read(config_file)
        self.alpha = self.config.getfloat('parameters', 'alpha')
        self.random_factor = self.config.getfloat('parameters', 'random_factor')
        self.drop_factor = self.config.getfloat('parameters', 'drop_factor')
        self.test_interval = self.config.getint('parameters', 'test_interval')
        self.max_episodes = self.config.getint('parameters', 'max_episodes')
        self.solved_threshold = self.config.getfloat('parameters', 'solved_threshold')
        self.consecutive_solved = self.config.getint('parameters', 'consecutive_solved')
        self.window_plot_len = self.config.getint('parameters', 'window_plot_len')
        self.device = self.config.get('parameters', 'device')

def plot_moving_average_with_std(data, window_size, alpha=0.3, data_name='Reward', file_save=None):
    """
    Plots the moving average of the data with a standard deviation shaded area.
    
    Args:  
        data: 1D numpy array of data points
        window_size: integer size of the moving average window
        alpha: transparency of the shaded area
        data_name: name of the data to be used in the plot legend
        file_save: path to save the plot to. If None, the plot is not saved.
    """
    moving_average = np.convolve(data, np.ones(window_size), mode='valid') / window_size
    moving_std = np.array([np.std(data[i:i+window_size]) for i in range(len(data) - window_size + 1)])

    # Define the upper and lower bounds for the shaded area
    upper_bound = moving_average + moving_std
    lower_bound = moving_average - moving_std

    x_values = np.arange(window_size - 1, len(data))
    plt.plot(x_values, moving_average, label='Moving Average', color='blue')
    plt.fill_between(x_values, lower_bound, upper_bound, color='blue', alpha=alpha, label='Moving Std Deviation')
    plt.title(f'{data_name} Moving Average')
    plt.xlabel('Episode')
    plt.ylabel(data_name)
    if file_save is not None:
        plt.savefig(file_save)
    else:
        plt.show()
    plt.close()


def plot_maze(maze, maze_csv, display_fig=False, save_fig=True):
    """
    Plot the maze
    Args:
        maze (list): The maze
        maze_csv (str): The path to the maze csv file
        display_fig (bool): Whether to display the figure
        save_fig (bool): Whether to save the figure
    """
    height, width = len(maze), len(maze[0])
    plt.figure(figsize=(10, 10))
    cmap = matplotlib.colors.ListedColormap(['white', (0, 0, 155/255), 'black', (155/255, 0, 0)])
    # Update maze values to match the colormap
    modified_maze = np.array(maze)
    modified_maze[height-1][width-1] = 3
    modified_maze[modified_maze == 1] = 2
    modified_maze[0][0] = 1
    plt.imshow(modified_maze, cmap=cmap)

    # Add grid lines
    for x in range(width + 1):
        plt.axvline(x-0.5, linestyle='-', color='black', linewidth=0.5)
    for y in range(height + 1):
        plt.axhline(y-0.5, linestyle='-', color='black', linewidth=0.5)

    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    if save_fig:
        maze_name = maze_csv.split('.')[0].split('/')[-1]
        plt.savefig(f'imgs/{maze_name}_layout.png', dpi=300, bbox_inches='tight', pad_inches=0)
    if display_fig:
        plt.show()
    plt.close()

def plot_visitation_frequency(x_test, y_test, maze_shape, maze_csv):
    """
    Plot the visitation frequency of the maze
    Args:
        x_test (list): The x coordinates of the test data
        y_test (list): The y coordinates of the test data
        maze_shape (tuple): The shape of the maze
        maze_csv (str): The path to the maze csv file
    """
    visitation_frequency = np.zeros(maze_shape)


    for x, y in zip(x_test, y_test):
        visitation_frequency[int(y), int(x)] += 1

    visitation_frequency /= np.max(visitation_frequency)

    fig, ax = plt.subplots(figsize=(10, 10))
    plt.title('Cell Visitation Frequency', fontsize='30', y=1.05)
    im = ax.imshow(visitation_frequency, cmap='viridis')

    # Add grid lines
    height, width = maze_shape
    for x in range(width + 1):
        ax.axvline(x-0.5, linestyle='-', color='black', linewidth=0.5)
    for y in range(height + 1):
        ax.axhline(y-0.5, linestyle='-', color='black', linewidth=0.5)

    ax.set_xticks([])
    ax.set_yticks([])

    # Add colorbar
    cbar = fig.colorbar(im, ax=ax)
    #cbar.set_label("Visited Frequency", rotation=270, labelpad=20)
    cbar.ax.set_ylabel('Frequency', fontsize='25')  # Update this line

    plt.tight_layout()
    filename = maze_csv.split('.')[0].split('/')[-1]
    plt.savefig(f'imgs/{filename}_visitation_frequency.png', dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()
