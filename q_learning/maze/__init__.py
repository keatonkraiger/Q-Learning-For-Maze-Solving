from gym.envs.registration import register

register(
    id='maze-v0',
    entry_point='maze.maze_env:MazeEnvSample5x5',
    max_episode_steps=2000,
)

register(
    id='maze-sample-5x5-v0',
    entry_point='maze.maze_env:MazeEnvSample5x5',
    max_episode_steps=2000,
)

register(
    id='maze-random-5x5-v0',
    entry_point='maze.maze_env:MazeEnvRandom5x5',
    max_episode_steps=2000,
    nondeterministic=True,
)

register(
    id='maze-sample-10x10-v0',
    entry_point='maze.maze_env:MazeEnvSample10x10',
    max_episode_steps=10000,
)

register(
    id='maze-random-10x10-v0',
    entry_point='maze.maze_env:MazeEnvRandom10x10',
    max_episode_steps=10000,
    nondeterministic=True,
)

register(
    id='maze-sample-3x3-v0',
    entry_point='maze.maze_env:MazeEnvSample3x3',
    max_episode_steps=1000,
)

register(
    id='maze-random-3x3-v0',
    entry_point='maze.maze_env:MazeEnvRandom3x3',
    max_episode_steps=1000,
    nondeterministic=True,
)


register(
    id='maze-sample-100x100-v0',
    entry_point='maze.maze_env:MazeEnvSample100x100',
    max_episode_steps=1000000,
)

register(
    id='maze-random-100x100-v0',
    entry_point='maze.maze_env:MazeEnvRandom100x100',
    max_episode_steps=1000000,
    nondeterministic=True,
)

register(
    id='maze-random-10x10-plus-v0',
    entry_point='maze.maze_env:MazeEnvRandom11x10Plus',
    max_episode_steps=1000000,
    nondeterministic=True,
)

register(
    id='maze-random-20x20-plus-v0',
    entry_point='maze.maze_env:MazeEnvRandom20x20Plus',
    max_episode_steps=1000000,
    nondeterministic=True,
)

register(
    id='maze-random-30x30-plus-v0',
    entry_point='maze.maze_env:MazeEnvRandom30x30Plus',
    max_episode_steps=1000000,
    nondeterministic=True,
)

# Register the maze environment with OpenAI gym. I register the 3x3 example maze.
register(
    id='maze-regular-3x3-v0',
    entry_point='maze.maze_env:MazeEnvRegular3x3',
    max_episode_steps=1000000,
)
