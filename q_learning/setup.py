from setuptools import setup

setup(name="maze",
      version="0.1",
      packages=["maze", "maze.maze_env"],
      package_data = {
          "maze.maze_env": ["maze_samples/*.npy"]
      },
)
