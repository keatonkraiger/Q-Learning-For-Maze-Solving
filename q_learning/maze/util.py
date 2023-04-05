import configparser
import random
import torch
import numpy as np
from collections import deque
from gym.wrappers import RecordVideo

class CFG:
    def __init__(self, config_file):
        self.config = configparser.ConfigParser()
        self.config.read('config.cfg')
        self.MIN_LEARNING_RATE = self.config.getfloat('parameters', 'MIN_LEARNING_RATE')
        self.MIN_EXPLORE_RATE = self.config.getfloat('parameters', 'MIN_EXPLORE_RATE')
        self.NUM_EPISODES = self.config.getint('parameters', 'NUM_EPISODES')
        self.STREAK_TO_END = self.config.getint('parameters', 'STREAK_TO_END')
        self.AVG_N = self.config.getint('parameters', 'AVG_N')
        self.ENV_NAME = self.config.get('env', 'ENV_NAME')
        self.DEBUG_MODE = self.config.getboolean('env', 'DEBUG_MODE')
        self.RENDER_MAZE = self.config.getboolean('env', 'RENDER_MAZE')
        self.ENABLE_RECORDING = self.config.getboolean('env', 'ENABLE_RECORDING')
        self.RECORDING_FOLDER = self.config.get('env', 'RECORDING_FOLDER')
        self.RECORDING_RATE = self.config.getint('env', 'RECORDING_RATE')
