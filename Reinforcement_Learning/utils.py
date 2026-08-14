import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation

import csv
import os
import psutil

class MetricLog:
    def __init__(self, path="metrics.csv"):
        self.path = path
        self.process = psutil.Process(os.getpid())
        with open(self.path, "w") as f:
            writer = csv.writer(f)
            writer.writerow([
                "iteration", "step_time_s", "rss_mb", "cpu_percent"
            ])

    def log(self, iteration,step_time_s):
        rss = self.process.memory_info().rss / (1024 * 1024)
        cpu = self.process.cpu_percent()
        with open(self.path, "a") as f:
            writer = csv.writer(f)
            writer.writerow([
                iteration, step_time_s, rss, cpu
            ])

def make_atari_env(env_id, render_mode=None, grayscale_obs=True):
    def _make():
        env = gym.make(env_id, render_mode=render_mode)
        env = AtariPreprocessing(
            env=env,
            frame_skip=1,
            screen_size=84,
            grayscale_obs=grayscale_obs
        )
        env = FrameStackObservation(env, stack_size=4)
        return env
    return _make