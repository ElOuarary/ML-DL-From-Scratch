import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation

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