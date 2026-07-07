import gymnasium as gym


ENV_NAME = "LunarLander-v3"


def make_env(render_mode=None):
    # 统一从这里创建环境
    return gym.make(ENV_NAME, render_mode=render_mode)
