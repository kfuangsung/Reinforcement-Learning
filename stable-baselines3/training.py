import gym
import numpy as np
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise
from sb3_settings import *
from sb3_utils import *

def main():
    n_steps = 1e5
    lr = 0.0025
    
    print('-'*50)
    print("***** Training *****")
    print(f"RL-Algorithm: {algo_name}")
    print(f"Environment: {env_name}")
    print(f"Learning rate: {lr}")
    print(f"Total timesteps: {n_steps}")
    print('-'*50)

    save_name = f"{algo_name}_{env_name}"
    model_save_path = f"./{algo_name}/models/{save_name}"
    tb_path = f"./{algo_name}/tensorboard/{env_name}/train/"

    # env = make_vec_env(env_name, n_envs=4)
    env = Monitor(gym.make(env_name))
    
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
    
    model = algo('MlpPolicy', env, verbose=0, learning_rate=lr, tensorboard_log=tb_path,
                 action_noise=action_noise)

    with ProgressBarManager(n_steps) as callback:
        model.learn(total_timesteps=n_steps, callback=callback, tb_log_name=f"{save_name}_lr{lr}")
    model.save(model_save_path)
    
if __name__ == "__main__":
    main()