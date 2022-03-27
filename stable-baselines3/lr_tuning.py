import os 
import gym
import numpy as np
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise
from sb3_settings import *
from sb3_utils import *

def main():
    tb_path = f'./{algo_name}/tensorboard/{env_name}/lr_tuning/'
    os.makedirs(tb_path, exist_ok=True)
    n_steps = 1e4
    learning_rates = [2.5*(10**-i) for i in range(1, 6)]

    print('-'*50)
    print("***** Learning Rates - Tuning *****")
    print(f"RL-Algorithm: {algo_name}")
    print(f"Environment: {env_name}")
    print(f"Learning rates: {learning_rates}")
    print('-'*50)

    env = Monitor(gym.make(env_name))
    
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
    
    for lr in tqdm(learning_rates):
        model = algo('MlpPolicy', env, verbose=0, tensorboard_log=tb_path, learning_rate=lr,
                     action_noise=action_noise)
        with ProgressBarManager(n_steps) as callback:
            model.learn(total_timesteps=n_steps, callback=callback, tb_log_name=f'lr{lr}')
            
if __name__ == '__main__':
    main()