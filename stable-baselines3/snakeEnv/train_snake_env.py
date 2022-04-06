from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env

from sb3_utils import *
from snake_env import *

# env = SnakeGameEnv()
# check_env(env, warn=True)

env = make_vec_env(SnakeGameEnv, n_envs=16)
tb_path = 'snake_game/PPO/'
model = PPO('MlpPolicy', env, verbose=0, tensorboard_log=tb_path)
n_train_steps = 1e6
n_train_times = 10

for i in tqdm(range(n_train_times)):
    with ProgressBarManager(n_train_steps*n_train_times) as callback:
        model.learn(total_timesteps=n_train_steps, 
                    callback=callback, 
                    tb_log_name=f'PPO_{int(n_train_steps)}x{int(n_train_times)}', 
                    reset_num_timesteps=False)
    model.save(f"snake_game/ppo_snake_game_{int(n_train_steps)}x{i+1}")
