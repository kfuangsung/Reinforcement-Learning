from stable_baselines3 import PPO
from sb3_utils import *
from snake_env import *

def main():

    learning_rates = [2.5*(10**-i) for i in range(1, 6)]
    env = SnakeGameEnv()
    tb_path = 'snake_game/PPO/'
    n_train_steps = 1e5

    for lr in tqdm(learning_rates):
        model = PPO('MlpPolicy', env, verbose=0, tensorboard_log=tb_path, learning_rate=lr)
        with ProgressBarManager(n_train_steps) as callback:
            model.learn(total_timesteps=n_train_steps, callback=callback, tb_log_name=f'PPO_{lr}')
            
if __name__ == '__main__':
    main()