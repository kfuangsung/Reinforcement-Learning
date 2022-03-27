from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from left_env import *
from sb3_utils import *

env = GoLeftEnv(grid_size=20)
check_env(env, warn=True)

model = PPO('MlpPolicy', env, verbose=0)
n_train_steps = 1000
with ProgressBarManager(n_train_steps) as callback:
    model.learn(total_timesteps=n_train_steps, callback=callback)

n_steps = 50
state = env.reset()
env.render()
for step in range(n_steps):
    action, _ = model.predict(state, deterministic=True)
    next_state, reward, done, info = env.step(action)
    print(f"Steps:{step+1}, action={action}, Reward: {reward}, done={done}")
    env.render()
    state = next_state
    if done:
        break