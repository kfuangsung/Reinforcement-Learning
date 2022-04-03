import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from snake_env import *

plt.style.use('bmh')

env = Monitor(SnakeGameEnv())
model = PPO.load("snake_game/ppo_snake_game_2")
n_eval_ep = 100
algo_name = 'PPO'
env_name = 'SnakeGameEnv'

rewards, ep_lens = evaluate_policy(model, env, 
                                   n_eval_episodes=n_eval_ep, 
                                   deterministic=True, 
                                   return_episode_rewards=True, 
                                   render=False)

mean_rewards = np.mean(rewards)
std_rewards = np.std(rewards)

mean_ep_lens = np.mean(ep_lens)
std_ep_lens = np.std(ep_lens)

print(f"Rewards: {mean_rewards:.2f} +/- {std_rewards:.2f}")
print(f"Episode length: {mean_ep_lens:.2f} +/- {std_ep_lens:.2f}")

fig, ax = plt.subplots(2, 1, figsize=(12,8))

ax[0].plot(rewards, color='tab:blue')
ax[0].axhline(mean_rewards, 0, n_eval_ep, color='k', linestyle='--')

ax[1].plot(ep_lens, color='tab:orange')
ax[1].axhline(mean_ep_lens, 0, n_eval_ep, color='k', linestyle='--')

fig.suptitle(f"{algo_name} | {env_name} | Evaluation",fontsize=20, fontweight='bold')
ax[0].set_title(f"Rewards: {mean_rewards:.2f} +/- {std_rewards:.2f}")
ax[1].set_title(f"Episode length: {mean_ep_lens:.2f} +/- {std_ep_lens:.2f}")

plt.tight_layout()
fig.savefig(f'./{algo_name}_{env_name}.png', dpi=200)