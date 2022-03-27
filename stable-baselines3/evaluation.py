import os
import gym
import matplotlib.pyplot as plt
import numpy as np 
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from sb3_settings import *
plt.style.use('bmh')

def main():
    n_eval_ep = 100
    
    print('-'*50)
    print("***** Evaluation *****")
    print(f"RL-Algorithm: {algo_name}")
    print(f"Environment: {env_name}")
    print(f"Episodes: {n_eval_ep}")
    print('-'*50)

    save_plot_path = f"{algo_name}/evaluation/"
    os.makedirs(save_plot_path, exist_ok=True)
    save_plot_name = f"{algo_name}_{env_name}.png"
    model_path = f"{algo_name}/models/{algo_name}_{env_name}"
    model = algo.load(model_path)

    env = Monitor(gym.make(env_name))
    rewards, ep_lens = evaluate_policy(model, env, 
                                    n_eval_episodes=n_eval_ep, 
                                    deterministic=True, 
                                    return_episode_rewards=True, 
                                    render=False)

    mean_rewards = np.mean(rewards)
    std_rewards = np.std(rewards)

    mean_ep_lens = np.mean(ep_lens)
    std_ep_lens = np.std(ep_lens)

    fig, ax = plt.subplots(2, 1, figsize=(12,8))

    ax[0].plot(rewards, color='tab:blue')
    ax[0].axhline(mean_rewards, 0, n_eval_ep, color='k', linestyle='--')
    
    ax[1].plot(ep_lens, color='tab:orange')
    ax[1].axhline(mean_ep_lens, 0, n_eval_ep, color='k', linestyle='--')
    
    fig.suptitle(f"{algo_name} | {env_name} | Evaluation",fontsize=20, fontweight='bold')
    ax[0].set_title(f"Rewards: {mean_rewards:.2f} +/- {std_rewards:.2f}")
    ax[1].set_title(f"Episode length: {mean_ep_lens:.2f} +/- {std_ep_lens:.2f}")

    plt.tight_layout()
    fig.savefig(os.path.join(save_plot_path, save_plot_name), dpi=300)

if __name__ == "__main__":
    main()
