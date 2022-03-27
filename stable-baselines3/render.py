import gym
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from sb3_settings import *

def main():
    print('-'*50)
    print("***** Rendering *****")
    print(f"RL-Algorithm: {algo_name}")
    print(f"Environment: {env_name}")
    print('-'*50)

    model_path = f"{algo_name}/models/{algo_name}_{env_name}"
    model = algo.load(model_path)

    env = Monitor(gym.make(env_name))
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=5, deterministic=True, render=True)
    print(f"Rewards: {mean_reward} +/- {std_reward}")
    
if __name__ == "__main__":
    main()