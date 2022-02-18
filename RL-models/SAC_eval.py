import os 

import gym 
import numpy as np 
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

import rlpy as rl 

plt.style.use('seaborn-whitegrid')
plt.rcParams['figure.figsize'] = (12,10)

def main():
    # set environment
    env_name = 'Pendulum-v0'
    env = gym.make(env_name)

    NUM_STATES = env.observation_space.shape[0]
    NUM_ACTIONS = env.action_space.shape[0]
    ACTION_BOUNDS = (env.action_space.low, env.action_space.high)

    print(env_name)
    print(f"Number of States: {NUM_STATES}\nNumber of Actions: {NUM_ACTIONS}")
    print("Max steps: ", env._max_episode_steps)
    
    # save path
    MAIN_PATH = 'SAC-models'
    if not os.path.exists(MAIN_PATH): os.mkdir(MAIN_PATH)

    PATH = os.path.join(MAIN_PATH, env_name)
    if not os.path.exists(PATH): os.mkdir(PATH)
    
    POLICY_HIDDEN_DIMS = (256,256)
    POLICY_LEARNING_RATE = 0.0005
    POLICY_PATH = os.path.join(PATH, f"SAC_policy_{env_name}_lr{POLICY_LEARNING_RATE}_{POLICY_HIDDEN_DIMS}.pt")

    # evaluation
    EVAL_EPISODES = 100
    eval_rewards = np.zeros(EVAL_EPISODES)
    eval_steps = np.zeros(EVAL_EPISODES)
    model =  rl.GaussianPolicyNetwork(NUM_STATES, ACTION_BOUNDS, hidden_dims=POLICY_HIDDEN_DIMS)
    model.load_state_dict(torch.load(POLICY_PATH))
    model.eval()

    for ep in tqdm(range(EVAL_EPISODES), desc='Evaluation'):
        state = env.reset()
        done = False
        ep_rewards = 0
        ep_steps = 0
        
        while not done:
            action = model.select_greedy_action(state)
            next_state, reward, done, _ = env.step(action)
            ep_steps += 1
            ep_rewards += reward
            state = next_state
        
        eval_rewards[ep] = ep_rewards
        eval_steps[ep] = ep_steps

    plt.figure()
    avg_rewards = rl.get_average(eval_rewards, int(0.1*EVAL_EPISODES))
    plt.subplot(2, 1, 1)
    plt.plot(avg_rewards, color='tab:red', linewidth=3)
    plt.ylabel("Rewards", fontsize=14)
    title = f"SAC_{env_name}_Evaluation"
    plt.title(title.replace('_',' | '), fontsize=16, fontweight='bold')

    avg_steps = rl.get_average(eval_steps, int(0.1*EVAL_EPISODES))
    plt.subplot(2, 1, 2)
    plt.plot(avg_steps, color='tab:red', linewidth=3)
    plt.ylabel("TimeSteps", fontsize=14)
    plt.xlabel("Episodes", fontsize=14)

    plt.tight_layout()
    plt.savefig(os.path.join(PATH, title+'.png') ,dpi=300)
    plt.cla()
    plt.clf()
    plt.close()

    env.close()
    
    
if __name__ == '__main__':
    main()