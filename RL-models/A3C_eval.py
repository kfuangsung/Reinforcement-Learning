import os 

import gym
import numpy as np 
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

import rlpy as rl 

plt.rcParams['figure.figsize'] = (14,6)
plt.style.use('seaborn-whitegrid')

env_name = 'LunarLander-v2'
env = gym.make(env_name)
NUM_ACTIONS = env.action_space.n
NUM_STATES = env.observation_space.shape[0]
print(env_name)
print(f"Number of Actions: {NUM_ACTIONS}\nNumber of States: {NUM_STATES}")


NUM_EVAL = 100
MAIN_PATH = 'A3C-models'
PATH = os.path.join(MAIN_PATH, env_name)
POLICY_HIDDEN_DIMS = (128, 64)
POLICY_LEARNING_RATE = 0.0005
POLICY_PATH = os.path.join(PATH, f"A3C_polciy_{env_name}_lr{POLICY_LEARNING_RATE}_{POLICY_HIDDEN_DIMS}.pt")
eval_rewards = np.zeros(NUM_EVAL)
model = rl.DiscreteActionPolicyNetwork(NUM_STATES, NUM_ACTIONS, POLICY_HIDDEN_DIMS)
model.load_state_dict(torch.load(POLICY_PATH))
model.eval()

for ep in tqdm(range(NUM_EVAL), desc='Evaluation'):
    state = env.reset()
    done = False
    rewards = 0
    
    while not done:
        with torch.no_grad():
            action = model.select_greedy_action(state)
        next_state, reward, done, _ = env.step(action)
        rewards += reward
        state = next_state
    
    eval_rewards[ep] = rewards
    
avg_rewards = rl.get_average(eval_rewards, int(0.1*NUM_EVAL))
plt.plot(avg_rewards, color='tab:red', linewidth=3)
title = f"A3C | {env_name} | Average rewards | Evaluation"
plt.title(title, fontsize=16, fontweight='bold')
plt.xlabel("Episodes", fontsize=14)
plt.ylabel("Rewards", fontsize=14)
plt.savefig(os.path.join(PATH, title+'.png'), dpi=300)