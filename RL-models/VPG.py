import os 

import gym 
import numpy as np
import torch
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt 
from tqdm.notebook import tqdm

import rlpy as rl 

plt.style.use('seaborn-whitegrid')
plt.rcParams['figure.figsize'] = (12,6)

env_name = 'LunarLander-v2'
env = gym.make(env_name)
NUM_ACTIONS = env.action_space.n
NUM_STATES = env.observation_space.shape[0]
print(f"Number of Actions: {NUM_ACTIONS}\nNumber of States: {NUM_STATES}")

# VPG
NUM_EPISODES = 5000
GAMMA = 0.999
ENTROPY_LOSS_WEIGHT = 0.001

POLICY_HIDDEN_DIMS = (128, 64)
POLICY_LEARNING_RATE = 0.0005
POLICY_MAX_GRAD_NORM = 1
policy_network = rl.DiscreteActionPolicyNetwork(NUM_STATES, NUM_ACTIONS, POLICY_HIDDEN_DIMS)
policy_optimizer = optim.Adam(policy_network.parameters(), lr=POLICY_LEARNING_RATE)

VALUE_HIDDEN_DIMS = (256, 128)
VALUE_LEARNING_RATE = 0.0007
VALUE_MAX_GRAD_NORM = float('inf')
value_network =rl.StateValueNetwork(NUM_STATES, VALUE_HIDDEN_DIMS)
value_optimizer = optim.RMSprop(value_network.parameters(), lr=VALUE_LEARNING_RATE)

MAIN_PATH = 'VPG-models'
if not os.path.exists(MAIN_PATH):
    os.mkdir(MAIN_PATH)
    
PATH = os.path.join(MAIN_PATH, env_name)
if not os.path.exists(PATH):
    os.mkdir(PATH)

episode_rewards = np.zeros(NUM_EPISODES)
for ep in tqdm(range(NUM_EPISODES)):
    state = env.reset()
    done = False
    rewards = []
    logpas = []
    entropies = []
    values = []
    
    while not done:
        action, logpa, entropy = policy_network.full_pass(state)
        next_state, reward, done, info = env.step(action)
        value = value_network(state)
        
        rewards.append(reward)
        logpas.append(logpa)
        entropies.append(entropy)
        values.append(value)
        
        state = next_state
        
    episode_rewards[ep] = np.sum(rewards)
        
    T = len(rewards)
    discounts = np.logspace(0, T, num=T, base=GAMMA, endpoint=False)
    returns = np.array([np.sum(discounts[:T-t] * rewards[t:]) for t in range(T)])
    
    discounts = torch.tensor(discounts, dtype=torch.float32)
    returns = torch.tensor(returns, dtype=torch.float32)
    
    logpas = torch.cat(logpas)
    entropies = torch.cat(entropies)
    values = torch.cat(values)
    
    value_error = returns - values
    #Policy network optimize
    # add negative to do gradient ascent
    policy_loss = -(discounts * value_error.detach() * logpas).mean()
    entropy_loss = entropies.mean()
    loss = policy_loss + ENTROPY_LOSS_WEIGHT * entropy_loss
    policy_optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy_network.parameters(), POLICY_MAX_GRAD_NORM)
    policy_optimizer.step()
    
    # Value network optimize
    value_loss = value_error.pow(2).mul(0.5).mean()
    value_optimizer.zero_grad()
    value_loss.backward()
    torch.nn.utils.clip_grad_norm_(value_network.parameters(), VALUE_MAX_GRAD_NORM)
    value_optimizer.step()
    
    # save model
    if (ep+1) % max(1, int(0.1*NUM_EPISODES)):
        policy_path = os.path.join(PATH, f"VPG_policy_{env_name}_lr{POLICY_LEARNING_RATE}_{POLICY_HIDDEN_DIMS}")
        value_path = os.path.join(PATH, f"VPG_value_{env_name}_lr{VALUE_LEARNING_RATE}_{VALUE_HIDDEN_DIMS}")
        
        torch.save(policy_network.state_dict(), policy_path)
        torch.save(value_network.state_dict(), value_path)

print(f"Training rewards mean: {episode_rewards.mean():.2f}")
print(f"Training rewards std: {episode_rewards.std():.2f}")

avg_rewards = rl.get_average(episode_rewards, int(0.1*NUM_EPISODES))
plt.plot(avg_rewards, color='tab:blue', linewidth=3)
title = f'VPG | {env_name} | Average Rewards | Training'
plt.title(title, fontsize=16, fontweight='bold')
plt.xlabel('Episodes', fontsize=14)
plt.ylabel('Rewards', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(PATH, title+'.png'), dpi=300)
# plt.show()

# Evaluation
model = rl.DiscreteActionPolicyNetwork(NUM_STATES, NUM_ACTIONS, POLICY_HIDDEN_DIMS)
model.load_state_dict(torch.load(policy_path))
model.eval()

NUM_EVAL = 100
eval_rewards = np.zeros(NUM_EVAL)

for ep in tqdm(range(NUM_EVAL)):
    state = env.reset()
    done = False
    rewards = 0
    
    while not done:
        with torch.no_grad():
            action = model.select_greedy_action(state)
        next_state, reward, done, info = env.step(action)
        rewards += reward
        state = next_state
    
    eval_rewards[ep] = rewards

print(f"Evaluation rewards mean: {eval_rewards.mean():.2f}")
print(f"Evaluation rewards std: {eval_rewards.std():.2f}")

avg_rewards = rl.get_average(eval_rewards, int(0.1*NUM_EVAL))
plt.plot(avg_rewards, color='tab:red', linewidth=3)
title = f'VPG | {env_name} | Average Rewards | Evaluation'
plt.title(title, fontsize=16, fontweight='bold')
plt.xlabel('Episodes', fontsize=14)
plt.ylabel('Rewards', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(PATH, title+'.png'), dpi=300)
# plt.show()

render = False
if render:
    for i in range(3):    
        state = env.reset()
        done = False
        env.render()
        rewards = 0

        while not done:
            with torch.no_grad():
                action = model.select_greedy_action(state)
            next_state, reward, done, info = env.step(action)
            rewards += reward
            env.render()
            state = next_state

        print(f"{i} | Rewards: {rewards}")


