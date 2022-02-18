import os 

import gym 
import numpy as np
import torch
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg
from tqdm.notebook import tqdm

import rlpy as rl 

plt.style.use('seaborn-whitegrid')
plt.rcParams['figure.figsize'] = (12,6)

env_name = 'CartPole-v1'
env = gym.make(env_name)
NUM_ACTIONS = env.action_space.n
NUM_STATES = env.observation_space.shape[0]
print(f"Number of Actions: {NUM_ACTIONS}\nNumber of States: {NUM_STATES}")

# REINFORCE

NUM_EPISODES = 10000
GAMMA = 0.99
HIDDEN_DIMS = (128, 64)

MAIN_PATH = 'REINFORCE-models'
if not os.path.exists(MAIN_PATH):
    os.mkdir(MAIN_PATH)
    
PATH = os.path.join(MAIN_PATH, env_name)
if not os.path.exists(PATH):
    os.mkdir(PATH)
    
lr_rates = [5*(10**-i) for i in range(1,6)]
print(lr_rates)
train_stats = {}

for lr in tqdm(lr_rates):
    policy_network = rl.DiscreteActionPolicyNetwork(NUM_STATES, NUM_ACTIONS, HIDDEN_DIMS)
    optimizer = optim.Adam(policy_network.parameters(), lr=lr)
    episode_rewards = np.zeros(NUM_EPISODES)
    episode_steps = np.zeros(NUM_EPISODES)
    
    for ep in tqdm(range(NUM_EPISODES), desc=f"Learning Rate={lr}", leave=False):
        state = env.reset()
        done = False
        rewards = []
        logpas = []
        steps = 0

        while not done:
            action, logpa = policy_network.full_pass(state)
            next_state, reward, done, info = env.step(action)
            steps += 1
            rewards.append(reward)
            logpas.append(logpa)
            state = next_state

        # need to wait till episode is done before optimize
        T = len(rewards)
        discounts = np.logspace(0, T, num=T, endpoint=False, base=GAMMA)
        returns = np.array([np.sum(discounts[:T-t] * rewards[t:]) for t in range(T)])

        discounts = torch.tensor(discounts, dtype=torch.float32).unsqueeze(1)
        returns = torch.tensor(returns, dtype=torch.float32).unsqueeze(1)
        logpas = torch.cat(logpas)

        # add negative to do gradient ascent
        policy_loss = -(discounts * returns * logpas).mean()
        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()
        
        # save model 
        if (ep+1) % max(1, int(NUM_EPISODES*0.01)) == 0:
            filename = f"REINFORCE_{env_name}_lr{lr}_{HIDDEN_DIMS}.pt"
            save_path = os.path.join(PATH, filename)
            torch.save(policy_network.state_dict(), save_path)
            
        episode_rewards[ep] = np.sum(rewards)
        episode_steps[ep] = steps
    
    train_stats[lr] = {}
    train_stats[lr]['returns'] = episode_rewards
    train_stats[lr]['timesteps'] = episode_steps

    train_df = rl.get_stats_df(train_stats)
    rl.save_df_to_csv(train_df, f"REINFORCE_{env_name}_train_stats.csv", PATH)
    title = f"REINFORCE | {env_name} | Training"
    rl.plot_lr_stats(train_df, title, average=True, save_path=PATH)

rl.print_stats_table(train_df)

image_path = os.path.join(PATH, f"{title}.png")
img = mpimg.imread(image_path)
plt.figure(figsize=(12,8))
plt.imshow(img)
plt.axis(False)
plt.tight_layout()
plt.show()

# Evaluate models
EVAL_EPISODES = 100
eval_stats = {}

for lr in tqdm(lr_rates):
    filename = f"REINFORCE_{env_name}_lr{lr}_{HIDDEN_DIMS}.pt"
    save_path = os.path.join(PATH, filename)
    model = rl.DiscreteActionPolicyNetwork(NUM_STATES, NUM_ACTIONS, HIDDEN_DIMS)
    model.load_state_dict(torch.load(save_path))
    model.eval()
    
    episode_rewards = np.zeros(EVAL_EPISODES)
    episode_steps = np.zeros(EVAL_EPISODES)
    for ep in tqdm(range(EVAL_EPISODES), leave=False, desc=f"LearningRate={lr}"):
        state = env.reset()
        done = False
        rewards = 0
        steps = 0
        
        while not done:
            action = model.select_greedy_action(state)
            next_state, reward, done, info = env.step(action)
            rewards += reward
            steps += 1
            state = next_state
        
        episode_rewards[ep] = rewards
        episode_steps[ep] = steps
        
    eval_stats[lr] = {}
    eval_stats[lr]['returns'] = episode_rewards
    eval_stats[lr]['timesteps'] = episode_steps
    
    eval_df = rl.get_stats_df(eval_stats)
    rl.save_df_to_csv(eval_df, f"REINFORCE_{env_name}_eval_stats.csv", PATH)
    title = f"REINFORCE | {env_name} | Evaluation"
    rl.plot_lr_stats(eval_df, title, average=True, save_path=PATH)

rl.print_stats_table(eval_df)

image_path = os.path.join(PATH, f"{title}.png")
img = mpimg.imread(image_path)
plt.figure(figsize=(12,8))
plt.imshow(img)
plt.axis(False)
plt.tight_layout()
plt.show()


