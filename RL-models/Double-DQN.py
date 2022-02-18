import copy 

import gym 
import numpy as np
import matplotlib.pyplot as plt 
import torch
import torch.nn as nn 
import torch.optim as optim 
from tqdm import tqdm

import rlpy as rl 

plt.rcParams['figure.figsize'] = (14,6)
plt.style.use('seaborn-whitegrid')


env_name = "CartPole-v1"
env = gym.make(env_name)
NUM_ACTIONS = env.action_space.n
NUM_STATES = env.observation_space.shape[0]
print(f"Number of Actions: {NUM_ACTIONS}\nNumber of States: {NUM_STATES}")

NUM_EPISODES = 500
EPS_START = 1
EPS_END = 0.01
EPS_DECAY_RATE = 0.01
GAMMA = 0.999
MEMORY_SIZE = 100000
BATCH_SIZE = 128
NUM_WARMUP_BATCHES = 5
UPDATE_TARGET_STEPS = 10
LEARNING_RATE = 0.00025
HIDDEN_DIMS = (1024, 512)

epsilon_values = rl.exp_decay_schedule(EPS_START, EPS_END, NUM_EPISODES, EPS_DECAY_RATE)
replay_buffer = rl.ReplayBuffer(MEMORY_SIZE)
policy_network = rl.QNetwork(NUM_STATES, NUM_ACTIONS, HIDDEN_DIMS)
target_network = copy.deepcopy(policy_network)
optimizer = optim.RMSprop(policy_network.parameters(), lr=LEARNING_RATE)
criterion = nn.MSELoss()
returns = np.zeros(NUM_EPISODES)

for ep in tqdm(range(NUM_EPISODES)):
    state = env.reset()
    done = False
    ep_return = 0
    epsilon = epsilon_values[ep]
    steps = 0
    
    while not done:
        # interact with environment
        steps += 1
        action = rl.select_action_from_network(state, epsilon, policy_network)
        next_state, reward, done, info = env.step(action)
        ep_return += reward
        experience = (state, action, reward, next_state, done)
        replay_buffer.store(experience)
        state = next_state
        
        #sampling from replay buffer
        if len(replay_buffer) > (BATCH_SIZE * NUM_WARMUP_BATCHES):
            samples = replay_buffer.sample(BATCH_SIZE)
            samples = policy_network.transform_experiences(samples)
            
            # optimize policy network
            states, actions, rewards, next_states, dones = samples
            # select action from policy network
            argmax_q_policy = policy_network(next_states).max(1)[1] # max() --> no need to detach
            # use Q-values from target network
            q_values_target = target_network(next_states).detach()
            max_q_target = q_values_target[np.arange(BATCH_SIZE), argmax_q_policy].unsqueeze(1)
            target = rewards + (GAMMA * max_q_target * torch.logical_not(dones))
            q_sa = policy_network(states).gather(1, actions)
            loss = criterion(q_sa, target)
            
            optimizer.zero_grad()
            loss.backward()
#             for param in policy_network.parameters():
#                 param.grad.data.clamp_(-1, 1)
            optimizer.step()
            
        # update target network
        if steps % UPDATE_TARGET_STEPS == 0:
            target_network.load_state_dict(policy_network.state_dict())
            
    returns[ep] = ep_return
    
    if (ep+1) % int(NUM_EPISODES*0.05) == 0:
        window = max(0, int((ep+1)-(NUM_EPISODES*0.1)))
        avg_ret = returns[window:(ep+1)].mean()
        print(f"Episodes: {ep+1}/{NUM_EPISODES} | Epsilon: {epsilon:.2f} | Average Return: {avg_ret:.2f}")

avg_returns = rl.get_average(returns, int(NUM_EPISODES*0.1))

fig, ax = plt.subplots()
line1, = ax.plot(avg_returns, label='Average Returns', color='tab:blue', linewidth=3)
ax.set_title(f"Double-DQN {env_name}", fontdict={'size':18, 'weight':'bold'})
ax.set_xlabel('Episode', fontsize=15)
ax.set_ylabel('Average Returns', fontsize=15)

ax2 = plt.twinx()
line2, = ax2.plot(epsilon_values, label='Epsilon', color='tab:orange', linestyle='--')
ax2.set_ylabel('Epsilon', fontsize=15)

plt.legend(handles=[line1,line2], frameon=True, loc='upper left', fontsize=12)
plt.tight_layout()
plt.show()




