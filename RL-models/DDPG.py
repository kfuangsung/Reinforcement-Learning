import os 
import copy

import gym 
import numpy as np
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim 
import matplotlib.pyplot as plt
from tqdm import tqdm

import rlpy as rl 

plt.style.use('seaborn-whitegrid')
plt.rcParams['figure.figsize'] = (12,10)


def main():
    # DDPG --> continuous actions
    env_name = 'Pendulum-v0'
    env = gym.make(env_name)
    NUM_ACTIONS = env.action_space.shape[0]
    NUM_STATES = env.observation_space.shape[0]
    print(env_name)
    print(f"Number of States: {NUM_STATES}\nNumber of Actions: {NUM_ACTIONS}")

    NUM_EPISODES = 500
    GAMMA = 0.999
    TAU = 0.005
    ACTION_BOUNDS = (env.action_space.low, env.action_space.high)
    MEMORY_SIZE = 100000
    BATCH_SIZE = 256
    NUM_WARMUP_BATCHES = 5
    UPADATE_TARGET_EVERY_STEP = 1

    MAIN_PATH = 'DDPG-models'
    if not os.path.exists(MAIN_PATH):
        os.mkdir(MAIN_PATH)

    PATH = os.path.join(MAIN_PATH, env_name)
    if not os.path.exists(PATH):
        os.mkdir(PATH)

    training_strategy = rl.NormalNoiseStrategy(ACTION_BOUNDS)
    replay_buffer = rl.ReplayBuffer(MEMORY_SIZE)

    IS_RESUME_TRAINING = False

    POLICY_HIDDEN_DIMS = (256, 256)
    POLICY_MAX_GRAD_NORM = float('inf')
    POLICY_LEARNING_RATE = 0.0003
    online_policy_network = rl.DeterministicPolicyNetwork(NUM_STATES, ACTION_BOUNDS, POLICY_HIDDEN_DIMS)

    policy_path = os.path.join(PATH, f"DDPG_policy_{env_name}_lr{POLICY_LEARNING_RATE}_{POLICY_HIDDEN_DIMS}.pt")
    if os.path.exists(policy_path) and IS_RESUME_TRAINING:
        online_policy_network.load_state_dict(torch.load(policy_path))
        print('Resume traning Policy network')

    target_policy_network = copy.deepcopy(online_policy_network)
    policy_optimizer = optim.Adam(online_policy_network.parameters(), lr=POLICY_LEARNING_RATE)

    VALUE_HIDDEN_DIMS = (256, 256)
    VALUE_MAX_GRAD_NORM = float('inf')
    VALUE_LEARNING_RATE = 0.0003
    online_value_network = rl.QValueNetwork(NUM_STATES, NUM_ACTIONS, VALUE_HIDDEN_DIMS)

    value_path = os.path.join(PATH, f"DDPG_value_{env_name}_lr{VALUE_LEARNING_RATE}_{VALUE_HIDDEN_DIMS}.pt")
    if os.path.exists(value_path) and IS_RESUME_TRAINING:
        online_value_network.load_state_dict(torch.load(value_path))
        print('Resume training Value network')

    target_value_network = copy.deepcopy(online_value_network)
    value_optimizer = optim.Adam(online_value_network.parameters(), lr=VALUE_LEARNING_RATE)

    episode_rewards = np.zeros(NUM_EPISODES)
    episode_steps = np.zeros(NUM_EPISODES)
    for ep in tqdm(range(NUM_EPISODES), desc='Training'):
        state = env.reset()
        done = False
        ep_rewards = 0
        ep_steps = 0
        
        while not done:
            # interact with environment
            min_samples = NUM_WARMUP_BATCHES * BATCH_SIZE
            is_max_explore = len(replay_buffer) < min_samples
            action = training_strategy.select_action(online_policy_network, state, is_max_explore)
            next_state, reward, done, _ = env.step(action)
            ep_steps += 1
            ep_rewards += reward
            experience = (state, action, reward, next_state, done)
            replay_buffer.store(experience)
            state = next_state
            
            if len(replay_buffer) > min_samples:
                # sampling experiences
                samples = replay_buffer.sample(BATCH_SIZE)
                samples = online_value_network.format_experiences(samples)
                states, actions, rewards, next_states, dones = samples
                
                # optimize value network
                argmax_a_q_sp = target_policy_network(next_states)
                max_a_q_sp = target_value_network(next_states, argmax_a_q_sp)
                target_q_sa = rewards + GAMMA * max_a_q_sp * torch.logical_not(dones)
                q_sa = online_value_network(states, actions)
                td_error = q_sa - target_q_sa.detach()
                value_loss = td_error.pow(2).mul(0.5).mean()
                value_optimizer.zero_grad()
                value_loss.backward()
                torch.nn.utils.clip_grad_norm_(online_value_network.parameters(), VALUE_MAX_GRAD_NORM)
                value_optimizer.step()
                
                # optimize policy network
                argmax_a_q_s = online_policy_network(states)
                max_a_q_s = online_value_network(states, argmax_a_q_s)
                policy_loss = -max_a_q_s.mean()
                policy_optimizer.zero_grad()
                policy_loss.backward()
                torch.nn.utils.clip_grad_norm_(online_policy_network.parameters(), POLICY_MAX_GRAD_NORM)
                policy_optimizer.step()
                
            # update target networks
            if ep_steps % UPADATE_TARGET_EVERY_STEP == 0:
                
                for target, online in zip(target_value_network.parameters(), online_value_network.parameters()):
                    target_ratio = (1 - TAU) * target.data
                    online_ratio = TAU * online.data
                    mixed_weights = target_ratio + online_ratio
                    target.data.copy_(mixed_weights)
                
                for target, online in zip(target_policy_network.parameters(), online_policy_network.parameters()):
                    target_ratio = (1 - TAU) * target.data
                    online_ratio = TAU * online.data
                    mixed_weights = target_ratio + online_ratio
                    target.data.copy_(mixed_weights)
        
        episode_rewards[ep] = ep_rewards
        episode_steps[ep] = ep_steps   
                
        # save models
        torch.save(online_policy_network.state_dict(), policy_path)
        torch.save(online_value_network.state_dict(), value_path)

    plt.figure()
    avg_rewards = rl.get_average(episode_rewards, int(0.1*NUM_EPISODES))
    plt.subplot(2, 1, 1)
    plt.plot(avg_rewards, color='tab:blue', linewidth=3)
    plt.ylabel("Rewards", fontsize=14)
    title = f"DDPG_{env_name}_Training"
    plt.title(title.replace('_',' | '), fontsize=16, fontweight='bold')

    avg_steps = rl.get_average(episode_steps, int(0.1*NUM_EPISODES))
    plt.subplot(2, 1, 2)
    plt.plot(avg_steps, color='tab:blue', linewidth=3)
    plt.ylabel("TimeSteps", fontsize=14)
    plt.xlabel("Episodes", fontsize=14)

    plt.tight_layout()
    plt.savefig(os.path.join(PATH, title+'.png') ,dpi=300)
    plt.cla()
    plt.clf()
    plt.close()
    
    # Evaluation
    
    EVAL_EPISODES = 100
    eval_rewards = np.zeros(EVAL_EPISODES)
    eval_steps = np.zeros(EVAL_EPISODES)
    model =  rl.DeterministicPolicyNetwork(NUM_STATES, ACTION_BOUNDS, POLICY_HIDDEN_DIMS)
    model.load_state_dict(torch.load(policy_path))
    model.eval()
    eval_strategy = rl.GreedyStrategy(ACTION_BOUNDS)

    for ep in tqdm(range(EVAL_EPISODES), desc='Evaluation'):
        state = env.reset()
        done = False
        ep_rewards = 0
        ep_steps = 0
        
        while not done:
            action = eval_strategy.select_action(model, state)
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
    title = f"DDPG_{env_name}_Evaluation"
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