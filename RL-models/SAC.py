import os 
import copy

import gym 
import numpy as np 
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm

import rlpy as rl 

plt.style.use('seaborn-whitegrid')
plt.rcParams['figure.figsize'] = (12,10)

torch.autograd.set_detect_anomaly(True)

def main():
    
    # set environment
    env_name = 'MountainCarContinuous-v0'
    env = gym.make(env_name)

    NUM_STATES = env.observation_space.shape[0]
    NUM_ACTIONS = env.action_space.shape[0]
    ACTION_BOUNDS = (env.action_space.low, env.action_space.high)

    print(env_name)
    print(f"Number of States: {NUM_STATES}\nNumber of Actions: {NUM_ACTIONS}")
    print("Max steps: ", env._max_episode_steps)

    GAMMA = 0.99
    TAU = 0.005
    NUM_EPISODES = 200

    # save path
    MAIN_PATH = 'SAC-models'
    if not os.path.exists(MAIN_PATH): os.mkdir(MAIN_PATH)

    PATH = os.path.join(MAIN_PATH, env_name)
    if not os.path.exists(PATH): os.mkdir(PATH)

    # set policy network
    POLICY_HIDDEN_DIMS = (256,256)
    POLICY_LEARNING_RATE = 0.0005
    POLICY_MAX_GRAD_NORM = float('inf')
    policy_network = rl.GaussianPolicyNetwork(NUM_STATES, ACTION_BOUNDS, hidden_dims=POLICY_HIDDEN_DIMS)
    policy_optimizer = optim.Adam(policy_network.parameters(), lr=POLICY_LEARNING_RATE)
    POLICY_PATH = os.path.join(PATH, f"SAC_policy_{env_name}_lr{POLICY_LEARNING_RATE}_{POLICY_HIDDEN_DIMS}.pt")

    # set value network
    VALUE_HIDDEN_DIMS = (256,256)
    VALUE_LEARNING_RATE = 0.0007
    VALUE_MAX_GRAD_NORM = float('inf')
    UPDATE_TARGET_EVERY_STEP = 1

    online_value_network_a = rl.QValueSACNetwork(NUM_STATES, NUM_ACTIONS, VALUE_HIDDEN_DIMS)
    target_value_network_a = copy.deepcopy(online_value_network_a)
    value_optimizer_a = optim.Adam(online_value_network_a.parameters(), lr=VALUE_LEARNING_RATE)
    VALUE_PATH_A = os.path.join(PATH, f"SAC_value_a_{env_name}_lr{VALUE_LEARNING_RATE}_{VALUE_HIDDEN_DIMS}.pt")

    online_value_network_b = rl.QValueSACNetwork(NUM_STATES, NUM_ACTIONS, VALUE_HIDDEN_DIMS)
    target_value_network_b = copy.deepcopy(online_value_network_b)
    value_optimizer_b = optim.Adam(online_value_network_b.parameters(), lr=VALUE_LEARNING_RATE)
    VALUE_PATH_B = os.path.join(PATH, f"SAC_value_b_{env_name}_lr{VALUE_LEARNING_RATE}_{VALUE_HIDDEN_DIMS}.pt")

    # replay buffer
    MEMORY_SIZE = 100000
    BATCH_SIZE = 64
    NUM_WARMUP_BATHCES = 10
    replay_buffer = rl.ReplayBuffer(MEMORY_SIZE)

    episode_rewards = np.zeros(NUM_EPISODES)
    episode_steps = np.zeros(NUM_EPISODES)

    for ep in tqdm(range(NUM_EPISODES), desc="Training"):
        state = env.reset()
        done = False
        ep_reward = 0
        ep_step = 0
        
        while not done:
            min_samples = BATCH_SIZE * NUM_WARMUP_BATHCES
            
            if len(replay_buffer) < min_samples:
                action = policy_network.select_random_action(state)
            else:
                action = policy_network.select_action(state)
                
            next_state, reward, done, _ = env.step(action)
            experience = (state, action, reward, next_state, done)
            replay_buffer.store(experience)
            ep_reward += reward
            ep_step += 1
            state = next_state
            
            if len(replay_buffer) > min_samples:
                samples = replay_buffer.sample(BATCH_SIZE)
                samples = online_value_network_a.format_experiences(samples)
                states, actions, rewards, next_states, dones = samples
                
                current_actions, logpi_s, _ = policy_network.full_pass(states)
                target_alpha = (logpi_s + policy_network.target_entropy).detach()
                alpha_loss = -(policy_network.logalpha * target_alpha).mean()
                policy_network.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                policy_network.alpha_optimizer.step()
                alpha = policy_network.logalpha.exp()
                
                current_q_sa_a = online_value_network_a(states, current_actions)
                current_q_sa_b = online_value_network_b(states, current_actions)
                current_q_sa = torch.min(current_q_sa_a, current_q_sa_b)
                policy_loss = (alpha * logpi_s - current_q_sa).mean()
                
                policy_optimizer.zero_grad()
                policy_loss.backward()
                torch.nn.utils.clip_grad_norm_(policy_network.parameters(), POLICY_MAX_GRAD_NORM)
                policy_optimizer.step()
                
                ap, logpi_sp, _ = policy_network.full_pass(next_states)
                q_spap_a = target_value_network_a(next_states, ap)
                q_spap_b = target_value_network_b(next_states, ap)
                q_spap = torch.min(q_spap_a, q_spap_b) - alpha * logpi_sp
                target_q_sa = (rewards + GAMMA * q_spap * torch.logical_not(dones)).detach()
                q_sa_a = online_value_network_a(states, actions)
                q_sa_b = online_value_network_b(states, actions)
                qa_loss = (q_sa_a - target_q_sa).pow(2).mul(0.5).mean()
                qb_loss = (q_sa_b - target_q_sa).pow(2).mul(0.5).mean()
                
                value_optimizer_a.zero_grad()
                qa_loss.backward()
                torch.nn.utils.clip_grad_norm_(online_value_network_a.parameters(), VALUE_MAX_GRAD_NORM)
                value_optimizer_a.step()
                
                value_optimizer_b.zero_grad()
                qb_loss.backward()
                torch.nn.utils.clip_grad_norm_(online_value_network_b.parameters(), VALUE_MAX_GRAD_NORM)
                value_optimizer_b.step()
                
                
            if ep_step % UPDATE_TARGET_EVERY_STEP == 0:
                for target, online in zip(target_value_network_a.parameters(), online_value_network_a.parameters()):
                    target_ratio = (1 - TAU) * target.data
                    online_ratio = TAU * online.data
                    mixed_weights = target_ratio + online_ratio
                    target.data.copy_(mixed_weights)
                    
                for target, online in zip(target_value_network_b.parameters(), online_value_network_b.parameters()):
                    target_ratio = (1 - TAU) * target.data
                    online_ratio = TAU * online.data
                    mixed_weights = target_ratio + online_ratio
                    target.data.copy_(mixed_weights)
                    
        episode_rewards[ep] = ep_reward
        episode_steps[ep] = ep_step
            
        # save models
        torch.save(policy_network.state_dict(), POLICY_PATH)
        torch.save(online_value_network_a.state_dict(), VALUE_PATH_A)
        torch.save(online_value_network_b.state_dict(), VALUE_PATH_B)

        # plot training results
    plt.figure()
    avg_rewards = rl.get_average(episode_rewards, int(0.1*NUM_EPISODES))
    plt.subplot(2, 1, 1)
    plt.plot(avg_rewards, color='tab:blue', linewidth=3)
    plt.ylabel("Rewards", fontsize=14)
    title = f"SAC_{env_name}_Training"
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