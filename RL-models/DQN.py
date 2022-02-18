import os 
import copy

import gym 
import numpy as np 
import torch
import torch.nn as nn 
import torch.optim as optim
from tqdm import tqdm

import rlpy as rl 

def main():

    env_name = "MountainCar-v0"
    env = gym.make(env_name)
    NUM_ACTIONS = env.action_space.n
    NUM_STATES = env.observation_space.shape[0]
    print(env_name)
    print(f"Number of Actions: {NUM_ACTIONS}\nNumber of States: {NUM_STATES}")

    NUM_EPISODES = 500
    GAMMA = 0.99
    MEMORY_SIZE = 50000
    BATCH_SIZE = 64
    N_WARMUP_BATCHES = 5
    epsilon_values = rl.exp_decay_schedule(1, 0.1, NUM_EPISODES, 0.02)
    HIDDEN_DIMS = (1024,)
    UPDATE_TARGET_STEPS = 5
    
    MAIN_PATH = 'DQN-models'
    if not os.path.exists(MAIN_PATH): os.mkdir(MAIN_PATH)

    PATH = os.path.join(MAIN_PATH, env_name)
    if not os.path.exists(PATH): os.mkdir(PATH)
    
    lr_rates = [5*(10**-i) for i in range(1, 6)]
    train_stats = {}
    print(f"Learning Rates: {lr_rates}")
    
    for lr in tqdm(lr_rates, desc='Training'):
        replay_buffer = rl.ReplayBuffer(MEMORY_SIZE)
        policy_network = rl.QNetwork(NUM_STATES, NUM_ACTIONS, HIDDEN_DIMS)
        target_network = copy.deepcopy(policy_network)
        optimizer = optim.RMSprop(policy_network.parameters() ,lr=lr)
        criterion = nn.MSELoss()
        POLICY_PATH = os.path.join(PATH, f"DQN_{env_name}_lr{lr}_{HIDDEN_DIMS}.pt")
        episode_rewards = np.zeros(NUM_EPISODES)
        episode_steps = np.zeros(NUM_EPISODES)

        for ep in tqdm(range(NUM_EPISODES), desc=f"LearningRate:{lr}", leave=False):
            state = env.reset()
            done = False
            ep_reward = 0
            epsilon = epsilon_values[ep]
            ep_step = 0
            
            while not done:
                ep_step += 1
                action = rl.select_action_from_network(state, epsilon, policy_network)
                next_state, reward, done, info = env.step(action)
                ep_reward += reward
                experience = (state, action, reward, next_state, done)
                replay_buffer.store(experience)
                state = next_state
                
                # sampling experiences
                if len(replay_buffer) > (BATCH_SIZE * N_WARMUP_BATCHES):
                    samples = replay_buffer.sample(BATCH_SIZE)
                    samples = policy_network.transform_experiences(samples)
                    
                    # optimize policy_network
                    states, actions, rewards, next_states, dones = samples
                    max_q = target_network(next_states).detach().max(1)[0].unsqueeze(1)
                    target = rewards + (GAMMA * max_q * torch.logical_not(dones))
                    q_sa = policy_network(states).gather(1, actions)
                    loss = criterion(q_sa, target)
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                # update target network
                if ep_step % UPDATE_TARGET_STEPS == 0:
                    target_network.load_state_dict(policy_network.state_dict())
            
            episode_rewards[ep] = ep_reward
            episode_steps[ep] = ep_step
            
            torch.save(policy_network.state_dict(), POLICY_PATH)
            
        train_stats[lr] = {}
        train_stats[lr]['rewards'] = episode_rewards
        train_stats[lr]['timesteps'] = episode_steps

    train_df = rl.get_stats_df(train_stats)
    rl.save_df_to_csv(train_df, f"DQN_{env_name}_train_stats.csv", PATH)
    title = f"DQN_{env_name}_Training"
    rl.plot_lr_stats(train_df, title, average=True, epsilons=epsilon_values, save_path=PATH, show=False)
    rl.print_stats_table(train_df)
    
    # Evaluate models
    EVAL_EPISODES = 100
    eval_stats = {}

    for lr in tqdm(lr_rates, desc="Evaluation"):
        model = rl.QNetwork(NUM_STATES, NUM_ACTIONS, HIDDEN_DIMS)
        model.load_state_dict(torch.load(POLICY_PATH))
        model.eval()
        
        episode_rewards = np.zeros(EVAL_EPISODES)
        episode_steps = np.zeros(EVAL_EPISODES)
        for ep in tqdm(range(EVAL_EPISODES), leave=False, desc=f"LearningRate={lr}"):
            state = env.reset()
            done = False
            ep_reward = 0
            ep_step = 0
            
            while not done:
                ep_step += 1
                action = rl.select_action_from_network(state, 0, model)
                next_state, reward, done, info = env.step(action)
                ep_reward += reward
                state = next_state
            
            episode_rewards[ep] = ep_reward
            episode_steps[ep] = ep_step
            
        eval_stats[lr] = {}
        eval_stats[lr]['rewards'] = episode_rewards
        eval_stats[lr]['timesteps'] = episode_steps
        
    eval_df = rl.get_stats_df(eval_stats)
    rl.save_df_to_csv(eval_df, f"DQN_{env_name}_eval_stats.csv", PATH)
    title = f"DQN_{env_name}_Evaluation"
    rl.plot_lr_stats(eval_df, title, average=True, save_path=PATH, show=False)
    rl.print_stats_table(eval_df)


if __name__ == '__main__':
    main()
