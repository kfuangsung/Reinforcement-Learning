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

    env_name = 'LunarLander-v2'
    env = gym.make(env_name)
    NUM_STATES = env.observation_space.shape[0]
    NUM_ACTIONS = env.action_space.n
    print(env_name)
    print(f"Number of States: {NUM_STATES}\nNumber of Actions: {NUM_ACTIONS}")

    NUM_EPISODES = 500
    EPS_START = 1
    EPS_END = 0.01
    EPS_DECAY_RATE = 0.01
    epsilon_values = rl.exp_decay_schedule(EPS_START, EPS_END, NUM_EPISODES, EPS_DECAY_RATE)

    GAMMA = 0.999
    TAU = 0.1
    HIDDEN_DIMS = (1024,)
    MEMORY_SIZE = 100000
    BATCH_SIZE = 128
    NUM_WARMUP_BATCHES = 5
    UPDATE_TARGET_STEPS = 1

    # set learning rates
    lr_rates = [5*(10**-i) for i in range(1, 6)]
    train_stats = {}
    print(f"Learning Rates: {lr_rates}")

    # create model folder if not existed
    MAIN_PATH = "Dueling-DDQN-models"
    if not os.path.exists(MAIN_PATH):
        os.mkdir(MAIN_PATH)
        
    PATH = os.path.join(MAIN_PATH, env_name)
    if not os.path.exists(PATH):
        os.mkdir(PATH)

    for lr in tqdm(lr_rates, desc="Training"):
        replay_buffer = rl.ReplayBuffer(MEMORY_SIZE)
        policy_network = rl.DuelingQNetwork(NUM_STATES, NUM_ACTIONS, HIDDEN_DIMS)
        target_network = copy.deepcopy(policy_network)
        optimizer = optim.RMSprop(policy_network.parameters(), lr=lr)
        criterion = nn.MSELoss()
        episode_rewards = np.zeros(NUM_EPISODES)
        total_steps = np.zeros(NUM_EPISODES)

        for ep in tqdm(range(NUM_EPISODES), leave=False, desc=f"LearningRate={lr}"):
            state = env.reset()
            done = False
            ep_reward = 0
            epsilon = epsilon_values[ep]
            steps = 0

            while not done:
                steps += 1
                action = rl.select_action_from_network(state, epsilon, policy_network)
                next_state, reward, done, info = env.step(action)
                ep_reward += reward
                experience = (state, action, reward, next_state, done) 
                replay_buffer.store(experience)
                state = next_state

                if len(replay_buffer) > (BATCH_SIZE * NUM_WARMUP_BATCHES):
                    samples = replay_buffer.sample(BATCH_SIZE)
                    samples = policy_network.transform_experiences(samples)
                    states, actions, rewards, next_states, dones = samples

                    argmax_q_sp = policy_network(next_states).max(1)[1]
                    q_sp = target_network(next_states).detach()
                    max_q_sp = q_sp[np.arange(BATCH_SIZE), argmax_q_sp].unsqueeze(1)
                    q_target = rewards + (GAMMA * max_q_sp * torch.logical_not(dones))
                    q_sa = policy_network(states).gather(1, actions)

                    loss = criterion(q_sa, q_target)
                    optimizer.zero_grad()
                    loss.backward()
                    # for param in policy_network.parameters():
                        # param.grad.data.clamp_(-1, 1)
                    optimizer.step()

                if steps % UPDATE_TARGET_STEPS == 0:
                    for target, policy in zip(target_network.parameters(), policy_network.parameters()):
                        target_ratio = (1 - TAU) * target
                        policy_ratio = TAU * policy
                        target.data.copy_(target_ratio + policy_ratio)

            episode_rewards[ep] = ep_reward
            total_steps[ep] = steps
            
            # save model 
            if (ep+1) % max(1, int(NUM_EPISODES*0.01)) == 0:
                filename = f"DuelingDDQN_{env_name}_lr{lr}_{HIDDEN_DIMS}.pt"
                save_path = os.path.join(PATH, filename)
                torch.save(policy_network.state_dict(), save_path)
        
        train_stats[lr] = {}
        train_stats[lr]['rewards'] = episode_rewards
        train_stats[lr]['timesteps'] = total_steps
        
        train_df = rl.get_stats_df(train_stats)
        rl.save_df_to_csv(train_df, f"Dueling-DDQN_{env_name}_train_stats.csv", PATH)
        title = f"Dueling-DDQN | {env_name} | Training"
        rl.plot_lr_stats(train_df, title, average=True, epsilons=epsilon_values, save_path=PATH, show=False)

    rl.print_stats_table(train_df)

    # Evaluate models
    EVAL_EPISODES = 100
    eval_stats = {}

    for lr in tqdm(lr_rates, desc="Evaluation"):
        filename = f"DuelingDDQN_{env_name}_lr{lr}_{HIDDEN_DIMS}.pt"
        save_path = os.path.join(PATH, filename)
        model = rl.DuelingQNetwork(NUM_STATES, NUM_ACTIONS, HIDDEN_DIMS)
        model.load_state_dict(torch.load(save_path))
        model.eval()
        
        episode_rewards = np.zeros(EVAL_EPISODES)
        total_steps = np.zeros(EVAL_EPISODES)
        for ep in tqdm(range(EVAL_EPISODES), leave=False, desc=f"LearningRate={lr}"):
            state = env.reset()
            done = False
            ep_reward = 0
            steps = 0
            
            while not done:
                steps += 1
                action = rl.select_action_from_network(state, 0, model)
                next_state, reward, done, info = env.step(action)
                ep_reward += reward
                state = next_state
            
            episode_rewards[ep] = ep_reward
            total_steps[ep] = steps
            
        eval_stats[lr] = {}
        eval_stats[lr]['rewards'] = episode_rewards
        eval_stats[lr]['timesteps'] = total_steps
        
        eval_df = rl.get_stats_df(eval_stats)
        rl.save_df_to_csv(eval_df, f"Dueling-DDQN_{env_name}_eval_stats.csv", PATH)
        title = f"Dueling-DDQN | {env_name} | Evaluation"
        rl.plot_lr_stats(eval_df, title, save_path=PATH, show=False)

    rl.print_stats_table(eval_df)


if __name__ == "__main__":
    main()