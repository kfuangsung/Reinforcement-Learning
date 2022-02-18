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

    # save path
    MAIN_PATH = 'TD3-models'
    if not os.path.exists(MAIN_PATH): os.mkdir(MAIN_PATH)

    PATH = os.path.join(MAIN_PATH, env_name)
    if not os.path.exists(PATH): os.mkdir(PATH)

    IS_RESUME_TRAINING = False

    # set policy network
    POLICY_HIDDEN_DIMS = (256,256)
    POLICY_LEARNING_RATE = 0.0003
    POLICY_MAX_GRAD_NORM = float('inf')
    online_policy_network = rl.DeterministicPolicyNetwork(NUM_STATES, ACTION_BOUNDS, POLICY_HIDDEN_DIMS)

    POLICY_PATH = os.path.join(PATH, f"TD3_policy_{env_name}_lr{POLICY_LEARNING_RATE}_{POLICY_HIDDEN_DIMS}.pt")
    if os.path.exists(POLICY_PATH) and IS_RESUME_TRAINING:
        online_policy_network.load_state_dict(torch.load(POLICY_PATH))
        print("Resume training Policy network.")

    target_policy_network = copy.deepcopy(online_policy_network)
    policy_optimizer = optim.Adam(online_policy_network.parameters(), lr=POLICY_LEARNING_RATE)

    # set value network
    VALUE_HIDDEN_DIMS = (256,256)
    VALUE_LEARNING_RATE = 0.0003
    VALUE_MAX_GRAD_NORM = float('inf')
    online_value_network = rl.TwinQValueNetwork(NUM_STATES, NUM_ACTIONS, VALUE_HIDDEN_DIMS)

    VALUE_PATH = os.path.join(PATH, f"TD3_value_{env_name}_lr{VALUE_LEARNING_RATE}_{VALUE_HIDDEN_DIMS}.pt")
    if os.path.exists(VALUE_PATH) and IS_RESUME_TRAINING:
        online_value_network.load_state_dict(torch.load(VALUE_PATH))
        print("Resume training Value network.")

    target_value_network = copy.deepcopy(online_value_network)
    value_optimizer = optim.Adam(online_value_network.parameters(), lr=VALUE_LEARNING_RATE)

    UPDATE_VALUE_TARGET_EVERY_STEPS = 2
    UPDATE_POLICY_TARGET_EVERY_STEPS = 2
    TRAIN_POLICY_EVERY_STEPS = 2
    POLICY_NOISE_RATIO = 0.1
    POLICY_NOISE_CLIP_RATIO = 0.5
    TAU = 0.005
    GAMMA = 0.99

    # replay buffer
    MEMORY_SIZE = 100000
    BATCH_SIZE = 256
    NUM_WARMUP_BATHCES = 5
    replay_buffer = rl.ReplayBuffer(MEMORY_SIZE)

    NUM_EPISODES = 500
    episode_rewards = np.zeros(NUM_EPISODES)
    episode_steps = np.zeros(NUM_EPISODES)
    
    # traning_strategy
    INIT_NOISE = 1
    MIN_NOISE = 0.1 
    DECAY_STEPS = int(NUM_EPISODES * env._max_episode_steps * 0.4)
    traning_strategy = rl.NormalNoiseDecayStrategy(ACTION_BOUNDS, INIT_NOISE, MIN_NOISE, DECAY_STEPS)

    for ep in tqdm(range(NUM_EPISODES), desc='Training'):
        state = env.reset()
        done = False
        ep_reward = 0
        ep_step = 0
        
        while not done:
            min_samples = BATCH_SIZE * NUM_WARMUP_BATHCES
            is_max_explore = len(replay_buffer) < min_samples
            action = traning_strategy.select_action(online_policy_network, state, is_max_explore)
            next_state, reward, done, _ = env.step(action)
            experience = (state, action, reward, next_state, done)
            replay_buffer.store(experience)
            ep_reward += reward
            ep_step += 1
            state = next_state
            
            if len(replay_buffer) > min_samples:
                samples = replay_buffer.sample(BATCH_SIZE)
                samples = online_value_network.format_experiences(samples)
                states, actions, rewards, next_states, dones = samples
                
                # get optimize value target
                with torch.no_grad():
                    a_range = target_policy_network.action_max - target_policy_network.action_min
                    a_noise = torch.rand_like(actions) * POLICY_NOISE_RATIO * a_range
                    noise_min = target_policy_network.action_min * POLICY_NOISE_CLIP_RATIO
                    noise_max = target_policy_network.action_max * POLICY_NOISE_CLIP_RATIO
                    a_noise = torch.max(torch.min(a_noise, noise_max), noise_min)

                    argmax_a_q_sp = target_policy_network(next_states)
                    noisy_argmax_a_q_sp = argmax_a_q_sp + a_noise
                    noisy_argmax_a_q_sp = torch.max(torch.min(noisy_argmax_a_q_sp, 
                                                            target_policy_network.action_max), 
                                                    target_policy_network.action_min)
                    max_a_q_sp_a, max_a_q_sp_b = target_value_network(next_states, noisy_argmax_a_q_sp)
                    max_a_q_sp = torch.min(max_a_q_sp_a, max_a_q_sp_b)
                    target_q_sa = rewards + GAMMA * max_a_q_sp * torch.logical_not(dones)
                
                # optimize value network
                q_sa_a, q_sa_b = online_value_network(states, actions)
                td_error_a = q_sa_a - target_q_sa
                td_error_b = q_sa_b - target_q_sa
                value_loss = td_error_a.pow(2).mul(0.5).mean() + td_error_b.pow(2).mul(0.5).mean()
                value_optimizer.zero_grad()
                value_loss.backward()
                torch.nn.utils.clip_grad_norm_(online_value_network.parameters(), VALUE_MAX_GRAD_NORM)
                value_optimizer.step()
                
                # optimize policy network
                if ep_step % TRAIN_POLICY_EVERY_STEPS == 0:
                    argmax_a_q_s = online_policy_network(states)
                    max_a_q_s = online_value_network.Qa(states, argmax_a_q_s)
                    policy_loss = -max_a_q_s.mean()
                    policy_optimizer.zero_grad()
                    policy_loss.backward()
                    torch.nn.utils.clip_grad_norm_(online_policy_network.parameters(), POLICY_MAX_GRAD_NORM)
                    policy_optimizer.step()
                    
            # update target value network
            if ep_step % UPDATE_VALUE_TARGET_EVERY_STEPS == 0:
                for target, online in zip(target_value_network.parameters(), online_value_network.parameters()):
                    target_ratio = (1 - TAU) * target.data
                    online_ratio = TAU * online.data
                    mixed_weights = target_ratio + online_ratio
                    target.data.copy_(mixed_weights)
                    
            # update target policy network
            if ep_step % UPDATE_POLICY_TARGET_EVERY_STEPS == 0:
                for target, online in zip(target_policy_network.parameters(), online_policy_network.parameters()):
                    target_ratio = (1 - TAU) * target.data
                    online_ratio = TAU * online.data
                    mixed_weights = target_ratio + online_ratio
                    target.data.copy_(mixed_weights)
                
        episode_rewards[ep] = ep_reward
        episode_steps[ep] = ep_step
        
        # save models
        torch.save(online_policy_network.state_dict(), POLICY_PATH)
        torch.save(online_value_network.state_dict(), VALUE_PATH)
        
        
    # plot training results
    plt.figure()
    avg_rewards = rl.get_average(episode_rewards, int(0.1*NUM_EPISODES))
    plt.subplot(2, 1, 1)
    plt.plot(avg_rewards, color='tab:blue', linewidth=3)
    plt.ylabel("Rewards", fontsize=14)
    title = f"TD3_{env_name}_Training"
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
    model =  rl.DeterministicPolicyNetwork(NUM_STATES, ACTION_BOUNDS, POLICY_HIDDEN_DIMS)
    model.load_state_dict(torch.load(POLICY_PATH))
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
    title = f"TD3_{env_name}_Evaluation"
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