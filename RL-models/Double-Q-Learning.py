import os

import gym 
import numpy as np 
import matplotlib.pyplot as plt
from tqdm import tqdm

plt.rcParams['figure.figsize'] = (14,6)
plt.style.use('seaborn-whitegrid')

import rlpy as rl 

def main():

    env_name = "FrozenLake-v1"
    env = gym.make(env_name)
    NUM_ACTIONS = env.action_space.n
    NUM_STATES = env.observation_space.n
    print(f"Number of Actions: {NUM_ACTIONS}\nNumber of States: {NUM_STATES}")

    NUM_EPISODES = 100_000
    epsilon_values = rl.exp_decay_schedule(1, 0.01, NUM_EPISODES, 0.00005)
    alpha_values = rl.exp_decay_schedule(0.5, 0.01, NUM_EPISODES, 0.00005)
    GAMMA = 0.99
    Q1_table = np.zeros(shape=(NUM_STATES, NUM_ACTIONS))
    Q2_table = np.zeros(shape=(NUM_STATES, NUM_ACTIONS))
    episode_rewards = np.zeros(NUM_EPISODES)

    for ep in tqdm(range(NUM_EPISODES)):
        state = env.reset()
        ep_reward = 0
        done = False
        epsilon = epsilon_values[ep]
        alpha = alpha_values[ep]
        
        while not done:
            # choose action
            if np.random.random() <= epsilon:
                action = np.random.randint(NUM_ACTIONS)
            else:
                q = (Q1_table + Q2_table) / 2
                action = np.argmax(q[state])
                
            # interact with environment
            next_state, reward, done, info = env.step(action)
            ep_reward += reward
            
            # update Q-table
            # 50% of the time update Q1 else Q2
            if np.random.random() <= 0.5:
                # update Q1-table
                Q1_action = np.argmax(Q1_table[next_state])
                # use value from Q2 to update Q1
                target = reward + GAMMA * Q2_table[next_state, Q1_action] * (not done)
                error = target - Q1_table[state, action]
                Q1_table[state, action] += alpha * error
            else:
                # update Q2-table
                Q2_action = np.argmax(Q2_table[next_state])
                # use value from Q1 to update Q2
                target = reward + GAMMA * Q1_table[next_state, Q2_action] * (not done)
                error = target - Q2_table[state, action]
                Q2_table[state, action] += alpha * error
                
            state = next_state
            
        episode_rewards[ep] = ep_reward
        
        if (ep+1) % int(NUM_EPISODES*0.05) == 0:
            window = max(0, (ep+1)-(NUM_EPISODES*0.1))
            avg_ret = episode_rewards[int(window):ep+1].mean()
            tqdm.write(f"Episode: {ep+1}/{NUM_EPISODES} | Average Return: {avg_ret:.4f}")


    MAIN_PATH = 'Double-Q-Learning-models'
    if not os.path.exists(MAIN_PATH): os.mkdir(MAIN_PATH)

    PATH = os.path.join(MAIN_PATH, env_name)
    if not os.path.exists(PATH): os.mkdir(PATH)

    avg_rewards = rl.get_average(episode_rewards, int(NUM_EPISODES*0.1))

    fig, ax = plt.subplots()
    line1, = ax.plot(avg_rewards, label='Avg Rewards', color='tab:blue', linewidth=3)
    title = f"Double-Q-Learning_{env_name}_Training"
    ax.set_title(title.replace('_', ' | '), fontdict={"size":18, "weight":"bold"})
    ax.set_xlabel("Episode", fontsize=15)
    ax.set_ylabel("Rewards", fontsize=15)

    ax2 = ax.twinx()
    line2, = ax2.plot(epsilon_values, label='Epsilon', linestyle="--", alpha=0.8, color='tab:orange')
    line3, = ax2.plot(alpha_values, label='Alpha', linestyle="-.", alpha=0.8, color='tab:green')
    ax2.set_ylabel("Epsilon / Alpha", fontsize=15)

    ax.legend(handles=[line1, line2, line3], loc='lower left', fontsize=12, frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(PATH, title+'.png'), dpi=300)
    plt.cla()
    plt.clf()
    plt.close()


    # Evaluation 
    EVAL_EPISODES = 1000
    eval_rewards = np.zeros(EVAL_EPISODES)
    for ep in tqdm(range(EVAL_EPISODES)):
        state = env.reset()
        ep_reward = 0
        done = False
        
        while not done:
            # choose action
            q = (Q1_table + Q2_table) / 2
            action = np.argmax(q[state])
                
            # interact with environment
            next_state, reward, done, info = env.step(action)
            ep_reward += reward
            state = next_state
        
        eval_rewards[ep] = ep_reward
        
    avg_rewards = rl.get_average(eval_rewards, int(EVAL_EPISODES*0.1))
    plt.figure()
    plt.plot(avg_rewards, label='Avg Rewards', color='tab:red', linewidth=3)
    title = f"Double-Q-Learning_{env_name}_Evaluation"
    plt.title(title.replace('_', ' | '), fontdict={"size":18, "weight":"bold"})
    plt.xlabel("Episodes")
    plt.tight_layout()
    plt.savefig(os.path.join(PATH, title+'.png'), dpi=300)
    plt.cla()
    plt.clf()
    plt.close()
    
    
if __name__ == "__main__":
    main()