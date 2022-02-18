import os 

import gym
import numpy as np 
import matplotlib.pyplot as plt 
from tqdm import tqdm

plt.rcParams['figure.figsize'] = (12,6)
plt.style.use("seaborn-whitegrid")

import rlpy as rl 

def main():

    env_name = "FrozenLake-v1"
    env = gym.make(env_name)
    NUM_ACTIONS = env.action_space.n
    NUM_STATES = env.observation_space.n
    print(f"Number of Actions: {NUM_ACTIONS}\nNumber of States: {NUM_STATES}")

    NUM_EPISODES = 100_000
    eps_values = rl.exp_decay_schedule(1, 0.01, NUM_EPISODES, 0.00005)
    alpha_values = rl.exp_decay_schedule(0.5, 0.01, NUM_EPISODES, 0.00005)
    # plt.plot(eps_values, label='eps')
    # plt.plot(alpha_values, label='alpha')

    GAMMA = 0.99
    episode_rewards = np.zeros(NUM_EPISODES)
    Qtable = np.zeros(shape=(NUM_STATES, NUM_ACTIONS))

    for ep in tqdm(range(NUM_EPISODES), desc='Training'):
        state = env.reset()
        ep_return = 0
        done = False
        eps = eps_values[ep]
        alpha = alpha_values[ep]
        
        while not done:
            # choose action
            if np.random.random() <= eps:
                action = np.random.randint(NUM_ACTIONS)
            else:
                action = np.argmax(Qtable[state])
                
            # interact with environment
            next_state, reward, done, info = env.step(action)
            ep_return += reward
            
            # update Q-table
            target = reward + GAMMA * Qtable[next_state].max() * (not done)
            error = target - Qtable[state, action]
            Qtable[state, action] += alpha * error
            
            state = next_state
            
        episode_rewards[ep] = ep_return
        
        if (ep+1) % int(NUM_EPISODES*0.05) == 0:
            window = max(0, (ep+1)-int(NUM_EPISODES*0.1))
            avg_ret = episode_rewards[window:(ep+1)].mean()
            tqdm.write(f"Episode: {ep+1}/{NUM_EPISODES} | Average Returns:{avg_ret}")

    MAIN_PATH = 'Q-Learning-models'
    if not os.path.exists(MAIN_PATH): os.mkdir(MAIN_PATH)

    PATH = os.path.join(MAIN_PATH, env_name)
    if not os.path.exists(PATH): os.mkdir(PATH)

    avg_rewards = rl.get_average(episode_rewards, int(NUM_EPISODES*0.1))

    fig, ax = plt.subplots()
    line1, = ax.plot(avg_rewards, label='Avg Rewards', color='tab:blue', linewidth=3)
    title = f"QLearning_{env_name}_Training"
    ax.set_title(title.replace('_', ' | '), fontdict={"size":18, "weight":"bold"})
    ax.set_xlabel("Episode", fontsize=15)
    ax.set_ylabel("Rewards", fontsize=15)

    ax2 = ax.twinx()
    line2, = ax2.plot(eps_values, label='Epsilon', linestyle="--", alpha=0.8, color='tab:orange')
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
    for ep in tqdm(range(EVAL_EPISODES), desc='Evaluation'):
        state = env.reset()
        ep_reward = 0
        done = False
        
        while not done:
            # choose action
            action = np.argmax(Qtable[state])
                
            # interact with environment
            next_state, reward, done, info = env.step(action)
            ep_reward += reward
            state = next_state
        
        eval_rewards[ep] = ep_reward
        
    avg_rewards = rl.get_average(eval_rewards, int(EVAL_EPISODES*0.1))
    plt.figure()
    plt.plot(avg_rewards, label='Avg Rewards', color='tab:red', linewidth=3)
    title = f"QLearning_{env_name}_Evaluation"
    plt.title(title.replace('_', ' | '), fontdict={"size":18, "weight":"bold"})
    plt.xlabel("Episodes")
    plt.tight_layout()
    plt.savefig(os.path.join(PATH, title+'.png'), dpi=300)
    plt.cla()
    plt.clf()
    plt.close()
    
    
if __name__ == "__main__":
    main()