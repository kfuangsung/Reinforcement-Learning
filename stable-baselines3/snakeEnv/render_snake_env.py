import time
from stable_baselines3 import PPO
from snake_env import *

env = SnakeGameEnv()
model = PPO.load("snake_game/ppo_snake_game_1000000x10")

num_episode = 5
rewards = []
infos = []
steps = []

for i in range(num_episode):
    print(f"Episode: {i+1}")
    done = False
    state = env.reset()
    env.render()
    ep_step = 0
    ep_reward = 0
    while not done:
        action, _ = model.predict(state, deterministic=True)
        # action = np.random.randint(4)
        next_state, reward, done, info = env.step(action)
        ep_reward += reward
        ep_step += 1
        # print(f'step: {ep_step} | action: {action} |reward: {reward} | done: {done}')
        env.render()
        state = next_state
    
    rewards.append(ep_reward)
    infos.append(info['GameOver'])
    steps.append(ep_step)
    
for i , (reward, step, info) in enumerate(zip(rewards, steps, infos)):
    print(f"{i+1} | reward: {reward} | steps:{step} | {info}")