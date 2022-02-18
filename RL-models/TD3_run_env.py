import gym 
import torch

import rlpy as rl 

env_name = 'MountainCarContinuous-v0'
env = gym.make(env_name)
NUM_STATES = env.observation_space.shape[0]
NUM_ACTIONS = env.action_space.shape[0]
print(env_name)
print(f"Number of States: {NUM_STATES}\nNumber of Actions: {NUM_ACTIONS}")
print("Max steps: ", env._max_episode_steps)
ACTION_BOUNDS = (env.action_space.low, env.action_space.high)
POLICY_PATH = f"TD3-models/{env_name}/TD3_policy_MountainCarContinuous-v0_lr0.0003_(256, 256).pt"
POLICY_HIDDEN_DIMS = (256,256)
model =  rl.DeterministicPolicyNetwork(NUM_STATES, ACTION_BOUNDS, POLICY_HIDDEN_DIMS)
model.load_state_dict(torch.load(POLICY_PATH))
strategy = rl.GreedyStrategy(ACTION_BOUNDS)

for ep in range(3):
    state = env.reset()
    done = False
    rewards = 0
    steps = 0
    env.render()
    
    while not done:
        action = strategy.select_action(model, state)
        next_state, reward, done, _ = env.step(action)
        env.render()
        rewards += reward
        steps += 1
        state = next_state
        
    print(f"{ep+1} | Rewards: {rewards} | Steps: {steps}")