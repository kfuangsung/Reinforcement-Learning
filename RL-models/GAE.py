import os 

import gym 
import numpy as np 
import torch
import torch.multiprocessing as mp
from tqdm import tqdm

import rlpy as rl 

def main():
    #%% define gym environment

    env_name = 'MountainCar-v0'
    env = gym.make(env_name)
    NUM_ACTIONS = env.action_space.n
    NUM_STATES = env.observation_space.shape[0]
    print(env_name)
    print(f"Number of Actions: {NUM_ACTIONS}\nNumber of States: {NUM_STATES}")

    NUM_EPISODES = 1000
    GAMMA = 0.999
    TAU = 0.95
    ENTROPY_LOSS_WEIGHT = 0.001
    MAX_STEPS = 50
    NUM_WORKERS = 8

    POLICY_HIDDEN_DIMS = (128, 64)
    POLICY_LEARNING_RATE = 0.0005
    POLICY_MAX_GRAD_NORM = 1
    shared_policy_network = rl.DiscreteActionPolicyNetwork(NUM_STATES, NUM_ACTIONS, POLICY_HIDDEN_DIMS)
    shared_policy_optimizer = rl.SharedAdam(shared_policy_network.parameters(), lr=POLICY_LEARNING_RATE)

    VALUE_HIDDEN_DIMS = (256, 128)
    VALUE_LEARNING_RATE = 0.0007
    VALUE_MAX_GRAD_NORM = float('inf')
    shared_value_network = rl.StateValueNetwork(NUM_STATES, VALUE_HIDDEN_DIMS)
    shared_value_optimizer = rl.SharedRMSprop(shared_value_network.parameters(), lr=VALUE_LEARNING_RATE)

    MAIN_PATH = 'GAE-models'
    if not os.path.exists(MAIN_PATH):
        os.mkdir(MAIN_PATH)

    PATH = os.path.join(MAIN_PATH, env_name)
    if not os.path.exists(PATH):
        os.mkdir(PATH)

    POLICY_PATH = os.path.join(PATH, f"GAE_polciy_{env_name}_lr{POLICY_LEARNING_RATE}_{POLICY_HIDDEN_DIMS}.pt")
    VALUE_PATH = os.path.join(PATH, f"GAE_value_{env_name}_lr{VALUE_LEARNING_RATE}_{VALUE_HIDDEN_DIMS}.pt")
    
    if os.path.exists(POLICY_PATH):
        shared_policy_network.load_state_dict(torch.load(POLICY_PATH))
        print("Resume training policy network")
        
    if os.path.exists(VALUE_PATH):
        shared_value_network.load_state_dict(torch.load(VALUE_PATH))
        print("Resume training value network")

    def worker(rank):
        local_env = gym.make(env_name)
        
        local_policy_network = rl.DiscreteActionPolicyNetwork(NUM_STATES, NUM_ACTIONS, POLICY_HIDDEN_DIMS)
        local_policy_network.load_state_dict(shared_policy_network.state_dict())
        
        local_value_network = rl.StateValueNetwork(NUM_STATES, VALUE_HIDDEN_DIMS)
        local_value_network.load_state_dict(local_value_network.state_dict())
        
        for ep in tqdm(range(NUM_EPISODES), leave=True, desc=f"worker{rank}"):
            state = local_env.reset()
            done = False
            rewards = []
            logpas = []
            entropies = []
            values = []
            step_start = 0
            steps = 1
            
            while not done:
                action, logpa, entropy = local_policy_network.full_pass(state)
                new_state, reward, done, _ = local_env.step(action)
                steps += 1
                value = local_value_network(state)
                
                rewards.append(reward)
                logpas.append(logpa)
                entropies.append(entropy)
                values.append(value)
                
                state = new_state
                
                if done or ((steps - step_start) == MAX_STEPS):
                    
                    if done: 
                        next_value = 0
                    else:
                        next_value = local_value_network(state).detach().item()
                    rewards.append(next_value)
                    values.append(torch.FloatTensor([[next_value,],]))
                    
                    # optimize 
                    # GAE
                    T = len(rewards)
                    discounts = np.logspace(0, T, num=T, endpoint=False, base=GAMMA)
                    returns = np.array([np.sum(discounts[:T-t] * rewards[t:]) for t in range(T)])
                    
                    logpas = torch.cat(logpas)
                    entropies = torch.cat(entropies)
                    values = torch.cat(values)
                    
                    np_values = values.view(-1).data.numpy()
                    tau_discounts = np.logspace(0, T-1, num=T-1, endpoint=False, base=GAMMA*TAU)
                    advs = rewards[:-1] + GAMMA * np_values[1:] - np_values[:-1]
                    gaes = np.array([np.sum(tau_discounts[:T-1-t] * advs[t:]) for t in range(T-1)])
                    
                    values = values[:-1, ...] # select all except last row
                    discounts = torch.tensor(discounts[:-1], dtype=torch.float32).unsqueeze(1)
                    returns = torch.tensor(returns[:-1], dtype=torch.float32).unsqueeze(1)
                    gaes = torch.tensor(gaes, dtype=torch.float32).unsqueeze(1)
                    
                    policy_loss = -(discounts * gaes.detach() * logpas).mean()
                    entropy_loss = -entropies.mean()
                    loss = policy_loss + ENTROPY_LOSS_WEIGHT*entropy_loss
                    
                    shared_policy_optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(local_policy_network.parameters(), 
                                                    POLICY_MAX_GRAD_NORM)
                    for param, shared_param in zip(local_policy_network.parameters(), 
                                                shared_policy_network.parameters()):
                        if shared_param.grad is None: 
                            shared_param._grad = param.grad
                    shared_policy_optimizer.step()
                    local_policy_network.load_state_dict(shared_policy_network.state_dict())
                    
                    value_error = returns - values
                    value_loss = value_error.pow(2).mul(0.5).mean()
                    shared_value_optimizer.zero_grad()
                    value_loss.backward()
                    torch.nn.utils.clip_grad_norm_(local_value_network.parameters(),
                                                    VALUE_MAX_GRAD_NORM)
                    for param, shared_param in zip(local_value_network.parameters(),
                                                shared_value_network.parameters()):
                        if shared_param.grad is None:
                            shared_param._grad = param.grad
                    shared_value_optimizer.step()
                    local_value_network.load_state_dict(shared_value_network.state_dict())
                    
                    rewards = []
                    logpas = []
                    entropies = []
                    values = []
                    step_start = steps
                    
            torch.save(shared_policy_network.state_dict(), POLICY_PATH)
            torch.save(shared_value_network.state_dict(), VALUE_PATH)

    #%% Training

    workers = [mp.Process(target=worker, args=(rank,)) for rank in range(NUM_WORKERS)]
    [w.start() for w in workers]; [w.join for w in workers]
            
            
if __name__ == '__main__':
    main()