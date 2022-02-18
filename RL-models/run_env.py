import torch
import rlpy as rl 

def main():
    env_name = "MountainCar-v0"
    PATH = f'GAE-models/{env_name}/GAE_polciy_{env_name}_lr0.0005_(128, 64).pt'
    HIDDEN_DIMS = (128, 64)
    print(env_name)
    print(PATH)
    
    env, model = rl.load_model(env_name, rl.DiscreteActionPolicyNetwork, PATH, HIDDEN_DIMS)
    for i in range(3):
        state = env.reset()
        done = False
        rewards = 0
        env.render()
        
        while not done:
            with torch.no_grad():
                action = model.select_greedy_action(state)
            next_state, reward, done, _ = env.step(action)
            env.render()
            rewards += reward
            state = next_state
        
        print(f"{i} | Rewards: {rewards}")
    
    env.close()
        

if __name__ == "__main__":
    main()    