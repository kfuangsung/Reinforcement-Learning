import numpy as np 
import gym
from gym import spaces

class GoLeftEnv(gym.Env):
    """
    1-D grid world where Agent must learn to go left.
    """
    
    def __init__(self, grid_size=10):
        super().__init__()
        
        # actions -->  either go left or right.
        self.left = 0
        self.right = 1
        self.grid_size = grid_size
        # start at the right
        self.agent_pos = self.grid_size - 1
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(low=0, high=self.grid_size, 
                                            shape=(1,), dtype=np.float32)
        
    def reset(self):
        # reset to the right of the grid
        self.agent_pos = self.grid_size - 1
        
        return np.array([self.agent_pos], dtype=np.float32)
    
    def step(self, action):
        if action == self.left:
            self.agent_pos -= 1
        elif action == self.right:
            self.agent_pos += 1
        else:
            raise ValueError(f"Invalid action, action={action} is not part of action space.")
        
        self.agent_pos = np.clip(self.agent_pos, 0, self.grid_size)
        done = bool(self.agent_pos == 0)
        reward = 1 if self.agent_pos == 0 else 0
        info = {}
        
        return np.array([self.agent_pos], dtype=np.float32), reward, done, info
    
    def render(self, mode='console'):
        if mode != 'console':
            raise NotImplementedError("Please choose mode='console'")
        
        print("| "*self.agent_pos, end="")
        print("|X|", end="")
        print(" |"*(self.grid_size - self.agent_pos - 1))
        
    def close(self):
        pass