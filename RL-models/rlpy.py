
import os 

import gym
import numpy as np
import pandas as pd 
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from tabulate import tabulate

plt.rcParams['figure.figsize'] = (14,6)
plt.style.use('seaborn-whitegrid')


def linear_decay_schedule(init_value, end_value, num_episodes, decay_portion):
    values = np.zeros(num_episodes)
    for step in range(num_episodes):
        N = num_episodes * decay_portion
        r = max((N - step) / N, 0)
        val = (init_value - end_value) * r + end_value
        values[step] = val

    return values

def exp_decay_schedule(init_value, end_value, num_episodes, decay_rate):
    values = np.zeros(num_episodes)
    for step in range(num_episodes):
        eps = end_value + (init_value - end_value) * np.exp(-1 * step / (1/decay_rate))
        values[step] = eps

    return values


def get_average(values, window):
    return pd.DataFrame(values).rolling(window).mean().values

    
def select_action_from_network(state, epsilon, network):
    with torch.no_grad():
        q_values = network(state).cpu().detach().data.numpy().squeeze()
    
    if np.random.random() <= epsilon:
        action = np.random.randint(len(q_values))
    else:
        action = np.argmax(q_values)
        
    return action


def get_stats_df(stats_dict):
    df = pd.DataFrame(stats_dict, index=None).T
    df = df.reset_index().rename(columns={'index':'learningRate'})
    df['meanRewards'] = df.rewards.apply(lambda x: np.mean(x))
    df['stdRewards'] = df.rewards.apply(lambda x: np.std(x))
    df['meanTimesteps'] = df.timesteps.apply(lambda x: np.mean(x))
    df['stdTimesteps'] = df.timesteps.apply(lambda x: np.std(x))
    
    return df 


def save_df_to_csv(df, filename, save_path='.'):
    df.to_csv(os.path.join(save_path, filename), index=False)
    
    
def plot_lr_stats(stats_df, title, average=False, epsilons=None, save_path='.', show=True):
    colors_list = list(mcolors.TABLEAU_COLORS.keys())
    fig, (ax1, ax2) = plt.subplots(2, figsize=(12,8))
    for i in range(len(stats_df)):
        rewards = stats_df.rewards.iloc[i]
        timesteps = stats_df.timesteps.iloc[i]
        if average: 
            rewards = get_average(rewards, int(len(rewards)*0.1))
            timesteps = get_average(timesteps, int(len(timesteps)*0.1))
        
        color = colors_list[i]
        
        ax1.plot(rewards, color=color)
        ax1.text(len(rewards), rewards[-1], 
                 f"lr={stats_df.learningRate[i]}", 
                 fontsize=16, fontweight="bold", color=color)
        
        ax2.plot(timesteps, color=color)
        ax2.text(len(timesteps), timesteps[-1], 
                 f"lr={stats_df.learningRate[i]}", 
                 fontsize=16, fontweight="bold", color=color)
    
    if not epsilons is None:
        ax1_ = ax1.twinx()
        ax1_.plot(epsilons, color='k', alpha=0.7, linestyle='--', label='Epsilon')
        ax1_.axis(False)
        ax1_.legend(loc='lower left', frameon=True)
        
        ax2_ = ax2.twinx()
        ax2_.plot(epsilons, color='k', alpha=0.7, linestyle='--', label='Epsilon')
        ax2_.axis(False)
        ax2_.legend(loc='lower left', frameon=True)
        
    ax1.set_ylabel("Rewards", fontsize=15)
    ax2.set_ylabel("Timesteps", fontsize=15)
    ax2.set_xlabel("Episodes", fontsize=15)
    fig.suptitle(title, fontsize=18, fontweight='bold')
    plt.tight_layout()
    plt.savefig(fname=os.path.join(save_path, f"{title}.png"), dpi=300)
    plt.cla()
    plt.clf()
    plt.close()
    


def print_stats_table(stats_df):
    df_summary = stats_df.loc[:, ['learningRate','meanRewards','stdRewards','meanTimesteps','stdTimesteps']]
    print(tabulate(df_summary, headers='keys', showindex=False, tablefmt='psql'))
    

def load_model(env_name, network, model_path, hidden_dims):
    env = gym.make(env_name)
    NUM_STATES = env.observation_space.shape[0]
    NUM_ACTIONS = env.action_space.n
    model = network(NUM_STATES, NUM_ACTIONS, hidden_dims)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    return env, model
    
    
def run_env(env, model):
    state = env.reset()
    done = False
    env.render()
    ret = 0
    steps = 0
    
    while not done:
        action = select_action_from_network(state, 0, model)
        next_state, reward, done, info = env.step(action)
        ret += reward
        steps += 1
        env.render()
        state = next_state
        
    print(f"Return: {ret} | Timesteps: {steps}")
    
# ----------------------------------------------------------------------------------------

class ReplayBuffer():
    def __init__(self, 
                 memory_size=10000):
        self.memory_size = memory_size
        self.state_mem = np.empty(self.memory_size, dtype=np.ndarray)
        self.action_mem = np.empty(self.memory_size, dtype=np.ndarray)
        self.reward_mem = np.empty(self.memory_size, dtype=np.ndarray)
        self.next_state_mem = np.empty(self.memory_size, dtype=np.ndarray)
        self.done_mem = np.empty(self.memory_size, dtype=np.ndarray)
        self.index = 0
        self.size = 0
        
    def store(self, experience):
        state, action, reward, next_state, done = experience
        self.state_mem[self.index] = state
        self.action_mem[self.index] = action
        self.reward_mem[self.index] = reward
        self.next_state_mem[self.index] = next_state
        self.done_mem[self.index] = done
        
        self.index += 1
        self.index = self.index % self.memory_size
        self.size += 1
        self.size = min(self.size, self.memory_size)
        
    def sample(self, batch_size):
        indexes = np.random.choice(self.size, size=batch_size, replace=False)
        state_sample = np.vstack(self.state_mem[indexes])
        action_sample = np.vstack(self.action_mem[indexes])
        reward_sample = np.vstack(self.reward_mem[indexes])
        next_state_sample = np.vstack(self.next_state_mem[indexes])
        done_sample = np.vstack(self.done_mem[indexes])
        
        return (state_sample, action_sample, reward_sample, next_state_sample, done_sample)
    
    def __len__(self):
        
        return self.size

# -----------------------------------------------------------------------------------

class QNetwork(nn.Module):
    def __init__(self,
                 input_dim,
                 output_dim,
                 hidden_dims=(32,32),
                 activation_fn=F.relu):
        super().__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dims[0])
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_dims)-1):
            hidden_layer = nn.Linear(hidden_dims[i], hidden_dims[i+1])
            self.hidden_layers.append(hidden_layer)
        self.output_layer = nn.Linear(hidden_dims[-1], output_dim)
        self.activation_fn = activation_fn
        
        if torch.cuda.is_available():
            device = "cuda:0"
        else:
            device = "cpu"
        
        self.device = torch.device(device)
        self.to(self.device)
        
    def transform_state(self, state):
        x = state
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(state, dtype=torch.float32).to(self.device)
            x = x.unsqueeze(0)
        
        return x    
        
    def forward(self, x):
        x = self.transform_state(x)
        x = self.activation_fn(self.input_layer(x))
        for hidden_layer in self.hidden_layers:
            x = self.activation_fn(hidden_layer(x))
        x = self.output_layer(x)
        
        return x
    
    def transform_experiences(self, experience):
        states, actions, rewards, next_states, dones = experience
        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.int64).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.bool).to(self.device)
        
        return (states, actions, rewards, next_states, dones)
    
# ------------------------------------------------------------------------------------

class DuelingQNetwork(nn.Module):
    def __init__(self,
                 input_dim,
                 output_dim,
                 hidden_dims=(32,32),
                 activation_fn=F.relu):
        super().__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dims[0])
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_dims)-1):
            hidden_layer = nn.Linear(hidden_dims[i], hidden_dims[i+1])
            self.hidden_layers.append(hidden_layer)
        self.output_layer = nn.Linear(hidden_dims[-1], output_dim)
        self.output_value = nn.Linear(hidden_dims[-1], 1)
        self.activation_fn = activation_fn
        
        if torch.cuda.is_available():
            device = "cuda:0"
        else:
            device = "cpu"
        
        self.device = torch.device(device)
        self.to(self.device)
        
    def transform_state(self, state):
        x = state
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(state, dtype=torch.float32).to(self.device)
            x = x.unsqueeze(0)
        
        return x    
        
    def forward(self, x):
        x = self.transform_state(x)
        x = self.activation_fn(self.input_layer(x))
        for hidden_layer in self.hidden_layers:
            x = self.activation_fn(hidden_layer(x))
        a = self.output_layer(x)
        v = self.output_value(x).expand_as(a)
        q = v + a - a.mean(1, keepdim=True).expand_as(a)
        
        return q
    
    def transform_experiences(self, experience):
        states, actions, rewards, next_states, dones = experience
        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.int64).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.bool).to(self.device)
        
        return (states, actions, rewards, next_states, dones)
    
# ----------------------------------------------------------------------------------

class DiscreteActionPolicyNetwork(nn.Module):
    def __init__(self,
                 input_dim,
                 output_dim,
                 hidden_dims=(32,32),
                 activation_fn=F.relu):
        super().__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dims[0])
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_dims)-1):
            hidden_layer = nn.Linear(hidden_dims[i], hidden_dims[i+1])
            self.hidden_layers.append(hidden_layer)
        self.output_layer = nn.Linear(hidden_dims[-1], output_dim)
        self.activation_fn = activation_fn
        
    def format_state(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
            x = x.unsqueeze(0)
        
        return x
        
    def forward(self, state):
        x = self.format_state(state)
        x = self.activation_fn(self.input_layer(x))
        for hidden_layer in self.hidden_layers:
            x = self.activation_fn(hidden_layer(x))
        x = self.output_layer(x)
        
        return x
    
    def full_pass(self, state):
        logits = self.forward(state)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        logpa = dist.log_prob(action).unsqueeze(-1)
        entropy = dist.entropy().unsqueeze(-1)
        
        return action.item(), logpa, entropy
    
    def select_action(self, state):
        logits = self.forward(state)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        
        return action.item()
    
    def select_greedy_action(self, state):
        logits = self.forward(state)
        action = np.argmax(logits.detach().numpy())
        
        return action
    
# ----------------------------------------------------------------------------------------

class StateValueNetwork(nn.Module):
    def __init__(self, 
                 input_dim,
                 hidden_dims=(32,32),
                 activation_fn=F.relu):
        super().__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dims[0])
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_dims)-1):
            hidden_layer = nn.Linear(hidden_dims[i], hidden_dims[i+1])
            self.hidden_layers.append(hidden_layer)
        self.output_layer = nn.Linear(hidden_dims[-1], 1)
        self.activation_fn = activation_fn
        
    def format_state(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
            x = x.unsqueeze(0)
        
        return x
    
    def forward(self, state):
        x = self.format_state(state)
        x = self.activation_fn(self.input_layer(x))
        for hidden_layer in self.hidden_layers:
            x = self.activation_fn(hidden_layer(x))
        x = self.output_layer(x)
        
        return x
    
# --------------------------------------------------------------------------------

class SharedAdam(torch.optim.Adam):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), 
                 eps=1e-8, weight_decay=0, amsgrad=False):
        super().__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad)
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['shared_step'] = torch.zeros(1).share_memory_()
                state['exp_avg'] = torch.zeros_like(p.data).share_memory_()
                state['exp_avg_sq'] = torch.zeros_like(p.data).share_memory_()
                if weight_decay:
                    state['weight_decay'] = torch.zeros_like(p.data).share_memory_()
                if amsgrad:
                    state['max_exp_avg_sq'] = torch.zeros_like(p.data).share_memory_()
                    
    def step(self, closure=None):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None: continue
                self.state[p]['steps'] = self.state[p]['shared_step'].item()
                self.state[p]['shared_step'] += 1
        super().step(closure)

# --------------------------------------------------------------------------------------

class SharedRMSprop(torch.optim.RMSprop):
    def __init__(self, params, lr=1e-2, alpha=0.99, eps=1e-8, 
                 weight_decay=0, momentum=0, centered=False):
        super().__init__(params, lr=lr, alpha=alpha, eps=eps, weight_decay=weight_decay, 
                         momentum=momentum, centered=centered)
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['shared_step'] = torch.zeros(1).share_memory_()
                state['square_avg'] = torch.zeros_like(p.data).share_memory_()
                if weight_decay:
                    state['weight_decay'] = torch.zeros_like(p.data).share_memory_()
                if momentum > 0:
                    state['momentum_buffer'] = torch.zeros_like(p.data).share_memory_()
                if centered:
                    state['grad_avg'] = torch.zeros_like(p.data).share_memory_()
                    
    def step(self, closure=None):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None: continue
                self.state[p]['steps'] = self.state[p]['shared_step'].item()
                self.state[p]['shared_step'] += 1
        super().step(closure)
        
# ----------------------------------------------------------------------------

class QValueNetwork(nn.Module):
    def __init__(self,
                 input_dim, 
                 output_dim,
                 hidden_dims=(32,32),
                 activation_fn=F.relu):
        super().__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dims[0])
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_dims)-1):
            in_dim = hidden_dims[i]
            if i == 0: 
                in_dim += output_dim
            hidden_layer = nn.Linear(in_dim, hidden_dims[i+1])
            self.hidden_layers.append(hidden_layer)
        self.output_layer = nn.Linear(hidden_dims[-1], 1)
        self.activation_fn = activation_fn
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        self.to(self.device)
        
    def format_input(self, state, action):
        x, u = state, action
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32, device=self.device)
            x = x.unsqueeze(0)
        if not isinstance(u, torch.Tensor):
            u = torch.tensor(u, dtype=torch.float32, device=self.device)
            u = u.unsqueeze(0)
            
        return x, u
        
    def forward(self, state, action):
        x, u = self.format_input(state, action) 
        x = self.activation_fn(self.input_layer(x))
        for i, hidden_layer in enumerate(self.hidden_layers):
            if i == 0:
                x = torch.cat((x, u), dim=1)
            x = self.activation_fn(hidden_layer(x))
        x = self.output_layer(x)
        
        return x 
    
    def format_experiences(self, experiences):
        states, actions, rewards, next_states, dones = experiences
        states = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions = torch.tensor(actions, dtype=torch.float32, device=self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32, device=self.device) 
        dones = torch.tensor(dones, dtype=torch.bool, device=self.device)
        
        return states, actions, rewards, next_states, dones
    
# -------------------------------------------------------------------------------
    
class DeterministicPolicyNetwork(nn.Module):
    def __init__(self,
                 input_dim,
                 action_bounds,
                 hidden_dims=(32,32),
                 activation_fn=F.relu,
                 out_activation_fn=torch.tanh):
        super().__init__()
        self.action_min, self.action_max = action_bounds
        self.input_layer = nn.Linear(input_dim, hidden_dims[0])
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_dims)-1):
            hidden_layer = nn.Linear(hidden_dims[i], hidden_dims[i+1])
            self.hidden_layers.append(hidden_layer)
        self.output_layer = nn.Linear(hidden_dims[-1], len(self.action_max))
        self.activation_fn = activation_fn
        self.out_activation_fn = out_activation_fn
        
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        self.to(self.device)
        
        self.action_min = torch.tensor(self.action_min, dtype=torch.float32, device=self.device)
        self.action_max = torch.tensor(self.action_max, dtype=torch.float32, device=self.device)
        self.nn_min = self.out_activation_fn(torch.tensor([float('-inf')], dtype=torch.float32, device=self.device))
        self.nn_max = self.out_activation_fn(torch.tensor([float('inf')], dtype=torch.float32, device=self.device))
        self.rescale_fn = lambda x: (x-self.nn_min) * (self.action_max-self.action_min) / (self.nn_max-self.nn_min) + self.action_min 
        
    def format_input(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32, device=self.device)
            x = x.unsqueeze(0)
        
        return x   
            
    def forward(self, state):
        x = self.format_input(state)
        x = self.activation_fn(self.input_layer(x))
        for hidden_layer in self.hidden_layers:
            x = self.activation_fn(hidden_layer(x))
        x = self.output_layer(x)
        x = self.out_activation_fn(x)
        x = self.rescale_fn(x)
        
        return x

# ------------------------------------------------------------------------------

class NormalNoiseStrategy:
    def __init__(self, bounds, exploration_noise_ratio=0.1):
        self.low, self.high = bounds 
        self.exploration_noise_ratio = exploration_noise_ratio
        
    def select_action(self, model, state, max_exploration=False):
        if max_exploration:
            noise_scale = self.high
        else:
            noise_scale = self.exploration_noise_ratio * self.high
        
        with torch.no_grad():
            greedy_action = model(state).cpu().detach().data.numpy().squeeze()
            
        noise = np.random.normal(loc=0, scale=noise_scale, size=len(self.high))
        noisy_action = greedy_action + noise
        action = np.clip(noisy_action, self.low, self.high)
        
        return action
   
# -------------------------------------------------------------------------------- 
    
class GreedyStrategy:
    def __init__(self, bounds):
       self.low, self.high = bounds

    def select_action(self, model, state):
        with torch.no_grad():
            greedy_action = model(state).cpu().detach().data.numpy().squeeze()
        action = np.clip(greedy_action, self.low, self.high)
    
        return np.reshape(action, self.high.shape)
    
# -----------------------------------------------------------------------------

class NormalNoiseDecayStrategy:
    def __init__(self, bounds, init_noise_ratio=0.5, min_noise_ratio=0.1, decay_steps=10000):
        self.t = 0
        self.low, self.high = bounds
        self.noise_ratio = init_noise_ratio
        self.init_noise_ratio = init_noise_ratio
        self.min_noise_ratio = min_noise_ratio
        self.decay_steps = decay_steps
        
    def update_noise_ratio(self):
        noise_ratio = 1 - self.t / self.decay_steps
        noise_ratio = (self.init_noise_ratio - self.min_noise_ratio) * noise_ratio + self.min_noise_ratio
        noise_ratio = np.clip(noise_ratio, self.min_noise_ratio, self.init_noise_ratio)
        self.t += 1
        
        return noise_ratio
    
    def select_action(self, model, state, max_exploration=False):
        if max_exploration:
            noise_scale = self.high
        else:
            noise_scale = self.high * self.noise_ratio
            
        with torch.no_grad():
            greedy_action = model(state).cpu().detach().data.numpy().squeeze()
        noise = np.random.normal(loc=0, scale=noise_scale, size=len(self.high))
        noisy_action = greedy_action + noise 
        action = np.clip(noisy_action, self.low, self.high)
        self.noise_ratio = self.update_noise_ratio()
            
        return action
    
# --------------------------------------------------------------------------------

class TwinQValueNetwork(nn.Module):
    def __init__(self,
                 input_dim,
                 output_dim,
                 hidden_dims=(32,32),
                 activation_fn=F.relu):
        super().__init__()
        self.input_layer_a = nn.Linear(input_dim + output_dim, hidden_dims[0])
        self.input_layer_b = nn.Linear(input_dim + output_dim, hidden_dims[0])
        
        self.hidden_layers_a = nn.ModuleList()
        self.hidden_layers_b = nn.ModuleList()
        
        for i in range(len(hidden_dims)-1):
            hidden_layer_a = nn.Linear(hidden_dims[i], hidden_dims[i+1])
            self.hidden_layers_a.append(hidden_layer_a)
            
            hidden_layer_b = nn.Linear(hidden_dims[i], hidden_dims[i+1])
            self.hidden_layers_b.append(hidden_layer_b)
            
        self.output_layer_a = nn.Linear(hidden_dims[-1], 1)
        self.output_layer_b = nn.Linear(hidden_dims[-1], 1)
        
        self.activation_fn = activation_fn
        
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        self.to(self.device)
        
    def format_input(self, state, action):
        x, u = state, action
        
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32, device=self.device)
            x = x.unsqueeze(0)
            
        if not isinstance(u, torch.Tensor):
            u = torch.tensor(u, dtype=torch.float32, device=self.device)
            u = u.unsqueeze(0)
            
        return x, u
    
    def forward(self, state, action):
        x, u = self.format_input(state, action)
        x = torch.cat((x, u), dim=1)
        xa = self.activation_fn(self.input_layer_a(x))
        xb = self.activation_fn(self.input_layer_b(x))
        
        for hidden_layer_a, hidden_layer_b in zip(self.hidden_layers_a, self.hidden_layers_b):
            xa = self.activation_fn(hidden_layer_a(xa))
            xb = self.activation_fn(hidden_layer_b(xb))
            
        xa = self.output_layer_a(xa)
        xb = self.output_layer_b(xb)
        
        return xa, xb
    
    def Qa(self, state, action):
        x, u = self.format_input(state, action)
        x = torch.cat((x, u), dim=1)
        xa = self.activation_fn(self.input_layer_a(x))
        for hidden_layer_a in self.hidden_layers_a:
            xa = self.activation_fn(hidden_layer_a(xa))
        xa = self.output_layer_a(xa)
        
        return xa
    
    def format_experiences(self, experiences):
        states, actions, rewards, next_states, dones = experiences
        states = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions = torch.tensor(actions, dtype=torch.float32, device=self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32, device=self.device) 
        dones = torch.tensor(dones, dtype=torch.bool, device=self.device)
        
        return states, actions, rewards, next_states, dones
    
# -------------------------------------------------------------------------

class QValueSACNetwork(nn.Module):
    def __init__(self,
                input_dim,
                output_dim,
                hidden_dims=(32,32),
                activation_fc=F.relu):
        super().__init__()
        self.activation_fc = activation_fc
        self.input_layer = nn.Linear(input_dim + output_dim, hidden_dims[0])
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_dims)-1):
            hidden_layer = nn.Linear(hidden_dims[i], hidden_dims[i+1])
        self.hidden_layers.append(hidden_layer)
        self.output_layer = nn.Linear(hidden_dims[-1], 1)
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        self.to(self.device)
        
    def _format(self, state, action):
        x, u = state, action
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, device=self.device,dtype=torch.float32)
            x = x.unsqueeze(0)
        if not isinstance(u, torch.Tensor):
            u = torch.tensor(u, device=self.device, dtype=torch.float32)
            u = u.unsqueeze(0)
        return x, u
    
    def forward(self, state, action):
        x, u = self._format(state, action)
        x = self.activation_fc(self.input_layer(torch.cat((x, u), dim=1)))
        for i, hidden_layer in enumerate(self.hidden_layers):
            x = self.activation_fc(hidden_layer(x))
        x = self.output_layer(x)
        
        return x
    
    def format_experiences(self, experiences):
        states, actions, rewards, next_states, dones = experiences
        states = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions = torch.tensor(actions, dtype=torch.float32, device=self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32, device=self.device) 
        dones = torch.tensor(dones, dtype=torch.bool, device=self.device)
        
        return states, actions, rewards, next_states, dones
    
# -------------------------------------------------------------------

class GaussianPolicyNetwork(nn.Module):
    
    def __init__(self, 
                 input_dim, 
                 action_bounds,
                 log_std_min=-20, 
                 log_std_max=2,
                 hidden_dims=(32,32), 
                 activation_fc=F.relu,
                 entropy_lr=0.001):
        super().__init__()
        self.activation_fc = activation_fc
        self.env_min, self.env_max = action_bounds
        
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        self.input_layer = nn.Linear(input_dim, 
                                     hidden_dims[0])
        
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_dims)-1):
            hidden_layer = nn.Linear(
                hidden_dims[i], hidden_dims[i+1])
            self.hidden_layers.append(hidden_layer)

        self.output_layer_mean = nn.Linear(hidden_dims[-1], len(self.env_max))
        self.output_layer_log_std = nn.Linear(hidden_dims[-1], len(self.env_max))

        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda:0"
        self.device = torch.device(device)
        self.to(self.device)

        self.env_min = torch.tensor(self.env_min,
                                    device=self.device, 
                                    dtype=torch.float32)

        self.env_max = torch.tensor(self.env_max,
                                    device=self.device, 
                                    dtype=torch.float32)
        
        self.nn_min = torch.tanh(torch.Tensor([float('-inf')])).to(self.device)
        self.nn_max = torch.tanh(torch.Tensor([float('inf')])).to(self.device)
        self.rescale_fn = lambda x: (x - self.nn_min) * (self.env_max - self.env_min) / \
                                    (self.nn_max - self.nn_min) + self.env_min

        self.target_entropy = -np.prod(self.env_max.shape)
        self.logalpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optimizer = optim.Adam([self.logalpha], lr=entropy_lr)

    def _format(self, state):
        x = state
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, 
                             device=self.device, 
                             dtype=torch.float32)
            x = x.unsqueeze(0)
        return x

    def forward(self, state):
        x = self._format(state)
        x = self.activation_fc(self.input_layer(x))
        for hidden_layer in self.hidden_layers:
            x = self.activation_fc(hidden_layer(x))
        x_mean = self.output_layer_mean(x)
        x_log_std = self.output_layer_log_std(x)
        x_log_std = torch.clamp(x_log_std, 
                                self.log_std_min, 
                                self.log_std_max)
        return x_mean, x_log_std

    def full_pass(self, state, epsilon=1e-6):
        mean, log_std = self.forward(state)

        pi_s = torch.distributions.Normal(mean, log_std.exp())
        pre_tanh_action = pi_s.rsample()
        tanh_action = torch.tanh(pre_tanh_action)
        action = self.rescale_fn(tanh_action)

        log_prob = pi_s.log_prob(pre_tanh_action) - torch.log(
            (1 - tanh_action.pow(2)).clamp(0, 1) + epsilon)
        log_prob = log_prob.sum(dim=1, keepdim=True)

        return action, log_prob, self.rescale_fn(torch.tanh(mean))

    def _update_exploration_ratio(self, greedy_action, action_taken):
        env_min, env_max = self.env_min.cpu().numpy(), self.env_max.cpu().numpy()
        self.exploration_ratio = np.mean(abs((greedy_action - action_taken)/(env_max - env_min)))

    def _get_actions(self, state):
        mean, log_std = self.forward(state)

        action = self.rescale_fn(torch.tanh(torch.distributions.Normal(mean, log_std.exp()).sample()))
        greedy_action = self.rescale_fn(torch.tanh(mean))
        random_action = np.random.uniform(low=self.env_min.cpu().numpy(),
                                          high=self.env_max.cpu().numpy())

        action_shape = self.env_max.cpu().numpy().shape
        action = action.detach().cpu().numpy().reshape(action_shape)
        greedy_action = greedy_action.detach().cpu().numpy().reshape(action_shape)
        random_action = random_action.reshape(action_shape)

        return action, greedy_action, random_action

    def select_random_action(self, state):
        action, greedy_action, random_action = self._get_actions(state)
        self._update_exploration_ratio(greedy_action, random_action)
        return random_action

    def select_greedy_action(self, state):
        action, greedy_action, random_action = self._get_actions(state)
        self._update_exploration_ratio(greedy_action, greedy_action)
        return greedy_action

    def select_action(self, state):
        action, greedy_action, random_action = self._get_actions(state)
        self._update_exploration_ratio(greedy_action, action)
        return action
    
# ---------------------------------------------------------------------------------

    