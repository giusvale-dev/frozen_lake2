import torch.nn as nn
import random
import torch
import torch.nn.functional as F
from collections import namedtuple, deque
import gymnasium as gym
import torch.optim as optim
import math
from itertools import count

# A single transition in the environment (s, a, s', r)
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

def make_env(is_slippery=True, map_name="4x4", render_mode=None):
    env = gym.make("FrozenLake-v1", is_slippery=is_slippery, map_name=map_name, render_mode=render_mode)
    return env

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_n_actions(env:gym.Env):
    return env.action_space.n

def get_n_observations(env:gym.Env):
    return env.observation_space.n

def epsilon_decay(episode, epsilon_min=0.05, epsilon_start=0.9, decay_rate=0.995, num_episodes = 1000):
    
    if episode < num_episodes * 0.1:
        return epsilon_start
    epsilon = epsilon_start * (decay_rate ** episode)
    return max(epsilon, epsilon_min)

class DQN3L(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN3L, self).__init__()
        self.layer1 = nn.Linear(n_observations, 64)
        self.layer2 = nn.Linear(64, 32)
        self.layer3 = nn.Linear(32, n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

class DQN4L(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN4L, self).__init__()
        self.layer1 = nn.Linear(n_observations, 64)
        self.layer2 = nn.Linear(64, 64)
        self.layer3 = nn.Linear(64, 32)
        self.layer4 = nn.Linear(32, n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        return self.layer4(x)
    
class DQN5L(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN5L, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 64)
        self.layer3 = nn.Linear(64, 64)
        self.layer4 = nn.Linear(64, 32)
        self.layer5 = nn.Linear(32, n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        x = F.relu(self.layer4(x))
        return self.layer5(x)


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

def select_action(state, steps_done, n_actions, policy_net, device,
                  eps_start=0.9, eps_end=0.01, decay_rate=0.995, num_episodes=1000):
    # Epsilon decay
    epsilon = epsilon_decay(steps_done, epsilon_min=eps_end, epsilon_start=eps_start, decay_rate=decay_rate, num_episodes=num_episodes)
    
    if random.random() > epsilon:
        with torch.no_grad():
            return policy_net(state).argmax(dim=1).view(1, 1)
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)

def optimize_model(optimizer: optim.Adam, policy_net: nn.Module, memory: ReplayMemory, batch_size=128, gamma=0.95):

    if len(memory) < batch_size:
        return

    transitions = memory.sample(batch_size)
    batch = Transition(*zip(*transitions))
    device = get_device()

    # Mask and tensors
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)),
                                  device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a)
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) = max_a Q(s', a) for all next states using policy_net 
    next_state_values = torch.zeros(batch_size, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = policy_net(non_final_next_states).max(1).values

    # Expected Q value
    expected_state_action_values = reward_batch + (gamma * next_state_values)

    # Loss = Huber loss (smooth L1)
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()


def train(env:gym.Env, policy_net: nn.Module, num_episodes=50, learning_rate=0.1, gamma=0.95, batch_size=128, memory_size=10000):

    device = get_device()
    rewards_per_episodes = [0] * num_episodes
    steps_done = 0
    n_actions = get_n_actions(env)
    n_observations = get_n_observations(env)

    optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
    memory = ReplayMemory(memory_size)

    policy_net.train()
    
    for episode in range(num_episodes):
        state_idx, info = env.reset()

        # One unsqueeze to get shape [1,16 in the 4x4 case]
        state = F.one_hot(torch.tensor(state_idx), num_classes=n_observations).float().to(device).unsqueeze(0)

        total_reward = 0
        done = False

        while not done:
            action = select_action(state, steps_done, n_actions, policy_net, device, num_episodes=num_episodes)
            steps_done += 1

            observation, reward, terminated, truncated, _ = env.step(action.item())
            total_reward += reward
            done = terminated or truncated

            if not done:
                next_state = F.one_hot(torch.tensor(observation), num_classes=n_observations).float().to(device).unsqueeze(0)
            else:
                next_state = None

            memory.push(state, action, next_state, torch.tensor([reward], device=device))

            state = next_state

            optimize_model(optimizer=optimizer, policy_net=policy_net, memory=memory, batch_size=batch_size, gamma=gamma)

        rewards_per_episodes[episode] = (1 if total_reward > 0 else 0)
        

    with torch.no_grad():
        all_states = torch.eye(n_observations, device=device)
        q_values = policy_net(all_states)
        policy = torch.argmax(q_values, dim=1).cpu().numpy()

    return q_values.cpu(), policy, rewards_per_episodes

def run_trained_agent(policy_net: nn.Module, env, num_episodes=10):
    policy_net.eval()
    n_actions = get_n_actions(env)
    n_observations = get_n_observations(env)

    for i in range(num_episodes):
        state_idx, info = env.reset()
        state = F.one_hot(torch.tensor(state_idx), num_classes=n_observations).float().unsqueeze(0)
        done = False
        total_reward = 0

        while not done:
            with torch.no_grad():
                q_values = policy_net(state)
                action = q_values.argmax(dim=1).item()

            observation, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated

            if not done:
                state = F.one_hot(torch.tensor(observation), num_classes=n_observations).float().unsqueeze(0)

        print(f"Episode {i+1}: Total Reward: {total_reward}")
