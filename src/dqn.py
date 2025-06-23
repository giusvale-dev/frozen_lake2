import torch.nn as nn
import random
import torch
import torch.nn.functional as F
from collections import namedtuple, deque
import gymnasium as gym
import torch.optim as optim
import math
from itertools import count
from util import epsilon_decay

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

class DQN2L(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN2L, self).__init__()
        self.layer1 = nn.Linear(n_observations, 16)
        self.layer2 = nn.Linear(16, n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        return self.layer2(x)

class DQN3L(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN3L, self).__init__()
        self.layer1 = nn.Linear(n_observations, 32)
        self.layer2 = nn.Linear(32, 16)
        self.layer3 = nn.Linear(16, n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)
    
class DQN4L(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN4L, self).__init__()
        self.layer1 = nn.Linear(n_observations, 64)
        self.layer2 = nn.Linear(64, 32)
        self.layer3 = nn.Linear(32, 16)
        self.layer4 = nn.Linear(16, n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        return self.layer4(x)


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
                  eps_start=0.995, eps_end=0.05, num_episodes=1000):
    # Epsilon decay
    epsilon = epsilon_decay(steps_done, epsilon_min=eps_end, epsilon_start=eps_start, num_episodes=num_episodes)
    
    if random.random() > epsilon:
        with torch.no_grad():
            return policy_net(state).argmax(dim=1).view(1, 1)
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)

# def optimize_model(optimizer: optim.Adam, policy_net: nn.Module, memory: ReplayMemory, batch_size=128, gamma=0.95):

#     # Ensure that we have sufficient experience from a full batch, otherwise skip training
#     if len(memory) < batch_size:
#         return

#     # Randomly take a batch of tranistions from the ReplayMemory 
#     transitions = memory.sample(batch_size)

#     # * operator unpack the list so we will have:
#     # Transition('state1', 'action1', 'next_state1', 'reward1')
#     # Transition('state2', 'action2', 'next_state2', 'reward2')
#     # ...
#     # Then zip converts a list of tuples into a tuples of lists, e,g:
#     # 
#     # ( (state1, state2, ...),     # all states
#     #   (action1, action2, ...),   # all actions
#     #   (reward1, reward2, ...),   # all rewards
#     #   (next_state1, ...),        # all next_states
#     #   (reward1, reward2, ...)    # all rewards
#     # )
#     # At the end of this procedure batch will be a new Transition object where each field is a batch of values (list of tensors)
#     # The final result will be:
#     # batch.state       # tuple of all states
#     # batch.action      # tuple of all actions
#     # batch.reward      # tuple of all rewards
#     # batch.next_state  # tuple of all next_states

#     batch = Transition(*zip(*transitions))

#     # return torch.device (CPU or CUDA)
#     device = get_device()

#     # Mask and tensors
#     non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)),
#                                   device=device, dtype=torch.bool)
#     non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
#     state_batch = torch.cat(batch.state)
#     action_batch = torch.cat(batch.action)
#     reward_batch = torch.cat(batch.reward)

#     # Compute Q(s_t, a)
#     state_action_values = policy_net(state_batch).gather(1, action_batch)

#     # Compute V(s_{t+1}) = max_a Q(s', a) for all next states using policy_net 
#     next_state_values = torch.zeros(batch_size, device=device)
#     with torch.no_grad():
#         next_state_values[non_final_mask] = policy_net(non_final_next_states).max(1).values

#     # Expected Q value
#     expected_state_action_values = reward_batch + (gamma * next_state_values)

#     criterion = nn.MSELoss()
#     loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

#     # Optimize the model
#     optimizer.zero_grad()
#     loss.backward()
#     # In-place gradient clipping
#     torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
#     optimizer.step()

def optimize_model(optimizer, policy_net, memory, batch_size=128, gamma=0.95):
    if len(memory) < batch_size:
        return

    transitions = memory.sample(batch_size)
    batch = Transition(*zip(*transitions))
    device = next(policy_net.parameters()).device

    state_batch = torch.cat(batch.state).to(device)
    action_batch = torch.cat(batch.action).to(device)
    reward_batch = torch.cat(batch.reward).to(device)

    non_final_mask = torch.tensor([s is not None for s in batch.next_state], device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None]).to(device)

    state_action_values = policy_net(state_batch).gather(1, action_batch)

    next_state_values = torch.zeros(batch_size, device=device)
    if non_final_next_states.size(0) > 0:
        next_state_values[non_final_mask] = policy_net(non_final_next_states).max(1)[0].detach()

    expected_state_action_values = reward_batch + gamma * next_state_values

    loss_fn = nn.MSELoss()
    loss = loss_fn(state_action_values.squeeze(), expected_state_action_values)

    optimizer.zero_grad()
    loss.backward()
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
            action = select_action(state, steps_done, n_actions, policy_net, device, num_episodes=num_episodes, eps_start=1)
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

def run_trained_agent(policy, env, num_episodes=1000):
    rewards_per_episodes = [0] * num_episodes
    
    n_observations = get_n_observations(env)

    for i in range(num_episodes):
        state_idx, info = env.reset()
        state = F.one_hot(torch.tensor(state_idx), num_classes=n_observations).float().unsqueeze(0)
        done = False
        total_reward = 0

        while not done:
            with torch.no_grad():
                action = policy[state_idx]

            observation, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated

            if not done:
                state_idx = observation

        rewards_per_episodes[i] = total_reward
    return rewards_per_episodes