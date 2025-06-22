import gymnasium as gym
from gymnasium import Env
import numpy as np
import math

def build_env(stochastic: bool, map_name: str, render_mode: str) -> Env:
    return gym.make('FrozenLake-v1', map_name=map_name, is_slippery=stochastic, render_mode=render_mode)

def epsilon_decay(episode, epsilon_min=0.05, epsilon_start=0.9, decay_rate=0.995, num_episodes = 1000):
    
    if episode < num_episodes * 0.1:
        return epsilon_start
    epsilon = epsilon_start * (decay_rate ** episode)
    return max(epsilon, epsilon_min)

def train_deterministic(
    env: Env,
    num_episodes: int = 1000,
    discount_factor: float = 0.95,
    epsilon: float = 0.9
    
):
    n_states = env.observation_space.n
    n_actions = env.action_space.n

    # Step 1
    # for each s, a initialize table entry Qˆ[s, a] ← 0
    Q = np.zeros((n_states, n_actions)) 

    rewards_per_episodes = [0] * num_episodes # initialize to 0

    
    # Iterate over num_episodes
    for episode in range(num_episodes):
        
        epsilon = epsilon_decay(episode=episode, epsilon_start=epsilon, num_episodes=num_episodes)
        done = False
    
        # Step 2 - observe the current state
        state, _ = env.reset()

        # Step 3 - until termination condition
        while not done:

            # How does agent chooses next action? Eploitation vs Exploration
            
            # ε-greedy exploration
            if np.random.rand() < epsilon:    
                # Exploration
                action = env.action_space.sample()
            else:
                # Exploitation: Chooses an action that maximize Qˆ(s, a)
                action = np.argmax(Q[state, :])
            
            # Execute the given action
            next_state, reward, truncated, terminated, _ = env.step(action)

            # Update done to include the truncated case (time limiting)
            done = terminated or truncated
            
            # Update the table entry
            Q[state, action] = reward + discount_factor * np.max(Q[next_state, :])
            
            # Update the current state
            state = next_state
        
            # Overwrite the item at index episode with value 1
            if reward == 1:
                rewards_per_episodes[episode] = 1

            
            
    # For each state s (each row in Q), it finds the action index with the highest Q-value
    policy = np.argmax(Q, axis=1) # policy is a 1D array of length n_states 

    env.close()
    return Q, policy, rewards_per_episodes

def train_stochastic(
    env: Env,
    num_episodes: int = 1000,
    discount_factor: float = 0.95,
    epsilon: float = 0.9,
    learning_rate: float = 0.1
):
    n_states = env.observation_space.n
    n_actions = env.action_space.n

    Q = np.zeros((n_states, n_actions))
    rewards_per_episodes = [0] * num_episodes

    for episode in range(num_episodes):

        epsilon = epsilon_decay(episode=episode, epsilon_start=epsilon, num_episodes=num_episodes)
        done = False
        state, _ = env.reset()

        while not done:
            # ε-greedy action selection
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(Q[state, :])

            next_state, reward, truncated, terminated, _ = env.step(action)
            done = terminated or truncated

            # Q-learning update with learning rate alpha
            Q[state, action] = Q[state, action] + learning_rate * (
                    reward + discount_factor * np.max(Q[next_state, :]) - Q[state, action]
                )
            
            state = next_state

            if reward == 1:
                rewards_per_episodes[episode] = 1

    policy = np.argmax(Q, axis=1)

    env.close()
    return Q, policy, rewards_per_episodes


def run_agent(env, policy, num_episodes):
    results = []
    for _ in range(num_episodes):
        done = False
        state = env.reset()[0]
        while not done:
            action = policy[state]
            new_state, reward, terminated, truncated, _ = env.step(action)
            state = new_state
            done = terminated or truncated
        results.append(1 if reward > 0 else 0)
    success_rate = sum(results) / num_episodes
    return success_rate, results