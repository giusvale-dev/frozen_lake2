import gymnasium as gym
from gymnasium import Env
import numpy as np
import math

def build_env(deterministic: bool, map_name: str, render_mode: str) -> Env:
    return gym.make('FrozenLake-v1', map_name=map_name, is_slippery=deterministic, render_mode=render_mode)

def softmax(q_values, k):
    q_scaled = k * q_values
    q_scaled -= np.max(q_scaled)
    exp_q = np.exp(q_scaled)
    return exp_q / np.sum(exp_q)

def epsilon_decay(episode, epsilon_min=0.01, epsilon_start=0.9, decay_rate=0.001):
    epsilon = epsilon_min + (epsilon_start - epsilon_min) * math.exp(-decay_rate * episode)
    return epsilon


def _train_deterministic(
    env: Env,
    num_episodes: int = 1000,
    discount_factor: float = 0.95,
    epsilon: float = 0.9,
    k_min: float = 0.1,
    k_max: float = 100,
    epsilon_greedy: bool = True
    
):
    n_states = env.observation_space.n
    n_actions = env.action_space.n

    # Step 1
    # for each s, a initialize table entry Qˆ[s, a] ← 0
    Q = np.zeros((n_states, n_actions)) 

    # Random Generator used in softmax strategy
    rng = np.random.default_rng() # Generator instance

    rewards_per_episodes = [0] * num_episodes # initialize to 0

    # Iterate over num_episodes
    for episode in range(num_episodes):
        
        done = False

        # Step 2 - observe the current state
        state, _ = env.reset()

        # Step 3 - until termination condition
        while not done:

            # How does agent chooses next action? Eploitation vs Exploration
            
            # ε-greedy exploration
            if epsilon_greedy:

                # ε decreases over time (prefer exploration first, then exploitation as suggested in the slides)

                epsilon = epsilon_decay(episode=episode, epsilon_start=epsilon)

                if np.random.rand() < epsilon:
                    # Exploration
                    action = env.action_space.sample()
                else:
                    # Exploitation: Chooses an action that maximize Qˆ(s, a)
                    action = np.argmax(Q[state, :])
            else:
                
                # Softmax strategy as suggested in the slides k may increase over time (first exploration, then exploitation)
                k = k_min + (k_max - k_min) * (episode / num_episodes)

                #softmax function assigns different probabilities to each action — depending on their Q-values.
                action_probs = softmax(Q[state], k) 
                
                # randomly selects an action using a softmax probability distribution over actions
                action = rng.choice(n_actions, p=action_probs)

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

    return Q, policy, rewards_per_episodes

def train_deterministic_epsilon_greedy(env: Env,
    num_episodes: int = 1000,
    discount_factor: float = 0.95,
    epsilon: float = 0.9,
    ):
    return _train_deterministic(env=env, num_episodes=num_episodes, discount_factor=discount_factor, epsilon=epsilon, epsilon_greedy=True)

def train_deterministic_softmax(env: Env,
    num_episodes: int = 1000,
    discount_factor: float = 0.95,
    k_min: float = 0.1,
    k_max: float = 100,
    ):
    return _train_deterministic(env=env, num_episodes=num_episodes, discount_factor=discount_factor, k_min=k_min, k_max=k_max, epsilon_greedy=False)