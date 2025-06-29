import random
import math

def epsilon_decay(episode, epsilon_min=0.05, epsilon_start=0.9, num_episodes=1000, min_exploration_phase = 0.3):
    
    exploration_phase = int(num_episodes * min_exploration_phase)
    
    if episode >= num_episodes:
        return epsilon_min
    
    if episode < exploration_phase:
        return epsilon_start

    
    decay_duration = num_episodes - exploration_phase
    decay_progress = (episode - exploration_phase) / decay_duration
    epsilon = epsilon_start - decay_progress * (epsilon_start - epsilon_min)
    
    return max(epsilon, epsilon_min)

def epsilon_decay_exp(episode, epsilon_min=0.05, epsilon_start=0.9, num_episodes=1000):
    
    decay_rate = math.log(epsilon_start / epsilon_min) / num_episodes
    epsilon = epsilon_min + (epsilon_start - epsilon_min) * math.exp(-decay_rate * episode)
    return epsilon