import random
def epsilon_decay(episode, epsilon_min=0.05, epsilon_start=0.9, num_episodes=1000):
    
    exploration_phase = int(num_episodes * 0.3)
    
    if episode >= num_episodes:
        return epsilon_min
    
    if episode < exploration_phase:
        return epsilon_start

    
    decay_duration = num_episodes - exploration_phase
    decay_progress = (episode - exploration_phase) / decay_duration
    epsilon = epsilon_start - decay_progress * (epsilon_start - epsilon_min)
    
    return max(epsilon, epsilon_min)