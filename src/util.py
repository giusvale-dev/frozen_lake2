def epsilon_decay(episode, epsilon_min=0.05, epsilon_start=0.9, num_episodes = 1000):
    
    if episode >= num_episodes:
        return epsilon_min
    
    epsilon = epsilon_start - (episode / num_episodes) * (epsilon_start - epsilon_min)
    return epsilon