from qlearning import train_deterministic, build_env, run_agent
import matplotlib.pyplot as plt
import numpy as np

def plot_cumulative_wins_vs_episode(series_dict, num_episodes):
    cumulative_episodes = np.arange(1, num_episodes + 1)
    for label, values in series_dict.items():
        
        # Compute cumulative wins
        cumulative_wins = np.cumsum(values) 

        # Compute losses
        cumulative_losses = cumulative_episodes - cumulative_wins

        # Avoid division by zero
        cumulative_wins =  np.maximum(cumulative_wins, 1)
        cumulative_losses = np.maximum(cumulative_losses, 1)

        # Compute log(win/loss ratio)
        log_win_loss_ratio = np.log(cumulative_wins / cumulative_losses)

        # Plot            
        plt.plot(log_win_loss_ratio, label=label)
    
    plt.xlabel('Episodes')
    plt.ylabel('log(Win/Loss)')
    plt.legend(loc='best')
    plt.grid(True)
    
    plt.savefig("cumulative_wins_vs_episode.png")

def simple_success_rate_plot(win_counts, window_size=100):
    """
    Simple version that just plots the rolling success rate
    """
    rolling_success_rate = []
    for i in range(len(win_counts)):
        start_idx = max(0, i - window_size + 1)
        window_wins = sum(win_counts[start_idx:i+1])
        window_episodes = i - start_idx + 1
        rolling_success_rate.append(window_wins / window_episodes)
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(win_counts) + 1), rolling_success_rate, 'b-', linewidth=2)
    plt.axhline(y=sum(win_counts)/len(win_counts), color='r', linestyle='--', 
                label=f'Overall Success Rate: {sum(win_counts)/len(win_counts):.3f}')
    plt.xlabel('Episode')
    plt.ylabel('Success Rate')
    plt.title(f'Agent Success Rate Over Time (Rolling Window: {window_size})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)
    plt.show()
    
    return rolling_success_rate

def main():

    NUM_EPISODES = 10000

    env = build_env(stochastic=False, map_name="4x4", render_mode=None)
    _, policy_09, rewards_per_episodes_gamma_09 = train_deterministic(env=env, epsilon=0.995, num_episodes=NUM_EPISODES, discount_factor=0.9)

    env = build_env(stochastic=False, map_name="4x4", render_mode=None)
    _, policy_05, rewards_per_episodes_gamma_05 = train_deterministic(env=env, epsilon=0.995, num_episodes=NUM_EPISODES, discount_factor=0.5)

    env = build_env(stochastic=False, map_name="4x4", render_mode=None)
    _, policy_01, rewards_per_episodes_gamma_01 = train_deterministic(env=env, epsilon=0.995, num_episodes=NUM_EPISODES, discount_factor=0.1)
    
    env = build_env(stochastic=False, map_name="4x4", render_mode=None)
    win_counts_09, loss_counts_09 = run_agent(env, policy_09, num_episodes=NUM_EPISODES)

    env = build_env(stochastic=False, map_name="4x4", render_mode=None)
    win_counts_05, loss_counts_05 = run_agent(env, policy_05, num_episodes=NUM_EPISODES)

    env = build_env(stochastic=False, map_name="4x4", render_mode=None)
    win_counts_01, loss_counts_01 = run_agent(env, policy_01, num_episodes=NUM_EPISODES)


    gamma_label = '\u03B3'

    training_series = { 
        gamma_label + "= 0.9": rewards_per_episodes_gamma_09,
        gamma_label + "= 0.5": rewards_per_episodes_gamma_05,
        gamma_label + "= 0.1": rewards_per_episodes_gamma_01
    }

    

    plot_cumulative_wins_vs_episode(series_dict=training_series, num_episodes=NUM_EPISODES)

if __name__=="__main__":
    main()