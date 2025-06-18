from qlearning import train_deterministic_epsilon_greedy, train_deterministic_softmax, build_env
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
        cumulative_losses = np.maximum(cumulative_losses, 1)

        cumulative_wins = np.maximum(cumulative_wins, 1)
        
        cumulative_losses = np.maximum(cumulative_losses, 1)

        # Compute log(win/loss ratio)
        log_win_loss_ratio = np.log(cumulative_wins / cumulative_losses)

        # Plot            
        plt.plot(log_win_loss_ratio, label=label)
    
    plt.xlabel('Episodes')
    plt.ylabel('Log(Win/Loss Ratio)')
    plt.legend(loc='best')
    plt.grid(True)
    #plt.ylim(-10, 0)
    plt.savefig("cumulative_wins_vs_episode.png")




def main():
    env = build_env(deterministic=True, map_name="4x4", render_mode=None)
    Q, policy, rewards_per_episodes = train_deterministic_epsilon_greedy(env=env, epsilon=0.9, num_episodes=10000)

    series = { 
        "Agent A": rewards_per_episodes
    }

    print(rewards_per_episodes)

    plot_cumulative_wins_vs_episode(series_dict=series, num_episodes=10000)





if __name__=="__main__":
    main()



