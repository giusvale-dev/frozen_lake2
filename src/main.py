from qlearning import train_deterministic, build_env, run_agent, train_stochastic
from dqn import DQN2L, DQN3L, DQN4L, get_device, make_env, train, run_trained_agent
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from util import epsilon_decay


gamma_label = '\u03B3'
alpha_label = '\u03B1'
epsilon_label = '\u03B5'

NUM_EPISODES = 5000
MEMORY_SIZE = 10000
BATCH_SIZE = 64

def plot_epsilon_decay():
    buffer = []
    for i in range(1000):
        epsilon = epsilon_decay(i, epsilon_min=0.05, epsilon_start=0.9, decay_rate=0.995, num_episodes = 1000)
        buffer.append(epsilon)
    
    plt.plot(buffer, label="Epsilon decay")
    plt.xlabel('Episodes')
    plt.ylabel('Epsilon')
    plt.title("Epsilon decay")
    plt.legend(loc='best')
    plt.grid(True)
    
    plt.savefig("epsilon_decay")
    plt.clf()


def plot_cumulative_wins_vs_episode(series_dict, num_episodes, title="Cumulative Wins vs Episode", saveas="cumulative_wins_vs_episode.png"):
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
    plt.title(title)
    plt.legend(loc='best')
    plt.grid(True)
    
    plt.savefig(saveas)
    plt.clf()

def plot_success_rate(series_dict, num_episodes, title="Success Rate vs Episode", saveas="success_rate_vs_episode.png"):
    
    for label, values in series_dict.items():
        # Compute cumulative wins
        cumulative_wins = np.cumsum(values[:num_episodes])
        episodes = np.arange(1, len(cumulative_wins) + 1)
        
        plt.plot(cumulative_wins, label=label)

    plt.xlabel("Episode")
    plt.ylabel("Cumulative Wins")
    plt.title(title)
    plt.legend(loc='best')
    plt.grid(True)
    
    plt.savefig(saveas)
    plt.clf()

def plot_qtable_heatmap(qtable: np.ndarray, title: str, filename: str):
    
    plt.figure(figsize=(10, 6))

    sns.heatmap(qtable, annot=True, fmt=".2f", cmap="viridis", cbar=True)

    plt.title(title)
    plt.xlabel("Actions (0: LEFT, 1: DOWN, 2: RIGHT, 3: UP)")
    plt.ylabel("States")
    plt.tight_layout()

    plt.savefig(filename)
    plt.clf()  # clear after showing to avoid overlapping figures

def dqn_vs_qlearning(stochastic: bool = True, map_name: str="4x4"):

    qlearning_rewards = []
    qlearning_policy = None

    q_learning_env = build_env(stochastic=stochastic, map_name=map_name, render_mode=None)
    # Train Q-learning
    print("Training Q-learning agent...")
    if stochastic:
        _, qlearning_policy, qlearning_rewards = train_stochastic(env=q_learning_env, epsilon_start=1, num_episodes=NUM_EPISODES, learning_rate=0.001, discount_factor=0.995)
    else:
        _, qlearning_policy, qlearning_rewards = train_deterministic(env=q_learning_env, epsilon=1, num_episodes=NUM_EPISODES, discount_factor=0.995)
    print("Q-learning training completed.")

    # Train DQNs
    env2l = make_env(is_slippery=stochastic, map_name=map_name, render_mode=None)
    dqn2l = DQN2L(env2l.observation_space.n, env2l.action_space.n)

    print("Training DQN with 2 layers...")
    _, policy2l, rewards_2l = train(env2l, dqn2l, num_episodes=NUM_EPISODES, learning_rate=0.001,gamma=0.995, batch_size=BATCH_SIZE, memory_size=MEMORY_SIZE)
    print("Training completed for DQN with 2 layers.")

    # env3l = make_env(is_slippery=stochastic, map_name=map_name, render_mode=None)
    # dqn3l = DQN3L(env3l.observation_space.n, env3l.action_space.n)

    # print("Training DQN with 3 layers...")
    # _, policy3l, rewards_3l = train(env3l, dqn3l, num_episodes=NUM_EPISODES, learning_rate=0.1,gamma=0.995, batch_size=BATCH_SIZE, memory_size=MEMORY_SIZE)
    # print("Training completed for DQN with 3 layers.")

    # env4l = make_env(is_slippery=stochastic, map_name=map_name, render_mode=None)
    # dqn4l = DQN4L(env4l.observation_space.n, env4l.action_space.n)

    # print("Training DQN with 4 layers...")
    # _, policy4l, rewards_4l = train(env4l, dqn4l, num_episodes=NUM_EPISODES, learning_rate=0.1,gamma=0.9, batch_size=BATCH_SIZE, memory_size=MEMORY_SIZE)
    # print("Training completed for DQN with 4 layers.")
    
    # Plot training results
    training_series = {
        "DQN 2 layers": rewards_2l,
        # "DQN 3 layers": rewards_3l,
        # "DQN 4 layers": rewards_4l,
        "Q-learning": qlearning_rewards
    }
    
    plot_cumulative_wins_vs_episode(series_dict=training_series, num_episodes=NUM_EPISODES, title="Wins vs Episode (DQNs vs Q-Learning)", saveas=f"dqn_vs_qlearning_training_curve_{map_name}_{"stochastic" if stochastic else "deterministic"}.png")

    env2l = make_env(is_slippery=stochastic, map_name=map_name, render_mode=None)
    running_series2l = run_trained_agent(policy2l, env2l, num_episodes=NUM_EPISODES)

    # running_series3l = run_trained_agent(policy3l, env3l, num_episodes=NUM_EPISODES)

    # running_series4l = run_trained_agent(policy4l, env4l, num_episodes=NUM_EPISODES)

    _, running_qlearning_series = run_agent(q_learning_env, qlearning_policy, num_episodes=NUM_EPISODES)
    
    
    running_series = { 
        "DQN 2 Layers": running_series2l,
        # "DQN 3 Layers": running_series3l,
        # "DQN 4 Layers": running_series4l,
        "Q-learning": running_qlearning_series
    }

    plot_success_rate(series_dict=running_series, num_episodes=NUM_EPISODES, title="Success Rate vs Episode (DQN vs Q-Learning)", saveas=f"dqn_vs_qlearning_success_rate_{map_name}_{"stochastic" if stochastic else "deterministic"}.png")

def main():
    selection = input("Select option\n")


    if selection == "1":
        #plot_epsilon_decay()
        dqn_vs_qlearning(stochastic=True, map_name="4x4")
        #dqn_vs_qlearning(stochastic=True, map_name="8x8")
        #dqn_vs_qlearning(stochastic=True, map_name="8x8")
        #dqn_vs_qlearning(stochastic=True, map_name="16x16")
    elif selection == "2":
        dqn_vs_qlearning(stochastic=True, map_name="8x8")
    else:
        dqn_vs_qlearning(stochastic=False, map_name="4x4")
        #dqn_vs_qlearning(stochastic=False, map_name="8x8")
        #dqn_vs_qlearning(stochastic=False, map_name="16x16")
        


if __name__=="__main__":
    main()