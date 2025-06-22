from qlearning import train_deterministic, build_env, run_agent, train_stochastic
from dqn import DQN2L, DQN3L, DQN4L, get_device, make_env, train, run_trained_agent
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

gamma_label = '\u03B3'
alpha_label = '\u03B1'
epsilon_label = '\u03B5'
NUM_EPISODES = 3000
MEMORY_SIZE = 5000

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

def non_stochastic_analysis():

    env = build_env(stochastic=False, map_name="4x4", render_mode=None)
    _, policy_09, rewards_per_episodes_gamma_09 = train_deterministic(env=env, epsilon=0.995, num_episodes=NUM_EPISODES, discount_factor=0.9)

    env = build_env(stochastic=False, map_name="4x4", render_mode=None)
    _, policy_05, rewards_per_episodes_gamma_05 = train_deterministic(env=env, epsilon=0.995, num_episodes=NUM_EPISODES, discount_factor=0.5)

    env = build_env(stochastic=False, map_name="4x4", render_mode=None)
    _, policy_01, rewards_per_episodes_gamma_01 = train_deterministic(env=env, epsilon=0.995, num_episodes=NUM_EPISODES, discount_factor=0.1)
    
    env = build_env(stochastic=False, map_name="4x4", render_mode=None)
    _, result1 = run_agent(env, policy_09, num_episodes=NUM_EPISODES)

    env = build_env(stochastic=False, map_name="4x4", render_mode=None)
    _, result2 = run_agent(env, policy_05, num_episodes=NUM_EPISODES)

    env = build_env(stochastic=False, map_name="4x4", render_mode=None)
    _, result3 = run_agent(env, policy_01, num_episodes=NUM_EPISODES)

    training_series = { 
        gamma_label + "= 0.9 (train)": rewards_per_episodes_gamma_09,
        gamma_label + "= 0.5 (train)": rewards_per_episodes_gamma_05,
        gamma_label + "= 0.1 (train)": rewards_per_episodes_gamma_01
    }

    plot_cumulative_wins_vs_episode(series_dict=training_series, num_episodes=NUM_EPISODES)

    running_series = { 
        gamma_label + "= 0.9 (eval)": result1,
        gamma_label + "= 0.5 (eval)": result2,
        gamma_label + "= 0.1 (eval)": result3
    }

    plot_success_rate(series_dict=running_series, num_episodes=NUM_EPISODES)

def stochastic_analysis_learning_rate():
    
    env = build_env(stochastic=True, map_name="4x4", render_mode=None)
    Q09, policy_09, rewards_per_episodes_alpha_09 = train_stochastic(env=env, epsilon=0.995, num_episodes=NUM_EPISODES, learning_rate=0.9)

    env = build_env(stochastic=True, map_name="4x4", render_mode=None)
    _, policy_05, rewards_per_episodes_alpha_05 = train_stochastic(env=env, epsilon=0.995, num_episodes=NUM_EPISODES, learning_rate=0.5)

    env = build_env(stochastic=True, map_name="4x4", render_mode=None)
    _, policy_01, rewards_per_episodes_alpha_01 = train_stochastic(env=env, epsilon=0.995, num_episodes=NUM_EPISODES, learning_rate=0.1)
    
    env = build_env(stochastic=True, map_name="4x4", render_mode=None)
    _, result1 = run_agent(env, policy_09, num_episodes=NUM_EPISODES)

    env = build_env(stochastic=True, map_name="4x4", render_mode=None)
    _, result2 = run_agent(env, policy_05, num_episodes=NUM_EPISODES)

    env = build_env(stochastic=True, map_name="4x4", render_mode=None)
    _, result3 = run_agent(env, policy_01, num_episodes=NUM_EPISODES)

    training_series = { 
        alpha_label + "= 0.9, " + gamma_label + "=0.95, " + epsilon_label + "0.9   (train)": rewards_per_episodes_alpha_09,
        alpha_label + "= 0.5, " + gamma_label + "=0.95, " + epsilon_label + "0.9   (train)": rewards_per_episodes_alpha_05,
        alpha_label + "= 0.1, " + gamma_label + "=0.95, " + epsilon_label + "0.9   (train)": rewards_per_episodes_alpha_01
    }

    plot_cumulative_wins_vs_episode(series_dict=training_series, num_episodes=NUM_EPISODES, title="Cumulative Wins vs Episode (Learning Rate Analysis)", saveas="cumulative_wins_vs_episode_learning_rate.png")

    running_series = { 
        alpha_label + "= 0.9, " + gamma_label + "=0.95, " + epsilon_label + "0.9   (eval)": result1,
        alpha_label + "= 0.5, " + gamma_label + "=0.95, " + epsilon_label + "0.9   (eval)": result2,
        alpha_label + "= 0.1, " + gamma_label + "=0.95, " + epsilon_label + "0.9   (eval)": result3
    }

    plot_success_rate(series_dict=running_series, num_episodes=NUM_EPISODES, title="Success Rate vs Episode (Learning Rate Analysis)", saveas="success_rate_vs_episode_learning_rate.png")

    plot_qtable_heatmap(qtable=Q09, title="Q-table Heatmap (Learning Rate 0.9)", filename="qtable_heatmap_learning_rate_09.png")


def stochastic_analysis_discount_factor():
    
    env = build_env(stochastic=True, map_name="4x4", render_mode=None)
    Q09, policy_09, rewards_per_episodes_alpha_09 = train_stochastic(env=env, epsilon=0.995, num_episodes=NUM_EPISODES, discount_factor=0.9)

    env = build_env(stochastic=True, map_name="4x4", render_mode=None)
    _, policy_05, rewards_per_episodes_alpha_05 = train_stochastic(env=env, epsilon=0.995, num_episodes=NUM_EPISODES, discount_factor=0.5)

    env = build_env(stochastic=True, map_name="4x4", render_mode=None)
    _, policy_01, rewards_per_episodes_alpha_01 = train_stochastic(env=env, epsilon=0.995, num_episodes=NUM_EPISODES, discount_factor=0.1)
    
    env = build_env(stochastic=True, map_name="4x4", render_mode=None)
    _, result1 = run_agent(env, policy_09, num_episodes=NUM_EPISODES)

    env = build_env(stochastic=True, map_name="4x4", render_mode=None)
    _, result2 = run_agent(env, policy_05, num_episodes=NUM_EPISODES)

    env = build_env(stochastic=True, map_name="4x4", render_mode=None)
    _, result3 = run_agent(env, policy_01, num_episodes=NUM_EPISODES)

    training_series = { 
        gamma_label + "= 0.9, " + alpha_label + "=0.1, " + epsilon_label + "0.9   (train)": rewards_per_episodes_alpha_09,
        gamma_label + "= 0.5, " + alpha_label + "=0.1, " + epsilon_label + "0.9   (train)": rewards_per_episodes_alpha_05,
        gamma_label + "= 0.1, " + alpha_label + "=0.1, " + epsilon_label + "0.9   (train)": rewards_per_episodes_alpha_01
    }

    plot_cumulative_wins_vs_episode(series_dict=training_series, num_episodes=NUM_EPISODES, title="Cumulative Wins vs Episode (Discount Factor Analysis)", saveas="cumulative_wins_vs_episode_discount_factor.png")

    running_series = { 
        gamma_label + "= 0.9, " + alpha_label + "=0.1, " + epsilon_label + "0.9   (eval)": result1,
        gamma_label + "= 0.5, " + alpha_label + "=0.1, " + epsilon_label + "0.9   (eval)": result2,
        gamma_label + "= 0.1, " + alpha_label + "=0.1, " + epsilon_label + "0.9   (eval)": result3
    }

    plot_success_rate(series_dict=running_series, num_episodes=NUM_EPISODES, title="Success Rate vs Episode (Discount Factor Analysis)", saveas="success_rate_vs_episode_discount_factor.png")

def dqn_vs_qlearning(stochastic: bool = True, map_name: str="4x4"):

    qlearning_rewards = []
    qlearning_policy = None

    q_learning_env = build_env(stochastic=stochastic, map_name=map_name, render_mode=None)
    # Train Q-learning
    print("Training Q-learning agent...")
    if stochastic:
        _, qlearning_policy, qlearning_rewards = train_stochastic(env=q_learning_env, epsilon=0.995, num_episodes=NUM_EPISODES, learning_rate=0.1, discount_factor=0.9)
    else:
        _, qlearning_policy, qlearning_rewards = train_deterministic(env=q_learning_env, epsilon=0.995, num_episodes=NUM_EPISODES, discount_factor=0.9)
    print("Q-learning training completed.")

    # Train DQNs
    env2l = make_env(is_slippery=stochastic, map_name=map_name, render_mode=None)
    dqn2l = DQN2L(env2l.observation_space.n, env2l.action_space.n)

    print("Training DQN with 2 layers...")
    _, policy2l, rewards_2l = train(env2l, dqn2l, num_episodes=NUM_EPISODES, learning_rate=0.1,gamma=0.9, batch_size=32, memory_size=MEMORY_SIZE)
    print("Training completed for DQN with 2 layers.")

    env3l = make_env(is_slippery=stochastic, map_name=map_name, render_mode=None)
    dqn3l = DQN3L(env3l.observation_space.n, env3l.action_space.n)

    print("Training DQN with 3 layers...")
    _, policy3l, rewards_3l = train(env3l, dqn3l, num_episodes=NUM_EPISODES, learning_rate=0.1,gamma=0.9, batch_size=32, memory_size=MEMORY_SIZE)
    print("Training completed for DQN with 3 layers.")

    # env4l = make_env(is_slippery=stochastic, map_name=map_name, render_mode=None)
    # dqn4l = DQN4L(env4l.observation_space.n, env4l.action_space.n)

    # print("Training DQN with 4 layers...")
    # _, policy4l, rewards_4l = train(env4l, dqn4l, num_episodes=NUM_EPISODES, learning_rate=0.1,gamma=0.9, batch_size=32, memory_size=MEMORY_SIZE)
    # print("Training completed for DQN with 4 layers.")
    
    # Plot training results
    training_series = {
        "DQN 2 layers": rewards_2l,
        "DQN 3 layers": rewards_3l,
        # "DQN 4 layers": rewards_4l,
        "Q-learning": qlearning_rewards
    }
    
    plot_cumulative_wins_vs_episode(series_dict=training_series, num_episodes=NUM_EPISODES, title="Wins vs Episode (DQNs vs Q-Learning)", saveas="dqn_vs_qlearning_training_curve.png")

    running_series2l = run_trained_agent(policy2l, env2l, num_episodes=NUM_EPISODES)

    running_series3l = run_trained_agent(policy3l, env3l, num_episodes=NUM_EPISODES)

    # running_series4l = run_trained_agent(policy4l, env4l, num_episodes=NUM_EPISODES)

    _, running_qlearning_series = run_agent(q_learning_env, qlearning_policy, num_episodes=NUM_EPISODES)
    
    
    running_series = { 
        "DQN 2 Layers": running_series2l,
        "DQN 3 Layers": running_series3l,
        # "DQN 4 Layers": running_series4l,
        "Q-learning": running_qlearning_series
    }

    plot_success_rate(series_dict=running_series, num_episodes=NUM_EPISODES, title="Success Rate vs Episode (DQN vs Q-Learning)", saveas="dqn_vs_qlearning_success_rate.png")

    


def stocastic_analysis_dqn():
   
    env2l = make_env(is_slippery=True, map_name="4x4", render_mode=None)
    dqn2l = DQN2L(env2l.observation_space.n, env2l.action_space.n)

    print("Training DQN with 2 layers...")
    _, policy2l, rewards_2l = train(env2l, dqn2l, num_episodes=NUM_EPISODES, learning_rate=0.1,gamma=0.9, batch_size=32, memory_size=MEMORY_SIZE)
    print("Training completed for DQN with 2 layers.")

    env3l = make_env(is_slippery=True, map_name="4x4", render_mode=None)
    dqn3l = DQN3L(env3l.observation_space.n, env3l.action_space.n)

    print("Training DQN with 3 layers...")
    _, policy3l, rewards_3l = train(env3l, dqn3l, num_episodes=NUM_EPISODES, learning_rate=0.1,gamma=0.9, batch_size=32, memory_size=MEMORY_SIZE)
    print("Training completed for DQN with 3 layers.")

    env4l = make_env(is_slippery=True, map_name="4x4", render_mode=None)
    dqn4l = DQN4L(env4l.observation_space.n, env4l.action_space.n)

    print("Training DQN with 4 layers...")
    _, policy4l, rewards_4l = train(env4l, dqn4l, num_episodes=NUM_EPISODES, learning_rate=0.1,gamma=0.9, batch_size=32, memory_size=MEMORY_SIZE)
    print("Training completed for DQN with 4 layers.")

    training_series = { 
        "DQN 2 Layers": rewards_2l,
        "DQN 3 Layers": rewards_3l,
        "DQN 4 Layers": rewards_4l
    }

    running_series2l = run_trained_agent(policy2l, env2l, num_episodes=NUM_EPISODES)

    running_series3l = run_trained_agent(policy3l, env3l, num_episodes=NUM_EPISODES)

    running_series4l = run_trained_agent(policy4l, env4l, num_episodes=NUM_EPISODES)
    
    
    running_series = { 
        "DQN 2 Layers": running_series2l,
        "DQN 3 Layers": running_series3l,
        "DQN 4 Layers": running_series4l
    }

    plot_success_rate(series_dict=running_series, num_episodes=NUM_EPISODES, title="Success Rate vs Episode (DQN Analysis)", saveas="success_rate_vs_episode_dqn.png")

    plot_cumulative_wins_vs_episode(series_dict=training_series, num_episodes=NUM_EPISODES, title="Cumulative Wins vs Episode (DQN Analysis)", saveas="dqn_analysis.png")


def main():
    selection = input("Select 1 for deterministic analysys, 2 for stochastic analysis")


    if selection == "1":
        non_stochastic_analysis()
    elif selection == "2":
        stochastic_analysis_learning_rate()
        stochastic_analysis_discount_factor()
    elif selection == "3":
        dqn_vs_qlearning()
    else:
        print("Invalid selection. Please choose 1, 2 or 3.")


if __name__=="__main__":
    main()