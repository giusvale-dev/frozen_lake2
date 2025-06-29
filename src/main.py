from qlearning import train_deterministic, build_env, run_agent, train_stochastic
from dqn import DQN, make_env, train, run_trained_agent
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from util import epsilon_decay

def plot_epsilon_decay(title: str, saveas: str, num_episodes):
    buffer = []
    for i in range(num_episodes):
        epsilon = epsilon_decay(i, epsilon_min=0.05, epsilon_start=0.9,  num_episodes = num_episodes)
        buffer.append(epsilon)
    
    plt.plot(buffer, label="Epsilon decay")
    plt.xlabel('Episodes')
    plt.ylabel('Epsilon')
    plt.title(title)
    plt.legend(loc='best')
    plt.grid(True)
    
    plt.savefig(saveas)
    plt.clf()

def plot_wins_vs_episodes(series_dict, num_episodes, title="Wins vs Episode", saveas="wins_vs_episode.png"):
    
    for label, values in series_dict.items():
        # Compute cumulative wins
        cumulative_wins = np.cumsum(values[:num_episodes])
        
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
    plt.clf()

def dqn_vs_qlearning_4x4(stochastic: bool):

    qlearning_rewards = []
    qlearning_policy = None
    num_episodes = 5000
    batch_size = 32
    memory_size = 10000

    q_learning_env = build_env(stochastic=stochastic, map_name="4x4", render_mode=None)
    # Train Q-learning
    print("Training Q-learning agent...")
    if stochastic:
        _, qlearning_policy, qlearning_rewards = train_stochastic(env=q_learning_env, epsilon_start=1, num_episodes=num_episodes, learning_rate=0.1, discount_factor=0.995)
    else:
        _, qlearning_policy, qlearning_rewards = train_deterministic(env=q_learning_env, epsilon_start=1, num_episodes=num_episodes, discount_factor=0.9)
    print("Q-learning training completed.")

    # Train DQNs
    env2l = make_env(is_slippery=stochastic, map_name="4x4", render_mode=None)
    dqn2l = DQN(input_states=env2l.observation_space.n, hidden_nodes=env2l.observation_space.n, out_states=env2l.action_space.n)

    print("Training DQN...")
    policy2l = None
    rewards_2l = None
    if not stochastic:
        _, policy2l, rewards_2l = train(env2l, dqn2l, num_episodes=num_episodes, learning_rate=0.001, gamma=0.9, batch_size=batch_size, memory_size=memory_size)
    else:
        _, policy2l, rewards_2l = train(env2l, dqn2l, num_episodes=num_episodes, learning_rate=0.001, gamma=0.995, batch_size=batch_size, memory_size=memory_size)
    print("Training completed for DQN")

    # Plot training results
    training_series = {
        "DQN": rewards_2l,
        "Q-learning": qlearning_rewards
    }
    
    plot_wins_vs_episodes(series_dict=training_series, num_episodes=num_episodes, title="Wins vs Episode (Train)", saveas=f"dqn_vs_qlearning_training_4x4_{"stochastic" if stochastic else "deterministic"}.png")

    print("Evaluation DQN...")
    env2l = make_env(is_slippery=stochastic, map_name="4x4", render_mode=None)
    running_series2l = run_trained_agent(policy2l, env2l, num_episodes=num_episodes)
    accuracy = np.sum(running_series2l)/num_episodes
    
    print(f"DQN: accuracy after training is {accuracy * 100:.2f}%")
    print("Evaluation completed for DQN")

    print("Evaluation of Q-Learning...")
    _, running_qlearning_series = run_agent(q_learning_env, qlearning_policy, num_episodes=num_episodes)
    accuracy = np.sum(running_qlearning_series)/num_episodes
    print(f"Q-Learning: accuracy after training is {accuracy * 100:.2f}%")
    print("Evaluation completed for Q-Learning")
    
    if not stochastic:
        # In deterministic case to avoid the overlap we generate 2 plots one for each curve
        running_series = { 
            "DQN ": running_series2l
        }
        plot_wins_vs_episodes(series_dict=running_series, num_episodes=num_episodes, title="Wins vs Episode (Evaluation)", saveas=f"dqn_vs_qlearning_evaluation_4x4_deterministic_dqn.png")
        running_series.clear()

        running_series = { 
            "Q-learning": running_qlearning_series
        }
        plot_wins_vs_episodes(series_dict=running_series, num_episodes=num_episodes, title="Wins vs Episode (Evaluation)", saveas=f"dqn_vs_qlearning_evaluation_4x4_deterministic_qlearning.png")
    else:
        running_series = { 
            "DQN ": running_series2l,
            "Q-learning": running_qlearning_series
        }
        plot_wins_vs_episodes(series_dict=running_series, num_episodes=num_episodes, title="Wins vs Episode (Evaluation)", saveas=f"dqn_vs_qlearning_evaluation_4x4_stochastic.png") 
   
def dqn_vs_qlearning_8x8(stochastic: bool):

    qlearning_rewards = []
    qlearning_policy = None
    num_episodes = 20000
    batch_size = 64
    memory_size = 10000

    q_learning_env = build_env(stochastic=stochastic, map_name="8x8", render_mode=None)
    # Train Q-learning
    print("Training Q-learning agent...")
    if stochastic:
        _, qlearning_policy, qlearning_rewards = train_stochastic(env=q_learning_env, epsilon_start=1, num_episodes=num_episodes, learning_rate=0.1, discount_factor=0.9)
    else:
        _, qlearning_policy, qlearning_rewards = train_deterministic(env=q_learning_env, epsilon_start=1, num_episodes=num_episodes, discount_factor=0.995)
    print("Q-learning training completed.")

    env_dqn = make_env(is_slippery=stochastic, map_name="8x8", render_mode=None)
    dqn = DQN(input_states=env_dqn.observation_space.n, hidden_nodes=env_dqn.observation_space.n, out_states=env_dqn.action_space.n)
    policy3l = None 
    rewards_3l = None    
    print("Training DQN...")
    if not stochastic:
        # Deterministic case
        _, policy3l, rewards_3l = train(env_dqn, dqn, num_episodes=num_episodes, learning_rate=0.001, gamma=0.8, batch_size=batch_size, memory_size=memory_size)
    else:
        # Stochastic case
        _, policy3l, rewards_3l = train(env_dqn, dqn, num_episodes=num_episodes, learning_rate=0.01, gamma=0.9, batch_size=batch_size, memory_size=memory_size)

    print("Training completed for DQN")

    # Plot training results
    training_series = {
        "DQN": rewards_3l,
        "Q-learning": qlearning_rewards
    }
    
    plot_wins_vs_episodes(series_dict=training_series, num_episodes=num_episodes, title="Wins vs Episode (Train)", saveas=f"dqn_vs_qlearning_training_8x8_{"stochastic" if stochastic else "deterministic"}.png")

    print("Evaluation DQN...")
    env_dqn = make_env(is_slippery=stochastic, map_name="8x8", render_mode=None)
    running_series2l = run_trained_agent(policy3l, env_dqn, num_episodes=num_episodes)
    accuracy = np.sum(running_series2l)/num_episodes
    
    print(f"DQN: accuracy after training is {accuracy * 100:.2f}%")
    print("Evaluation completed for DQN")

    print("Evaluation of Q-Learning...")
    _, running_qlearning_series = run_agent(q_learning_env, qlearning_policy, num_episodes=num_episodes)
    accuracy = np.sum(running_qlearning_series)/num_episodes
    print(f"Q-Learning: accuracy after training is {accuracy * 100:.2f}%")
    print("Evaluation completed for Q-Learning")
   
    if not stochastic:
        # In deterministic case to avoid the overlap we generate 2 plots one for each curve
        running_series = { 
            "DQN ": running_series2l
        }
        
        plot_wins_vs_episodes(series_dict=running_series, num_episodes=num_episodes, title="Wins vs Episode (Evaluation)", saveas=f"dqn_vs_qlearning_evaluation_8x8_deterministic_dqn.png")
        running_series.clear()

        running_series = { 
            "Q-learning": running_qlearning_series
        }
        plot_wins_vs_episodes(series_dict=running_series, num_episodes=num_episodes, title="Wins vs Episode (Evaluation)", saveas=f"dqn_vs_qlearning_evaluation_8x8_deterministic_qlearning.png")
    else:
        running_series = { 
            "DQN ": running_series2l,
            "Q-learning": running_qlearning_series
        }
        plot_wins_vs_episodes(series_dict=running_series, num_episodes=num_episodes, title="Wins vs Episode (Evaluation)", saveas=f"dqn_vs_qlearning_evaluation_8x8_stochastic.png")

def main():
    selection = input(
        "Select option...\n"
        "Press 1 for 4x4 stochastic\n"
        "Press 2 for 8x8 stochastic\n"
        "Press 3 for 4x4 deterministic\n"
        "Press 4 for 8x8 deterministic\n"
        "Press 5 for epsilon decay plot\n"
    )

    if selection == "1":
        dqn_vs_qlearning_4x4(True)
    elif selection == "2":
        dqn_vs_qlearning_8x8(True)
    elif selection == "3":
        dqn_vs_qlearning_4x4(False)
    elif selection == "4":
        dqn_vs_qlearning_8x8(False)
    elif selection == "5":
        plot_epsilon_decay("Epsilon Decay", "epsilon_decay", 20000)
    else:
        print("Invalid option...")
        exit()

if __name__=="__main__":
    main()