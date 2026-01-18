import nasim
import numpy as np
import matplotlib.pyplot as plt
from agents.qlearning_agent2 import TabularQLearningAgent
from agents.modified_ddqn_agent import DoubleQAgent
from statistics import fmean, stdev
from nasim.scenarios import make_benchmark_scenario
from nasim.envs.multiagent_env import NASimDAAEnv

SCENARIO_NAME = "tiny"
scenario = make_benchmark_scenario(SCENARIO_NAME, 42)


def evaluate_agents(agent1, agent2, n_episodes):
    env = NASimDAAEnv(scenario, flat_obs=True, max_exploit_steps=20)

    scores = []
    trajectory_steps =[]


    for episode in range(n_episodes):
        obs = env.reset()
        done = False
        trajectory_score = 0
        trajectory_step = 0

        while not done:

            trajectory_step+=1

            if not env.in_exploit_phase:
                # --- Structuring Agent (Agent 1) ---
                action = agent1.get_egreedy_action(obs)
                next_obs, reward, done, info = env.step_agent1(action)
                trajectory_score += reward
                
                obs = next_obs

            else:
                # --- Exploiting Agent (Agent 2) ---
                total_exploit_reward = 0.0

                while env.in_exploit_phase and not done:
                    action = agent2.get_egreedy_action(obs)
                    next_obs, reward, done, info = env.step_agent2(action)

                    total_exploit_reward += reward
                    obs = next_obs

                    if info["phase"] == "structuring":
                        break
    
                trajectory_score += total_exploit_reward
    
    
        scores.append(trajectory_score)
        trajectory_steps.append(trajectory_step)

    return scores, trajectory_steps




def make_plot(training_scores, is_eval=False):
    if not is_eval:
        plt.plot(training_scores, label="награда за траекторию")
        plt.title("Динамика наград за траекторию в процессе обучения")
        plt.ylabel("Награда")
        plt.xlabel("Эпизод обучения")
        plt.grid(True)
        plt.legend()
        plt.show()
    
    else:
        plt.plot(training_scores, label="награда за траекторию")
        plt.title("Награды за траекторию в процессе тестирования")
        plt.ylabel("Награда")
        plt.xlabel("Эпизод тестирования")
        plt.grid(True)
        plt.legend()
        plt.show()


if __name__ == '__main__':
    env = NASimDAAEnv(scenario, flat_obs=True, max_exploit_steps=150)

    terminal_obs = np.zeros(env.exploiting_observation_space.shape[0], dtype=np.float32)

    #agent1 = DoubleQAgent(observation_space_shape=env.structuring_observation_space.shape[0], 
    #            action_space_n=env.structuring_action_space_size)

    #agent2 = DoubleQAgent(observation_space_shape=env.structuring_observation_space.shape[0], 
    #            action_space_n=env.structuring_action_space_size)

    agent1 = TabularQLearningAgent(observation_space_shape=env.structuring_observation_space.shape, 
                action_space_n=env.structuring_action_space_size)
        
    agent2 = TabularQLearningAgent(observation_space_shape=env.exploiting_observation_space.shape, 
                action_space_n=env.exploiting_action_space_size)


    # Буфер для отложенной награды
    choose_host_step = None  # (obs, action)

    num_episodes = 1200
    scores = []

    for episode in range(num_episodes):
        obs = env.reset()
        done = False
        trajectory_score = 0

        while not done:
            if not env.in_exploit_phase:
                # --- Structuring Agent (Agent 1) ---
                action = agent1.choose_action(obs)
                next_obs, reward, done, info = env.step_agent1(action)
                trajectory_score += reward

                if info["phase"] == "exploit":
                    # Запоминаем шаг, чтобы позже обновить награду
                    choose_host_step = (obs, action)
                    obs = next_obs
                else:
                    agent1.learn(obs, action, reward, next_obs, done)
                    obs = next_obs

            else:
                # --- Exploiting Agent (Agent 2) ---
                total_exploit_reward = 0.0

                while env.in_exploit_phase and not done:
                    action = agent2.choose_action(obs)
                    next_obs, reward, done, info = env.step_agent2(action)

                    total_exploit_reward += reward

                    if info["phase"] == "structuring":
                        agent2.learn(obs, action, reward, terminal_obs, True)

                    else:
                        agent2.learn(obs, action, reward, next_obs, done)

                    obs = next_obs

                    if info["phase"] == "structuring":
                        break
                    

                # --- Обновляем награду для Structuring Agent ---
                if choose_host_step is not None:
                    obs1, action1 = choose_host_step

                    trajectory_score += total_exploit_reward
                    agent1.learn(obs1, action1, total_exploit_reward, obs, done)
                    choose_host_step = None
        
        scores.append(trajectory_score)
        
        if (episode + 1) % 10 == 0:
            last_10 = scores[-10:]
            avg_last_10 = fmean(last_10)
            std_last_10 = stdev(last_10) if len(last_10) > 1 else 0.0
            print(f"Episode {episode + 1}: "
                  f"avg of last 10 = {avg_last_10:.2f} ± {std_last_10:.2f}")


    evaluating_scores, trajectory_steps = evaluate_agents(agent1, agent2, 200)

    make_plot(evaluating_scores, is_eval=True)
    input()
    print("Среднее значение награды и среднее число шагов")
    print(fmean(evaluating_scores), fmean(trajectory_steps))
    print("стандартное отклонение награды и числа шагов")
    print(stdev(evaluating_scores), stdev(trajectory_steps))
