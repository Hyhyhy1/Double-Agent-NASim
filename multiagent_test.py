import nasim
import matplotlib.pyplot as plt
from agents.qlearning_agent2 import TabularQLearningAgent
from statistics import fmean, stdev

SCENARIO_NAME = "small"

def evaluate_agent(explorer_agent, attacker_agent, n_episodes):
    env = nasim.make_multiagent_benchmark(SCENARIO_NAME, render_mode="human")

    scores = []
    trajectory_steps =[]

    for i in range(n_episodes):
        terminated = False
        truncated = False
        explorer_obs, _ = env.reset()
        score = 0
        steps = 0

        while not (terminated or truncated):

            if env.current_agent == "Explorer":
                action = explorer_agent.get_egreedy_action(explorer_obs)
                print(f"Explorer action = {env.explorer_action_space.get_action(action).name}, action target = {env.explorer_action_space.get_action(action).target}")
                
                if env.explorer_action_space.get_action(action).name != "attack_host":
                    new_explorer_obs, reward, terminated, truncated, info = env.step(action)
                    explorer_obs = new_explorer_obs
                    score+=reward

                else:
                    attacker_obs, _, terminated, truncated, info = env.step(action)

            elif env.current_agent == "Attacker":
                action = attacker_agent.get_egreedy_action(attacker_obs)
                print(f"Attacker action = {env.attacker_action_space.get_action(action).name}, action target = {env.attacker_action_space.get_action(action).target}")
                
                if env.attacker_action_space.get_action(action).name != "stop_attack":
                    new_attacker_obs, reward, terminated, truncated, info = env.step(action)
                    attacker_obs = new_attacker_obs
                
                else:
                    new_explorer_obs, reward, terminated, truncated, info = env.step(action)
                    explorer_obs = new_explorer_obs
                    score+=reward
            
            steps+=1
    
        scores.append(score)
        trajectory_steps.append(steps)

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



if __name__ == "__main__":
     
    env = nasim.make_multiagent_benchmark(SCENARIO_NAME, render_mode="human")

    explorer_obs, info = env.reset()
    full_reward = 0

    done = False
    step_limit = False
    #env.render()


    explorer_agent = TabularQLearningAgent(observation_space_shape=env.explorer_observation_space.shape, 
                    action_space_n=env.explorer_action_space.n)
    
    attacker_agent = TabularQLearningAgent(observation_space_shape=env.attacker_observation_space.shape, 
                    action_space_n=env.attacker_action_space.n)

    _temp_action = None
    attacker_obs = None

    while not (done or step_limit):

        if env.current_agent == "Explorer":
            action = explorer_agent.get_egreedy_action(explorer_obs)
            print(f"Explorer action = {env.explorer_action_space.get_action(action).name}, action target = {env.explorer_action_space.get_action(action).target}")
            
            if env.explorer_action_space.get_action(action).name != "attack_host":
                _temp_action = action
                new_explorer_obs, reward, done, step_limit, info = env.step(action)
                explorer_agent.learn(explorer_obs, action, reward, explorer_obs, done)
                explorer_obs = new_explorer_obs
                full_reward += reward

            else:
                attacker_obs, _, done, step_limit, info = env.step(action)

        elif env.current_agent == "Attacker":
            action = attacker_agent.get_egreedy_action(attacker_obs)
            print(f"Attacker action = {env.attacker_action_space.get_action(action).name}, action target = {env.attacker_action_space.get_action(action).target}")
            
            if env.attacker_action_space.get_action(action).name != "stop_attack":
                new_attacker_obs, reward, done, step_limit, info = env.step(action)
                attacker_agent.learn(attacker_obs, action, reward, new_attacker_obs, done)
                attacker_obs = new_attacker_obs
            
            else:
                new_explorer_obs, reward, done, step_limit, info = env.step(action)
                explorer_agent.learn(explorer_obs, _temp_action, reward, new_explorer_obs, done)
                explorer_obs = new_explorer_obs
                full_reward += reward

        #env.render()


    print(f"Total reward: {full_reward}")

    evaluating_scores, trajectory_steps = evaluate_agent(explorer_agent, attacker_agent, 200)

    make_plot(evaluating_scores, is_eval=True)

    print("Среднее значение награды и среднее число шагов")
    print(fmean(evaluating_scores), fmean(trajectory_steps))
    print("стандартное отклонение награды и числа шагов")
    print(stdev(evaluating_scores), stdev(trajectory_steps))