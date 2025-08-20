import nasim
import matplotlib.pyplot as plt
from agents.heuristic_agent import Heuristic
from statistics import fmean, stdev

SCENARIO_NAME = "small"

def evaluate_agent(agent, n_episodes):
    env = nasim.make_benchmark(SCENARIO_NAME, fully_obs=True, flat_actions=True, render_mode="human")

    scores = []
    trajectory_steps =[]

    for i in range(n_episodes):
        terminated = False
        truncated = False
        state, _ = env.reset()
        score = 0
        steps = 0
        while not (terminated or truncated):

                action = agent.choose_action(state)
                new_state, reward, terminated, truncated, _ = env.step(action)

                state = new_state

                #env.render()
                steps += 1
                score += reward
    
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
     
    env = nasim.make_benchmark(SCENARIO_NAME, fully_obs=True, flat_actions=True, render_mode="human")

    obs, info = env.reset()
    full_reward = 0

    done = False
    step_limit = False
    env.render()


    agent = Heuristic(observation_space_shape=env.observation_space.shape, 
                    action_space=env.action_space, action_space_n=env.action_space.n, 
                    subnet_count= 4, host_count= 8, max_machines_in_subnet= 5)

    while not (done or step_limit):

        action = agent.choose_action(obs)
        print(f"action = {env.action_space.get_action(action).name}, target = {env.action_space.get_action(action).target}")
        new_obs, reward, done, step_limit, info = env.step(action)
        env.render()
        obs = new_obs
        full_reward += reward



    print(f"Total reward: {full_reward}")

    evaluating_scores, trajectory_steps = evaluate_agent(agent, 200)

    make_plot(evaluating_scores, is_eval=True)

    print("Среднее значение награды и среднее число шагов")
    print(fmean(evaluating_scores), fmean(trajectory_steps))
    print("стандартное отклонение награды и числа шагов")
    print(stdev(evaluating_scores), stdev(trajectory_steps))