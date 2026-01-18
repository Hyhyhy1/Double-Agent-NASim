import numpy as np
import gymnasium as gym
from gymnasium import spaces
from .state import State
from .network import Network
from .action import (
    StartExploitAction,
    StopExploitAction,
    StructuringActionSpace,
    ExploitingActionSpace,
    Action
)
from .observation import (
    get_structuring_observation,
    get_exploiting_observation
)



class NASimDAAEnv:

    def __init__(self, scenario, fully_obs=True, flat_obs=True, max_exploit_steps=10):

        self.network = Network(scenario)
        self.scenario = scenario
        self.fully_obs = fully_obs
        self.flat_obs = flat_obs
        self.max_exploit_steps = max_exploit_steps 


        self.state = None
        self.in_exploit_phase = False
        self.current_target_host = None
        self.exploit_step_count = 0
        self.total_exploit_reward = 0.0


        self.structuring_action_space = StructuringActionSpace(scenario)
        self.exploiting_action_space = None

        self.structuring_observation_space = spaces.Box(
            low= -np.inf, high= np.inf, shape= self.get_structuring_obs_shape(scenario, self.flat_obs)
        )

        self.exploiting_observation_space = spaces.Box(
            low= -np.inf, high= np.inf, shape= self.get_exploiting_obs_shape(scenario, self.flat_obs)
        )

        self.structuring_action_space_size = self.get_structuring_action_space_size(scenario)
        self.exploiting_action_space_size = self.get_exploiting_action_space_size(scenario)


        self.global_step_count = 0
        self.max_global_steps = getattr(scenario, 'step_limit', 1000)

    def reset(self):
        """Сброс среды. Возвращает наблюдение для Агента 1."""
        self.state = State.generate_initial_state(self.network)
        self.in_exploit_phase = False
        self.current_target_host = None
        self.exploit_step_count = 0
        self.total_exploit_reward = 0.0
        self.global_step_count = 0
        self.exploiting_action_space = None

        obs = get_structuring_observation(self.state)
        return obs.flatten() if self.flat_obs else obs

    def step_agent1(self, action):
        """
        Вызывается только когда НЕ в фазе эксплуатации.
        Возвращает:
          - Если scan: (structuring_obs, reward - action_cost, done, info)
          - Если choose_host: (exploit_obs, reward=0, done, info)
        """
        assert not self.in_exploit_phase, "Нельзя вызывать step_agent1 во время фазы эксплуатации"

        if not isinstance(action, Action):
            action = self.structuring_action_space.get_action(action)

        self.global_step_count += 1

        if not isinstance(action, StartExploitAction):
            next_state, action_result = self.network.perform_action(self.state, action)

            self.state = next_state

            reward = action_result.value - action.cost

            done = self.goal_reached() or self.global_step_count >= self.max_global_steps

            obs = get_structuring_observation(self.state)

            if self.flat_obs:
                obs = obs.flatten()
            return obs, reward, done, {"phase": "structuring"}

        target = action.target

        # Вход в фазу эксплуатации
        self.in_exploit_phase = True
        self.current_target_host = target
        self.exploit_step_count = 0
        self.total_exploit_reward = 0.0
        self.exploiting_action_space = ExploitingActionSpace(self.scenario, target)

        exploit_obs = get_exploiting_observation(self.state, target)
        if self.flat_obs:
            exploit_obs = exploit_obs.flatten()

        done = self.global_step_count >= self.max_global_steps
        return exploit_obs, 0.0, done, {"phase": "exploit"}

    def step_agent2(self, action):
        """
        Вызывается ТОЛЬКО в фазе эксплуатации.
        Может возвращать:
          - (exploit_obs, 0, done, info) — если агент продолжает атаку
          - (structuring_obs, total_reward, done, info) — если агент завершает атаку
        """
        assert self.in_exploit_phase, "step_agent2 можно вызывать только во время эксплуатации"

        if not isinstance(action, Action):
            action = self.exploiting_action_space.get_action(action)

        self.global_step_count += 1
        self.exploit_step_count += 1

        #Выход из фазы эксплуатации
        if isinstance(action, StopExploitAction) or self.exploit_step_count >= self.max_exploit_steps:
            return self._end_exploit_phase()


        next_state, action_result = self.network.perform_action(self.state, action)
        step_reward = action_result.value - action.cost
        self.total_exploit_reward += step_reward


        if self.goal_reached() or self.global_step_count >= self.max_global_steps:
            return self._end_exploit_phase()

        self.state = next_state
        exploit_obs = get_exploiting_observation(self.state, self.current_target_host)
        if self.flat_obs:
            exploit_obs = exploit_obs.flatten()

        return exploit_obs, step_reward, False, {"phase": "exploit"}

    def _end_exploit_phase(self):
        """Завершает фазу эксплуатации и возвращает контроль агенту 1."""
        total_reward = self.total_exploit_reward
        self.in_exploit_phase = False
        self.current_target_host = None
        self.exploit_step_count = 0
        self.total_exploit_reward = 0.0
        #self.exploiting_action_space = None

        struct_obs = get_structuring_observation(self.state)
        if self.flat_obs:
            struct_obs = struct_obs.flatten()

        done = self.global_step_count >= self.max_global_steps
        return struct_obs, total_reward, done, { #Возможно потом заменю на нулевую награду
            "phase": "structuring",
            "total_exploit_reward": total_reward
        }
    
    def goal_reached(self, state=None):
        """Check if the state is the goal state.

        The goal state is when all sensitive hosts have been compromised.

        Parameters
        ----------
        state : State, optional
            a state, if None will use current_state of environment
            (default=None)

        Returns
        -------
        bool
            True if state is goal state, otherwise False.
        """
        if state is None:
            state = self.state
        return self.network.all_sensitive_hosts_compromised(state)
    
    @staticmethod
    def get_structuring_obs_shape(scenario, flat=True):
        num_hosts = len(scenario.hosts)
        S = len(scenario.subnets)
        H_max = max(scenario.subnets)
        per_host = S + H_max + 6
        if flat:
            return (num_hosts * per_host,)
        else:
            return (num_hosts, per_host)

    @staticmethod
    def get_exploiting_obs_shape(scenario, flat=True):
        size = (
            4 +  # compromised, reachable, discovered, access
            scenario.num_os +
            scenario.num_services +
            scenario.num_processes
        )
        return (size,) if flat else (1, size)

    @staticmethod
    def get_structuring_action_space_size(scenario):
        return 5 * len(scenario.hosts)  # 4 scans + 1 choose_host per host

    @staticmethod
    def get_exploiting_action_space_size(scenario):
        return (
            3 +  # service, os, process scan
            len(scenario.exploits) +
            len(scenario.privescs) +
            1   # stop
        )