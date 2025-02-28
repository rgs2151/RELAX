import jax
import jax.numpy as jnp
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict

# Abstract base class for agents.
class Agent(ABC):
    @abstractmethod
    def __init__(self, *args, **kwargs):
        """
        Initialize the agent with necessary parameters, state variables,
        and internal data structures (e.g., Q-tables or policy networks).
        """
        raise NotImplementedError

    @abstractmethod
    def reset(self, randomize: bool = True):
        """
        Reset the agent's internal state for a new episode.
        Args:
            randomize (bool): If True, randomize the initial state.
        """
        raise NotImplementedError

    @abstractmethod
    def get_state(self):
        """
        Return the current state of the agent.
        """
        raise NotImplementedError

    @abstractmethod
    def choose_action(self, state):
        """
        Choose an action based on the current state using an exploration/exploitation policy.
        Args:
            state: The current state.
        Returns:
            The chosen action.
        """
        raise NotImplementedError

    @abstractmethod
    def step(self, action):
        """
        Execute the given action, updating the agent's state.
        Args:
            action: The action to perform.
        Returns:
            The new state after the action.
        """
        raise NotImplementedError

    @abstractmethod
    def update(self, state, action, reward, next_state):
        """
        Update the agent's internal parameters based on the observed transition.
        Args:
            state: The state before taking the action.
            action: The action taken.
            reward: The reward received.
            next_state: The resulting state.
        """
        raise NotImplementedError

class Environment(ABC):
    @abstractmethod
    def reset(self) -> Dict:
        """
        Reset any internal state of the environment and return environment info if needed.
        For example, it might return grid parameters.
        """
        raise NotImplementedError

    @abstractmethod
    def compute_rewards(self, agent_states: List[Tuple[int, int]]) -> List[float]:
        """
        Given the list of agent states, compute and return a reward for each agent.
        
        Args:
            agent_states (List[Tuple[int, int]]): List of (x, y) positions for all agents.
        
        Returns:
            List[float]: A list of rewards, one per agent.
        """
        raise NotImplementedError

    @abstractmethod
    def is_terminal(self, agent_states: List[Tuple[int, int]]) -> bool:
        """
        Determine if the environment is in a terminal state based on the agent states.
        
        Args:
            agent_states (List[Tuple[int, int]]): List of agent positions.
        
        Returns:
            bool: True if the simulation should terminate.
        """
        raise NotImplementedError
