from typing import List, Tuple, Dict
import jax
import jax.numpy as jnp
from ..abstractions import Environment

class GridCenterReward(Environment):
    def __init__(self, grid_size: int, center_reward: float = 1.0):
        """
        Initialize the grid environment.
        
        Args:
            grid_size (int): The width/height of the grid.
            center_reward (float): Reward given when an agent reaches the center.
        """
        self.grid_size = grid_size
        # Represent the center as a JAX array.
        self.center = jnp.array([grid_size // 2, grid_size // 2])
        self.center_reward = center_reward

    def reset(self) -> Dict:
        """
        Reset the environment. This environment is stateless,
        so reset simply returns grid info.
        """
        return {"grid_size": self.grid_size, "center": self.center}

    def compute_rewards(self, agent_states: List[Tuple[int, int]]) -> List[float]:
        """
        Award the center reward to any agent that reaches the center.
        
        Args:
            agent_states (List[Tuple[int, int]]): List of (x, y) positions for each agent.
            
        Returns:
            List[float]: A reward for each agent.
        """
        # Convert list of tuples to a JAX array of shape (n_agents, 2)
        states = jnp.array(agent_states)
        # Compare each state to the center in a vectorized way.
        is_center = jnp.all(states == self.center, axis=1)
        rewards = jnp.where(is_center, self.center_reward, 0.0)
        # Convert back to a Python list of floats.
        return list(rewards.tolist())

    def is_terminal(self, agent_states: List[Tuple[int, int]]) -> bool:
        """
        Determine if the episode should terminate.
        The episode terminates if any agent reaches the center.
        
        Args:
            agent_states (List[Tuple[int, int]]): List of (x, y) positions for each agent.
            
        Returns:
            bool: True if any agent is at the center.
        """
        states = jnp.array(agent_states)
        is_center = jnp.all(states == self.center, axis=1)
        return bool(jnp.any(is_center))
