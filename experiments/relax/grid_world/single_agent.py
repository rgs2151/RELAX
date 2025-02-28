from typing import Tuple
import jax
import jax.numpy as jnp

from ..abstractions import Agent


class SingleAgent(Agent):
    def __init__(self, grid_size, alpha=0.1, gamma=0.9, epsilon=0.1, seed=0):
        """
        Initialize the single-agent. Asumes a grid world environment.
        Args:
            grid_size (int): Size of the grid (assumes a square grid).
            alpha (float): Learning rate.
            gamma (float): Discount factor.
            epsilon (float): Exploration rate.
            seed (int): Random seed for reproducibility.
        """
        self.grid_size = grid_size
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.num_actions = 5  # Actions: 0:stay, 1:up, 2:down, 3:left, 4:right

        # Create a Q-table with shape: (grid_size, grid_size, num_actions)
        self.q_table = jnp.zeros((grid_size, grid_size, self.num_actions))
        
        # Initialize PRNG key for JAX-based randomness.
        self.key = jax.random.PRNGKey(seed)
        
        # Agent's initial position (starting at the center).
        self.x = grid_size // 2
        self.y = grid_size // 2

    def reset(self, randomize: bool = True):
        """
        Reset the agent's position.
        Args:
            randomize (bool): If True, set a random starting position; otherwise, use the center.
        """
        if randomize:
            self.key, subkey = jax.random.split(self.key)
            self.x = int(jax.random.randint(subkey, (), 0, self.grid_size))
            self.key, subkey = jax.random.split(self.key)
            self.y = int(jax.random.randint(subkey, (), 0, self.grid_size))
        else:
            self.x = self.y = self.grid_size // 2

    def get_state(self):
        """
        Return the agent's current state as a tuple (x, y).
        """
        return (self.x, self.y)

    @jax.jit
    def _move(self, pos, action, grid_size):
        """
        Compute the new position based on the current position and action.
        Args:
            pos (tuple): Current (x, y) position.
            action (int): Action index.
            grid_size (int): Size of the grid.
        Returns:
            A tuple representing the new position (x, y).
        """
        x, y = pos
        
        def stay():
            return (x, y)
        def up():
            return (x, jnp.maximum(y - 1, 0))
        def down():
            return (x, jnp.minimum(y + 1, grid_size - 1))
        def left():
            return (jnp.maximum(x - 1, 0), y)
        def right():
            return (jnp.minimum(x + 1, grid_size - 1), y)
        
        # jax.lax.switch dispatches based on the action index.
        new_pos = jax.lax.switch(action, [stay, up, down, left, right])
        return new_pos

    def choose_action(self, state):
        """
        Choose an action using an epsilon-greedy policy.
        Args:
            state (tuple): The current state (x, y).
        Returns:
            The selected action as an integer.
        """
        self.key, subkey = jax.random.split(self.key)
        rand_val = jax.random.uniform(subkey)
        if rand_val < self.epsilon:
            # Explore: choose a random action.
            self.key, subkey = jax.random.split(self.key)
            action = int(jax.random.randint(subkey, (), 0, self.num_actions))
            return action
        else:
            # Exploit: choose the action with the highest Q-value.
            x, y = state
            q_vals = self.q_table[x, y, :]
            action = int(jnp.argmax(q_vals))
            return action

    def step(self, action):
        """
        Execute an action and update the agent's position.
        Args:
            action (int): The action to perform.
        Returns:
            The new state (x, y) after the move.
        """
        current_state = self.get_state()
        new_state = self._move(current_state, action, self.grid_size)
        # Update the internal position.
        self.x, self.y = int(new_state[0]), int(new_state[1])
        return new_state

    def update(self, state, action, reward, next_state):
        """
        Update the Q-table using the Q-learning update rule.
        Args:
            state (tuple): The state before the action.
            action (int): The action taken.
            reward (float): The reward received.
            next_state (tuple): The state after the action.
        """
        x, y = state
        nx, ny = next_state
        
        # Current Q-value.
        current_q = self.q_table[x, y, action]
        # Maximum Q-value in the next state.
        max_next_q = jnp.max(self.q_table[nx, ny, :])
        
        # Q-learning update.
        new_q = (1 - self.alpha) * current_q + self.alpha * (reward + self.gamma * max_next_q)
        
        # Update the Q-table immutably.
        self.q_table = self.q_table.at[x, y, action].set(new_q)