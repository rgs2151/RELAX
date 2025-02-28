from typing import List
import jax
import jax.numpy as jnp
from .abstractions import Environment

class Trainer:
    def __init__(self, agents: List, environment: Environment, num_episodes: int = 1000, max_steps: int = 100):
        """
        Initialize the trainer.
        
        Args:
            agents (List): A list of agent instances.
            environment (Environment): An instance of the environment.
            num_episodes (int): Number of episodes to train.
            max_steps (int): Maximum steps per episode.
        """
        self.agents = agents
        self.env = environment
        self.num_episodes = num_episodes
        self.max_steps = max_steps

    def _episode_step(self, state: jnp.ndarray, unused):
        """
        A single time step for all agents using JAX vectorized operations.
        
        Args:
            state: jnp.ndarray of shape (num_agents, 2) representing current agent positions.
            unused: placeholder for scan (not used).
        
        Returns:
            next_state: Updated state after one step.
            info: Tuple containing (prev_state, actions, rewards, done).
        """
        num_agents = state.shape[0]
        
        # --- Action Selection ---
        # For each agent, call its choose_action function.
        # Here we loop over agents (since the number of agents is small),
        # but you can later try to re-write choose_action in a fully vectorized way.
        actions = []
        for i in range(num_agents):
            # Convert the i-th agent's state (a JAX array) to a tuple for compatibility.
            s = tuple(state[i].tolist())
            action = self.agents[i].choose_action(s)
            actions.append(action)
        actions = jnp.array(actions)  # shape (num_agents,)

        # --- Environment Step ---
        # Each agent takes a step with its chosen action.
        next_states = []
        for i in range(num_agents):
            # We assume agent.step returns a new state as a tuple.
            ns = self.agents[i].step(int(actions[i]))
            next_states.append(jnp.array(ns))
        next_state = jnp.stack(next_states)  # shape (num_agents, 2)

        # --- Compute Rewards & Terminal Condition ---
        # Convert next_states to a list of tuples for the environment.
        state_list = [tuple(s.tolist()) for s in next_state]
        rewards = jnp.array(self.env.compute_rewards(state_list))  # shape (num_agents,)
        done = self.env.is_terminal(state_list)

        return next_state, (state, actions, rewards, done)

    def train_jax(self):
        """
        Run a single episode using JAX's vectorized loop (lax.scan).
        
        Returns:
            final_states: The states of all agents after the episode.
            scan_info: A tuple containing per-step info (states, actions, rewards, done flags).
        """
        # Gather initial states from all agents as a JAX array of shape (num_agents, 2).
        init_states = jnp.stack([jnp.array(self.agents[i].get_state()) for i in range(len(self.agents))])
        
        # Run the episode loop for max_steps using lax.scan.
        final_states, scan_info = jax.lax.scan(self._episode_step, init_states, None, length=self.max_steps)
        return final_states, scan_info

    def train(self):
        """
        Run the training loop over multiple episodes.
        For each episode, the environment is reset and the vectorized (JAX) inner loop is run.
        """
        for episode in range(self.num_episodes):
            # Reset the environment and all agents.
            _ = self.env.reset()
            for agent in self.agents:
                agent.reset(randomize=True)
            
            # Run the episode in a vectorized manner.
            final_states, scan_info = self.train_jax()
            # (scan_info contains a tuple of (states, actions, rewards, done) for each step)
            
            # (Here you could aggregate rewards, update logging, etc.)
            print(f"Episode {episode+1} finished.")
