from __future__ import annotations

"""
Minimal DQN implementation from scratch using NumPy.

This module provides:
- Network: 2x64 fully connected ReLU network mapping state -> Q-values
- ReplayBuffer: experience replay storage
- DQNAgent: epsilon-greedy DQN with target network and SGD updates

Intended to be used by rl/train_dqn.py when --impl scratch is selected.
"""

from dataclasses import dataclass
from typing import Tuple

import numpy as np


class Network:
    """Two-hidden-layer fully-connected network with ReLU activations."""

    def __init__(self, state_size: int, action_size: int, hidden_size: int = 64):
        rng = np.random.default_rng()
        self.w1 = rng.normal(0, 0.1, size=(hidden_size, state_size))
        self.b1 = np.zeros((hidden_size,))
        self.w2 = rng.normal(0, 0.1, size=(hidden_size, hidden_size))
        self.b2 = np.zeros((hidden_size,))
        self.w3 = rng.normal(0, 0.1, size=(action_size, hidden_size))
        self.b3 = np.zeros((action_size,))

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass.

        Args:
            x: state batch, shape (batch_size, state_size)
        Returns:
            Q-values, shape (batch_size, action_size)
        """
        z1 = x @ self.w1.T + self.b1  # (B, H)
        a1 = np.maximum(0.0, z1)
        z2 = a1 @ self.w2.T + self.b2  # (B, H)
        a2 = np.maximum(0.0, z2)
        q = a2 @ self.w3.T + self.b3  # (B, A)
        return q

    def copy_from(self, other: "Network") -> None:
        """Copy parameters from another network."""
        self.w1 = other.w1.copy()
        self.b1 = other.b1.copy()
        self.w2 = other.w2.copy()
        self.b2 = other.b2.copy()
        self.w3 = other.w3.copy()
        self.b3 = other.b3.copy()

    def parameters(self) -> Tuple[np.ndarray, ...]:
        return self.w1, self.b1, self.w2, self.b2, self.w3, self.b3


class ReplayBuffer:
    """Simple cyclic replay buffer."""

    def __init__(self, capacity: int, state_size: int):
        self.capacity = capacity
        self.state_buf = np.zeros((capacity, state_size), dtype=np.float32)
        self.next_state_buf = np.zeros((capacity, state_size), dtype=np.float32)
        self.action_buf = np.zeros((capacity,), dtype=np.int64)
        self.reward_buf = np.zeros((capacity,), dtype=np.float32)
        self.done_buf = np.zeros((capacity,), dtype=np.float32)
        self.idx = 0
        self.size = 0

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self.state_buf[self.idx] = state
        self.next_state_buf[self.idx] = next_state
        self.action_buf[self.idx] = action
        self.reward_buf[self.idx] = reward
        self.done_buf[self.idx] = float(done)

        self.idx = (self.idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return (
            self.state_buf[idxs],
            self.action_buf[idxs],
            self.reward_buf[idxs],
            self.next_state_buf[idxs],
            self.done_buf[idxs],
        )

    def __len__(self) -> int:
        return self.size


@dataclass
class DQNConfig:
    state_size: int
    action_size: int
    gamma: float = 0.99
    lr: float = 5e-4
    batch_size: int = 32
    buffer_size: int = 10_000
    learning_starts: int = 1_000
    train_freq: int = 4
    target_update_interval: int = 500
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_fraction: float = 0.2  # fraction of total timesteps over which to decay


class DQNAgent:
    def __init__(self, config: DQNConfig, total_timesteps: int):
        self.cfg = config
        self.policy_net = Network(config.state_size, config.action_size)
        self.target_net = Network(config.state_size, config.action_size)
        self.target_net.copy_from(self.policy_net)
        self.replay_buffer = ReplayBuffer(config.buffer_size, config.state_size)
        self.total_timesteps = max(1, total_timesteps)
        self.timestep = 0

    def epsilon(self) -> float:
        """Linear epsilon decay from epsilon_start to epsilon_end."""
        decay_steps = int(self.cfg.epsilon_decay_fraction * self.total_timesteps)
        decay_steps = max(decay_steps, 1)
        if self.timestep >= decay_steps:
            return self.cfg.epsilon_end
        frac = self.timestep / decay_steps
        return self.cfg.epsilon_start + frac * (self.cfg.epsilon_end - self.cfg.epsilon_start)

    def act(self, state: np.ndarray) -> int:
        self.timestep += 1
        if np.random.rand() < self.epsilon():
            return np.random.randint(self.cfg.action_size)
        state_batch = state.reshape(1, -1).astype(np.float32)
        q_values = self.policy_net.forward(state_batch)[0]
        return int(np.argmax(q_values))

    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self.replay_buffer.push(state, action, reward, next_state, done)

    def maybe_update_target(self) -> None:
        if self.timestep % self.cfg.target_update_interval == 0:
            self.target_net.copy_from(self.policy_net)

    def train_step(self) -> float | None:
        """Perform one gradient step. Returns loss or None if not updated."""
        if (
            self.timestep < self.cfg.learning_starts
            or self.timestep % self.cfg.train_freq != 0
            or len(self.replay_buffer) < self.cfg.batch_size
        ):
            return None

        (
            states,
            actions,
            rewards,
            next_states,
            dones,
        ) = self.replay_buffer.sample(self.cfg.batch_size)

        # Forward pass
        q_values = self.policy_net.forward(states)  # (B, A)
        batch_indices = np.arange(self.cfg.batch_size)
        q_sa = q_values[batch_indices, actions]  # (B,)

        with np.errstate(over="ignore"):
            next_q_values = self.target_net.forward(next_states)
            max_next_q = np.max(next_q_values, axis=1)  # (B,)

        targets = rewards + self.cfg.gamma * max_next_q * (1.0 - dones)

        # Compute loss and gradient w.r.t q_values
        td_error = q_sa - targets  # (B,)
        loss = float(np.mean(td_error**2))

        grad_q_sa = (2.0 / self.cfg.batch_size) * td_error  # (B,)

        # Backprop through final layer
        # Recompute intermediate activations for current states
        z1 = states @ self.policy_net.w1.T + self.policy_net.b1  # (B, H)
        a1 = np.maximum(0.0, z1)
        z2 = a1 @ self.policy_net.w2.T + self.policy_net.b2  # (B, H)
        a2 = np.maximum(0.0, z2)

        # Gradients for final layer weights/biases: only for chosen actions
        grad_w3 = np.zeros_like(self.policy_net.w3)
        grad_b3 = np.zeros_like(self.policy_net.b3)
        for i in range(self.cfg.batch_size):
            a = actions[i]
            grad_w3[a] += grad_q_sa[i] * a2[i]
            grad_b3[a] += grad_q_sa[i]

        # Backprop into a2
        grad_a2 = np.zeros_like(a2)
        for i in range(self.cfg.batch_size):
            grad_a2[i] = grad_q_sa[i] * self.policy_net.w3[actions[i]]

        # ReLU derivative at z2
        grad_z2 = grad_a2 * (z2 > 0.0)

        grad_w2 = grad_z2.T @ a1  # (H, H)
        grad_b2 = np.sum(grad_z2, axis=0)

        # Backprop into a1
        grad_a1 = grad_z2 @ self.policy_net.w2  # (B, H)
        grad_z1 = grad_a1 * (z1 > 0.0)

        grad_w1 = grad_z1.T @ states  # (H, S)
        grad_b1 = np.sum(grad_z1, axis=0)

        # SGD update
        lr = self.cfg.lr
        self.policy_net.w3 -= lr * grad_w3
        self.policy_net.b3 -= lr * grad_b3
        self.policy_net.w2 -= lr * grad_w2
        self.policy_net.b2 -= lr * grad_b2
        self.policy_net.w1 -= lr * grad_w1
        self.policy_net.b1 -= lr * grad_b1

        return loss

    def save(self, path: str) -> None:
        """Save network parameters to a NumPy .npz file."""
        np.savez(
            path,
            w1=self.policy_net.w1,
            b1=self.policy_net.b1,
            w2=self.policy_net.w2,
            b2=self.policy_net.b2,
            w3=self.policy_net.w3,
            b3=self.policy_net.b3,
        )

