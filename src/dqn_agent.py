"""
DQN Agent for Leduc Hold'em.
"""

import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque


class QNetwork(nn.Module):
    def __init__(self, state_dim: int, num_actions: int, hidden_layers: list[int]):
        super().__init__()
        layers = []
        in_dim = state_dim
        for h in hidden_layers:
            layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        layers.append(nn.Linear(in_dim, num_actions))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states,      dtype=np.float32),
            np.array(actions,     dtype=np.int64),
            np.array(rewards,     dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones,       dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    use_raw = False  # RLCard requires this — use integer actions, not strings

    def __init__(
        self,
        state_dim: int,
        num_actions: int,
        hidden_layers: list[int] = [64, 64],
        lr: float = 1e-3,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: str = "exponential",   # "exponential" | "linear" | "constant"
        epsilon_decay_steps: int = 10_000,
        buffer_capacity: int = 10_000,
        batch_size: int = 64,
        target_update_freq: int = 200,
    ):
        self.num_actions = num_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.epsilon_decay_steps = epsilon_decay_steps
        # Added check for epsilon start being greater than 0 to avoid dividing by 0 error
        if epsilon_start > 0:
            self.epsilon_exp_rate = (epsilon_end / epsilon_start) ** (1.0 / epsilon_decay_steps)
        else:
            self.epsilon_exp_rate = 0.0

        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")   
                 
        self.q_net      = QNetwork(state_dim, num_actions, hidden_layers).to(self.device)
        self.target_net = QNetwork(state_dim, num_actions, hidden_layers).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer(buffer_capacity)

        self.step_count = 0

    def step(self, state: dict) -> int:
        """RLCard interface — called by env when it's the agent's turn."""
        obs = state["obs"]
        legal_actions = list(state["legal_actions"].keys())
        return self.select_action(obs, legal_actions)
    
    def eval_step(self, state: dict) -> tuple[int, dict]:
        """
        RLCard interface for evaluation. 
        Returns the best action and an empty info dict.
        """
        obs = state["obs"]
        legal_actions = list(state["legal_actions"].keys())
        
        # We call select_action. Since epsilon is 0.0 during evaluation, this will automatically be a greedy move.
        action = self.select_action(obs, legal_actions)
        
        return action, {}


    def select_action(self, state: np.ndarray, legal_actions: list[int]) -> int:
        """Epsilon-greedy: random action with prob epsilon, else best legal Q-value."""
        if random.random() < self.epsilon:
            return random.choice(legal_actions)

        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_net(state_t).squeeze(0)

        # Set illegal actions to -inf so argmax never picks them
        mask = torch.full((self.num_actions,), float("-inf"))
        for a in legal_actions:
            mask[a] = q_values[a]
        return int(mask.argmax().item())

    def learn(self):
        """One gradient update using a random mini-batch from the replay buffer."""
        if len(self.replay_buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        states_t      = torch.FloatTensor(states).to(self.device)
        actions_t     = torch.LongTensor(actions).to(self.device)
        rewards_t     = torch.FloatTensor(rewards).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t       = torch.FloatTensor(dones).to(self.device)

        # Q-value the network predicted for the action actually taken
        q_values = self.q_net(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)

        # Bellman target: reward + discounted best future Q (0 if terminal)
        with torch.no_grad():
            next_q = self.target_net(next_states_t).max(1).values
            targets = rewards_t + self.gamma * next_q * (1 - dones_t)

        loss = nn.functional.mse_loss(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self._decay_epsilon()

        # Periodically copy main network weights into the target network
        self.step_count += 1
        if self.step_count % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        return loss.item()

    def store(self, state, action, reward, next_state, done):
        self.replay_buffer.push(state, action, reward, next_state, done)

    def _decay_epsilon(self):
        if self.epsilon_decay == "exponential":
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_exp_rate)
        elif self.epsilon_decay == "linear":
            step_size = (1.0 - self.epsilon_end) / self.epsilon_decay_steps
            self.epsilon = max(self.epsilon_end, self.epsilon - step_size)
        # "constant" — do nothing

    def save(self, path: str):
        torch.save(self.q_net.state_dict(), path)

    def load(self, path: str):
        self.q_net.load_state_dict(torch.load(path, map_location=self.device))
        self.target_net.load_state_dict(self.q_net.state_dict())
