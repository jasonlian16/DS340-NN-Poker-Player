# DS340-NN-Poker-Player

Deep Q-Network (DQN) agent for Leduc Hold'em poker.
Team: Gillian Lee, Jason Lian — DS340 project.

## Setup

```bash
pip3 install -r requirements.txt
python3 src/env_test.py   # verify the environment works
```

## How to run

```bash
# Train with default settings
python3 src/train.py --run_name baseline

# Evaluate a saved model
python3 src/evaluate.py --model results/models/baseline.pth

# Run an experiment sweep (epsilon, architecture, or gamma)
python3 experiments/run_experiments.py --experiment epsilon
```

Results (logs, plots, model weights) are saved to `results/`.

---

## How it works

### The game

Leduc Hold'em is a simplified 2-player poker variant with a 6-card deck (Jack, Queen, King in two suits). Each player gets one private card, there's one community card revealed mid-game, and there are two betting rounds. The agent learns to play as player 0 against a random opponent.

### What the agent sees (the state)

RLCard gives the agent a 36-dimensional vector encoding its private card, the community card, and the betting history so far. This vector is the input to the neural network.

### How DQN works

The agent uses a Deep Q-Network to decide what action to take. The idea is to train a neural network that, given the current game state, outputs a "Q-value" for each possible action (fold, call, raise). A Q-value is an estimate of how many chips the agent expects to win if it takes that action. At each decision point, the agent just picks the action with the highest Q-value.

Training works like this: the agent plays a hand, and at the end it finds out how many chips it won or lost. It then goes back and updates its Q-value estimates using the Bellman equation — basically, "the value of a state-action pair should equal the immediate reward plus the discounted value of the best next state." Over thousands of hands, the Q-values get more accurate.

Three tricks make this stable:

**Experience replay** — instead of learning from each hand immediately in sequence, the agent stores past hands in a buffer and samples random batches to learn from. This prevents the network from overfitting to whatever just happened.

**Target network** — there are actually two copies of the network: the main network (updated every step) and a target network (updated every few hundred steps). The target network is used to compute the training targets. Without this, the targets keep shifting every update and training becomes unstable.

**Epsilon-greedy exploration** — early in training, the agent acts randomly most of the time (high epsilon) so it explores the space of situations. Over time, epsilon decays toward 0 and the agent mostly follows its learned policy. This is the explore/exploit tradeoff.

### Reward signal

The reward is 0 at every step during a hand except the last, where it equals the chip gain or loss. This is realistic for poker — you don't know if you've won until the hand is over.

---

## Experiments

Three hyperparameters are tested per the project proposal:

**Epsilon decay schedule** — how quickly the agent shifts from random to learned behavior.
- Exponential: decays fast early, slows down later
- Linear: steady decay at a fixed rate
- Constant (ε=0.1): no decay, always 10% random

**Network architecture** — how many hidden layers the Q-network has.
- Small: 1 hidden layer (64 units)
- Medium: 2 hidden layers (64, 64)
- Large: 3 hidden layers (128, 128, 64)

**Discount factor (gamma)** — how much the agent values future rewards vs. immediate ones.
- γ=0.80: short-sighted, cares mostly about the current hand
- γ=0.95: moderate
- γ=0.99: long-term, weighs future hands heavily

---

## Repo structure

```
src/
  env_test.py          smoke test — prints the observation space and runs one random hand
  dqn_agent.py         QNetwork, ReplayBuffer, DQNAgent
  train.py             training loop
  evaluate.py          win rate and avg reward against a random agent
experiments/
  run_experiments.py   sweeps epsilon / architecture / gamma configs
results/
  logs/                reward logs per run (JSON)
  plots/               learning curve PNGs
  models/              saved network weights (.pth)
```
