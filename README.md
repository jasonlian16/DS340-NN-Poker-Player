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

# Evaluate a saved model (default architecture: 64 64)
python3 src/evaluate.py --model results/models/baseline.pth

# Evaluate with a specific architecture (must match what the model was trained with)
python3 src/evaluate.py --model results/models/baseline.pth --hidden_layers 64 64 --episodes 5000

# Run an experiment sweep (epsilon, architecture, or gamma)
python3 experiments/run_experiments.py --experiment epsilon
```

### Evaluating experiment models

The `--hidden_layers` flag must match the architecture used during training. Use the values below:

**Gamma ablations** (all trained with `128 128`):
```bash
python src/evaluate.py --model results/models/gamma_0.95.pth --hidden_layers 128 128 --episodes 5000
python src/evaluate.py --model results/models/gamma_0.99.pth --hidden_layers 128 128 --episodes 5000
python src/evaluate.py --model results/models/gamma_0.8.pth  --hidden_layers 128 128 --episodes 5000
```

**Architecture ablations**:
```bash
python src/evaluate.py --model results/models/arch_small.pth  --hidden_layers 64         --episodes 5000
python src/evaluate.py --model results/models/arch_medium.pth --hidden_layers 128 128    --episodes 5000
python src/evaluate.py --model results/models/arch_large.pth  --hidden_layers 256 128 64 --episodes 5000
```

To find the hidden layer sizes for any saved model, check its training log:
```bash
python -c "import json; d=json.load(open('results/logs/<run_name>.json')); print(d['args']['hidden_layers'])"
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

Training works like this: the agent plays a hand, and at the end it finds out how many chips it won or lost. It then goes back and updates its Q-value estimates using the Bellman equation. Over thousands of hands, the Q-values get more accurate.

3 methods make this stable:

**Experience replay** — instead of learning from each hand immediately in sequence, the agent stores past hands in a buffer and samples random batches to learn from. This prevents the network from overfitting to whatever just happened.

**Target network** — there are actually two copies of the network: the main network (updated every step) and a target network (updated every few hundred steps). The target network is used to compute the training targets. Without this, the targets keep shifting every update and training becomes unstable.

**Epsilon-greedy exploration** — early in training, the agent acts randomly most of the time (high epsilon) so it explores the space of situations. Over time, epsilon decays toward 0 and the agent mostly follows its learned policy. This is the explore/exploit tradeoff.

### Reward signal

The reward is 0 at every step during a hand except the last, where it equals the chip gain or loss. This is realistic for poker — you don't know if you've won until the hand is over.

---

## Experiments

Four hyperparameters are tested:

**Epsilon decay schedule** — how quickly the agent shifts from random to learned behavior.
- Exponential: decays fast early, slows down later
- Linear: steady decay at a fixed rate
- Constant (ε=1.0): no decay, always fully random

**Network architecture** — how many hidden layers the Q-network has.
- Small: 1 hidden layer (64)
- Medium: 2 hidden layers (128, 128)
- Large: 3 hidden layers (256, 128, 64)

**Discount factor (gamma)** — how much the agent values future rewards vs. immediate ones.
- γ=0.80: short-sighted, cares mostly about the current hand
- γ=0.95: moderate
- γ=0.99: long-term, weighs future hands heavily

**Target network update frequency** — how often the frozen target network syncs with the main network.
- Every 200 steps (baseline)
- Every 100 steps (more frequent, tested in `target_100`)

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
