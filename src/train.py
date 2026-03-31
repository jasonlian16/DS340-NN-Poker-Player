"""
Training loop: DQN agent (player 0) vs. RandomAgent (player 1).
See README for a full explanation of the training process.

Usage:
    python src/train.py
    python src/train.py --episodes 20000 --gamma 0.95 --epsilon_decay linear
"""

import argparse
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import rlcard
from rlcard.agents import RandomAgent

from dqn_agent import DQNAgent


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes",            type=int,   default=10_000)
    p.add_argument("--hidden_layers",       type=int,   nargs="+", default=[64, 64])
    p.add_argument("--lr",                  type=float, default=1e-3)
    p.add_argument("--gamma",               type=float, default=0.99)
    p.add_argument("--epsilon_decay",       type=str,   default="exponential",
                   choices=["exponential", "linear", "constant"])
    p.add_argument("--epsilon_decay_steps", type=int,   default=8_000)
    p.add_argument("--batch_size",          type=int,   default=64)
    p.add_argument("--target_update_freq",  type=int,   default=200)
    p.add_argument("--buffer_capacity",     type=int,   default=10_000)
    p.add_argument("--log_interval",        type=int,   default=500)
    p.add_argument("--run_name",            type=str,   default="default")
    return p.parse_args()


def run_episode(env, agent, random_agent, is_training: bool):
    """
    Play one full hand. Returns the chip payoff for the DQN agent (player 0).
    We step manually so we can record every decision and store it in the replay buffer.
    Reward is 0 at every step except the last, where it's the chip gain/loss.
    """
    agents = [agent, random_agent]
    state, player_id = env.reset()
    player0_transitions = []

    while not env.is_over():
        current_agent = agents[player_id]
        action = current_agent.step(state)

        if is_training and player_id == 0:
            player0_transitions.append((state["obs"], action))

        next_state, next_player_id = env.step(action)
        state, player_id = next_state, next_player_id

    payoffs = env.get_payoffs()

    if is_training:
        for i, (obs, action) in enumerate(player0_transitions):
            done = (i == len(player0_transitions) - 1)
            reward = float(payoffs[0]) if done else 0.0
            next_obs = player0_transitions[i + 1][0] if not done else obs
            agent.store(obs, action, reward, next_obs, done)
            agent.learn()

    return payoffs[0]


def main():
    args = parse_args()

    env = rlcard.make("leduc-holdem")
    state_dim   = env.state_shape[0][0]
    num_actions = env.num_actions

    dqn_agent = DQNAgent(
        state_dim=state_dim,
        num_actions=num_actions,
        hidden_layers=args.hidden_layers,
        lr=args.lr,
        gamma=args.gamma,
        epsilon_decay=args.epsilon_decay,
        epsilon_decay_steps=args.epsilon_decay_steps,
        batch_size=args.batch_size,
        target_update_freq=args.target_update_freq,
        buffer_capacity=args.buffer_capacity,
    )
    random_agent = RandomAgent(num_actions=num_actions)

    reward_log = []
    running_rewards = []

    print(f"Training for {args.episodes} episodes  |  state_dim={state_dim}  |  run='{args.run_name}'")
    print(f"  hidden_layers={args.hidden_layers}  gamma={args.gamma}  "
          f"epsilon_decay={args.epsilon_decay}  lr={args.lr}")

    for ep in range(1, args.episodes + 1):
        payoff = run_episode(env, dqn_agent, random_agent, is_training=True)
        reward_log.append(payoff)

        if ep % args.log_interval == 0:
            avg = np.mean(reward_log[-args.log_interval:])
            running_rewards.append(avg)
            print(f"  ep {ep:>6d} | avg reward (last {args.log_interval}): {avg:+.4f} "
                  f"| epsilon: {dqn_agent.epsilon:.4f}")

    os.makedirs("results/logs",   exist_ok=True)
    os.makedirs("results/plots",  exist_ok=True)
    os.makedirs("results/models", exist_ok=True)

    log_path = f"results/logs/{args.run_name}.json"
    with open(log_path, "w") as f:
        json.dump({"reward_log": reward_log, "args": vars(args)}, f, indent=2)
    print(f"\nReward log saved to {log_path}")

    model_path = f"results/models/{args.run_name}.pth"
    dqn_agent.save(model_path)
    print(f"Model saved to {model_path}")

    # Plot learning curve. The dashed line at y=0 is the random agent baseline.
    x = [i * args.log_interval for i in range(1, len(running_rewards) + 1)]
    plt.figure(figsize=(8, 4))
    plt.plot(x, running_rewards, marker="o", markersize=3)
    plt.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    plt.xlabel("Episode")
    plt.ylabel(f"Avg reward (per {args.log_interval} eps)")
    plt.title(f"Learning Curve — {args.run_name}")
    plt.tight_layout()
    plot_path = f"results/plots/{args.run_name}_learning_curve.png"
    plt.savefig(plot_path, dpi=150)
    print(f"Plot saved to {plot_path}")
    plt.show()


if __name__ == "__main__":
    main()
