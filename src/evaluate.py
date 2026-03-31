"""
Evaluate a saved DQN agent against a random agent.

Usage:
    python src/evaluate.py --model results/models/default.pth --episodes 1000
"""

import argparse
import numpy as np
import rlcard
from rlcard.agents import RandomAgent

from dqn_agent import DQNAgent


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",         type=str, required=True, help="Path to saved .pth model")
    p.add_argument("--episodes",      type=int, default=1000)
    p.add_argument("--hidden_layers", type=int, nargs="+", default=[64, 64])
    return p.parse_args()


def main():
    args = parse_args()

    env = rlcard.make("leduc-holdem")
    state_dim   = env.state_shape[0][0]
    num_actions = env.num_actions

    agent = DQNAgent(
        state_dim=state_dim,
        num_actions=num_actions,
        hidden_layers=args.hidden_layers,
        epsilon_start=0.0,   # greedy at eval time
        epsilon_end=0.0,
    )
    agent.load(args.model)

    random_agent = RandomAgent(num_actions=num_actions)
    env.set_agents([agent, random_agent])

    payoffs = []
    for _ in range(args.episodes):
        _, ep_payoffs = env.run(is_training=False)
        payoffs.append(ep_payoffs[0])

    payoffs = np.array(payoffs)
    win_rate = np.mean(payoffs > 0)
    avg_reward = np.mean(payoffs)

    print(f"Evaluation over {args.episodes} episodes")
    print(f"  Win rate        : {win_rate:.1%}")
    print(f"  Avg reward      : {avg_reward:+.4f} chips")
    print(f"  Std dev         : {np.std(payoffs):.4f}")

if __name__ == "__main__":
    main()
