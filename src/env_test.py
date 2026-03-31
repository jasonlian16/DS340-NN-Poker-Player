import rlcard
from rlcard.agents import RandomAgent

env = rlcard.make("leduc-holdem")

print("Leduc Hold'em Environment")
print(f"Number of players : {env.num_players}")
print(f"Number of actions : {env.num_actions}")
print(f"State shape       : {env.state_shape}")
print(f"Action shape      : {env.action_shape}")

# Play one random hand and inspect observation
random_agents = [RandomAgent(num_actions=env.num_actions) for _ in range(env.num_players)]
env.set_agents(random_agents)

trajectories, payoffs = env.run(is_training=False)

print("\n Sample trajectory (player 0, first transition)")
first = trajectories[0][0]
print(f"obs vector (length {len(first['obs'])}):\n  {first['obs']}")
print(f"legal actions : {first['legal_actions']}")
print(f"raw obs keys  : {list(first.keys())}")

print(f"\n Payoffs")
for i, p in enumerate(payoffs):
    print(f"  Player {i}: {p:+.1f} chips")
