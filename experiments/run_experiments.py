import subprocess
import argparse

def run_experiment(run_name, extra_args):
    cmd = ["python3", "src/train.py", "--run_name", run_name, "--episodes", "10000"] # update to 10000 episodes later
    cmd.extend(extra_args)
    print(f"\n STARTING: {run_name}")
    subprocess.run(cmd, check=True)

def main():
    # Setup Argument Parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=str, choices=["arch", "gamma", "epsilon", "all"], default="all")
    args = parser.parse_args()

    # Testing NN architecture
    arch_configs = {
        "small": ["--hidden_layers", "64"],
        "medium": ["--hidden_layers", "128", "128"],
        "large": ["--hidden_layers", "256", "128", "64"]
    }

    # Testing epsilon (linear, constant, exponential)
    epsilon_configs = [
        {"name": "eps_linear", "args": ["--epsilon_decay", "linear"]},
        {"name": "eps_exponential", "args": ["--epsilon_decay", "exponential"]},
        {"name": "eps_constant", "args": ["--epsilon_decay", "constant"]}
    ]

    # Testing Discount Factor (Gamma) 
    gamma_configs = ["0.8", "0.95", "0.99"]


    # Conditional Execution
    if args.experiment in ["gamma", "all"]:
        for g in gamma_configs:
            # We keep the architecture 'medium' to ensure a fair test
            run_id = f"gamma_{g}"
            run_experiment(run_id, ["--gamma", g, "--hidden_layers", "128", "128"]) 

    if args.experiment in ["arch", "all"]:
        for name, lyrs in arch_configs.items():
            run_experiment(f"arch_{name}", lyrs)
    
    if args.experiment in ["epsilon", "all"]:
        for config in epsilon_configs:
            run_experiment(f"exp_{config['name']}", config['args'])


if __name__ == "__main__":
    main()