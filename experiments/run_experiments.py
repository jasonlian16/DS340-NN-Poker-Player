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

    # Define the Configurations
    arch_configs = {
        "small": ["--hidden_layers", "64"],
        "medium": ["--hidden_layers", "128", "128"],
        "large": ["--hidden_layers", "256", "128", "64"]
    }

    # Conditional Execution
    if args.experiment in ["arch", "all"]:
        for name, lyrs in arch_configs.items():
            run_experiment(f"arch_{name}", lyrs)


if __name__ == "__main__":
    main()