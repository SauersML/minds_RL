import sys
from verifiers.trainer import Trainer


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python run_experiment.py <config_path>")
        raise SystemExit(1)
    config_path = sys.argv[1]
    trainer = Trainer.from_config(config_path)
    trainer.train(max_steps=100, output_dir="runs/ghost_trace_01")


if __name__ == "__main__":
    main()
