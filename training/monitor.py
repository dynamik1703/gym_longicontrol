"""Plot the JSON history emitted by the supported trainer."""

import argparse
import json
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("history", type=Path)
    parser.add_argument(
        "--output", type=Path, help="Save a plot instead of opening a window"
    )
    args = parser.parse_args(argv)
    import matplotlib.pyplot as plt

    history = json.loads(args.history.read_text(encoding="utf-8"))
    figure, axes = plt.subplots(2, 1, sharex=True)
    steps = history["training_steps"]
    axes[0].plot(steps, history["eval_return"])
    axes[0].set(ylabel="Mean evaluation return")
    for name, values in history["losses"].items():
        axes[1].plot(steps, values, label=name)
    axes[1].set(xlabel="Training steps", ylabel="Loss")
    axes[1].legend()
    figure.tight_layout()
    if args.output:
        figure.savefig(args.output)
    else:
        plt.show()
    plt.close(figure)


if __name__ == "__main__":
    main()
