"""Compatibility entry point; prefer ``python -m training`` from the checkout."""

import sys
from pathlib import Path


def main():
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from training.cli import main as training_main

    return training_main()


if __name__ == "__main__":
    raise SystemExit(main())
