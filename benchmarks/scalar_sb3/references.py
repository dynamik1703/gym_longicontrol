"""Generate validation references using the shared controller definitions."""

from __future__ import annotations

import argparse
from pathlib import Path

from benchmarks.scalar_sac.references import (
    evaluate_references,
    save_reference_results,
)

from .config import DEFAULT_CONFIG_PATH, load_configuration


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    results = evaluate_references(load_configuration(args.config))
    print(save_reference_results(args.output, results))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised as a script
    raise SystemExit(main())
