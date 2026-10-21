"""Materialize V2 split and pair registration without performance evaluation."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from .constants import PAIR_LEDGER, SPLIT_LEDGER
from .integrity import controller_identities, load_frozen_policies
from .ledger import atomic_json, build_pair_ledger
from .split import materialize, render_split_ledger

HERE = Path(__file__).resolve().parent


def _atomic_text(path: Path, text: str) -> None:
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temporary.exists():
            temporary.unlink()


def prepare_artifacts() -> None:
    if SPLIT_LEDGER.exists() or PAIR_LEDGER.exists():
        raise FileExistsError("V2 frozen ledgers already exist; refusing replacement")
    split = materialize(SPLIT_LEDGER)
    policy_metadata, _ = load_frozen_policies()
    pair_ledger = build_pair_ledger(
        split,
        controller_identities=controller_identities(policy_metadata),
    )
    atomic_json(PAIR_LEDGER, pair_ledger)
    _atomic_text(HERE / "SPLIT_LEDGER.md", render_split_ledger(split))


if __name__ == "__main__":
    prepare_artifacts()
