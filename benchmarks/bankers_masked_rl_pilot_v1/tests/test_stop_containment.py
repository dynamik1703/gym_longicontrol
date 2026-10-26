"""Containment checks only; do not rerun the failed scientific gate."""

import json
from hashlib import sha256
from pathlib import Path

import pytest

from benchmarks.bankers_masked_rl_pilot_v1.execution import PilotStopped, main
from benchmarks.bankers_masked_rl_pilot_v1.evaluation import evaluate


def test_execution_and_evaluation_fail_before_reset():
    for entry in (main, evaluate):
        with pytest.raises(PilotStopped, match='INTEGRITY_STOP'):
            entry()


def test_failed_sources_preserved():
    study = Path(__file__).resolve().parents[1]
    expected = json.loads((study / 'tested_source_manifest.json').read_text())
    assert all(sha256((study / p).read_bytes()).hexdigest() == h
               for p, h in expected.items())
    report = json.loads((study / 'preflight_results.json').read_text())
    assert report['failed'] == 3 and not report['all_gates_passed']
    assert sha256((study / 'preflight_tests.xml').read_bytes()).hexdigest() == report['test_xml_sha256']
