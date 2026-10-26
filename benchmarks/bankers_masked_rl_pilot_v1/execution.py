"""Fail-closed entry point for the permanently stopped pilot Attempt set."""

import json
from pathlib import Path


class PilotStopped(RuntimeError):
    pass


def require_ready():
    report = json.loads(Path(__file__).with_name('preflight_results.json').read_text())
    if not report['all_gates_passed']:
        raise PilotStopped(report['classification'] + ': ' + report['failed_gate'])
    # This stopped package has no reviewed scientific execution transaction.
    raise PilotStopped('Scientific execution is not implemented or authorized in this stopped package')


def main():
    require_ready()


if __name__ == '__main__':
    main()
