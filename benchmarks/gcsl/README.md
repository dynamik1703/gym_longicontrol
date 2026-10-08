# Goal-Conditioned Supervised Learning

This directory is the frozen, execution-ready LongiControl GCSL V1 study. It
directly supervises the recorded action at a source state under an actually
achieved strict-future projected goal. It neither consumes task reward nor
learns a critic.

Read `SOURCE_AUDIT.md` for the pinned primary sources, `DESIGN.md` for the exact
likelihood and adaptations, `PROTOCOL.md` for split/budget/gating rules, and
`RESOURCE_REPORT.md` for bounded preparation measurements.

Safe preparation-only command:

```bash
python -m benchmarks.gcsl.runner preflight
```

The following commands are implemented but refuse to operate while
`main_training_authorized` and `main_training_enabled` are false:

```bash
python -m benchmarks.gcsl.runner run
python -m benchmarks.gcsl.runner resume
python -m benchmarks.gcsl.runner validate
```

There is intentionally no paper-final command.
