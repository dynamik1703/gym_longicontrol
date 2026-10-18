# Frozen protocol and execution gate

The scientific configuration is inherited without modification from Constrained V2
configuration SHA-256
`89d7142cc84ba60d98075313220b331b60c2c41997b6c65b11adeb1b08712060`.
The factorial configuration is [canonical.json](canonical.json). The protocol and
configuration hashes are recorded in `freeze.json`.

## Common configuration

- FSRL SACLag, gamma 0.99, n-step 2.
- 300,000 real transitions and exactly 30,000 gradient updates per condition.
- Replay capacity 100,000; batch size 256.
- Actor and three double critics: 128 × 128.
- Actor LR 0.0005; critic LR 0.001.
- Automatic alpha, initial alpha 1.0, target entropy -1, alpha LR 0.0003.
- Tau 0.05.
- PID `(0.05, 0.0005, 0.1)`, cost limits zero.
- Native timestep 0.1 s and maximum episode length 1,800 transitions.
- Negative signed step energy divided by 0.25, frozen speed cost, and frozen dense
  deadline-deficit cost.
- Development 2000–2008 at six 50k targets; Validation 3000–3008 once after four
  policies freeze; paper tracks 4000–4017 sealed.

No MVE, synthetic replay, learned/physics augmentation, reward shaping, speed margin,
progress reward, changed PID/alpha, or changed n-step is permitted.

## Initialization preflight

All four independently reconstructed current-runtime learner states are identical.
Their combined learner-state SHA-256 is
`beffa5734886fc714e4eeca5ddae17fa51e12af9839a2999af7743dc5287d87a`.
Their actor SHA-256 is the audit anchor
`72e9207137ecd2f8e1c6506915f1348eb59f47133a3b181cbf09098888e928c5`.
Individual critic, target, optimizer, alpha, and PID hashes are in `results.json`.

## Mandatory pre-training gates

Main execution is authorized only if every gate passes:

1. both streams reproduce their audit/archive anchors;
2. both collector schedules are implemented and tested;
3. all initial learner hashes match;
4. D matches current archived semantics;
5. resource estimates pass;
6. no Development, Validation, or paper track content occurs in training.

Gates 1 and 3 pass. Gate 6 fails. The exact stream requirements and content-level
split protection cannot both be satisfied:

- historical episode 158 equals Development 2006;
- current episode 163 equals Development 2006;
- both are necessarily reached because a 300k run needs at least 167 episodes.

The protocol therefore enters terminal state `EXECUTION_GATE_FAILED`. No training,
runtime pilot, Development evaluation, Validation opening, or paper evaluation is
authorized. Filtering, skipping, reseeding, or replacing a colliding track would be
an unrequested third intervention and is prohibited.

## Recovery status

No Attempt was created and no checkpoint exists. Exact resume and
`FRESH_RESTART_REQUIRED` are not applicable. Fresh restart remains unauthorized.
