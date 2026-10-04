# Agent-designed reward study: stopped at trainer gate

## Decision

The automated reward-design experiment was **not started**. Its mandatory first
gate requires a stable, credible SB3 trainer. The preceding controlled study
found that neither candidate satisfies the already-frozen scalar-baseline
criteria:

| trainer | final validation RSR by seed | mean RSR | completion | speed compliance | accepted |
| --- | --- | ---: | ---: | ---: | --- |
| SB3 SAC | 4/9, 0/9, 1/9 | 5/27 (18.5%) | 5/27 | 27/27 | no |
| SB3 PPO | 0/9, 0/9, 0/9 | 0/27 | 0/27 | 27/27 | no |

SAC's seed range is 4/9, above the allowed 1/9. PPO remains at zero RSR at every
50k checkpoint and all final policies stand still. Speed compliance is therefore
not evidence of competence. Full evidence is in
[`../scalar_sb3/RESULTS.md`](../scalar_sb3/RESULTS.md).

## Why stopping is necessary

Without a stable trainer, differences between Human V2-B, Agent One-Shot, and
Agent Iterative rewards would be inseparable from seed and training instability.
Allowing an automated reward agent to iterate against that noise would also
encourage development-set overfitting and would not answer whether an agent can
translate operational requirements more efficiently than a human.

This is the preregistered stop condition, not a negative result about automated
reward design. No claim about agent reward quality can be made because no agent
reward was proposed or trained.

## Actions deliberately not taken

- no reward-design prompt was submitted to a model;
- no one-shot or iterative reward was proposed;
- no development screening run was launched;
- no validation result was supplied as reward-design feedback;
- no Agent One-Shot or Agent Iterative validation policy was trained;
- no exploratory tracks 1000–1008 were used;
- no final-paper tracks 4000–4017 were generated, evaluated, plotted, or
  inspected;
- no constrained RL, MORL, HER, CRL, or requirement-conditioned method was
  added.

Accordingly, there is no `REWARD_DESIGN_PROMPT.md` or
`reward-development.json`: creating either would falsely imply that the reward
agent experiment had begun.

## Required prerequisite before resuming

Freeze a scalar trainer that meets the existing all-seed acceptance rule on the
development/validation protocol without using final-paper seeds. The SB3 study
suggests a bounded SAC-specific investigation of long-horizon discounting and
terminal credit assignment; it does not support automatic 1M training or PPO
tuning. Only after the trainer is frozen should the five-proposal budget,
one-shot prompt, development-only feedback loop, and nine final policies be
preregistered and run.
