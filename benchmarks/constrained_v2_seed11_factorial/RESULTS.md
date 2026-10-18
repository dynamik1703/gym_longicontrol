# Results

## Outcome

**EXECUTION_GATE_FAILED — no factorial policy was trained.**

Both exact frozen streams reproduce their required anchors, and all four initial
learner states are identical. However, canonical content fingerprints show that both
streams necessarily train on content identical to Development track 2006:

| Stream | Episode | Explicit seed | Canonical content SHA-256 |
|---|---:|---:|---|
| Historical reconstruction | 158 | n/a | `bb690f5dd203f2f950a6243c8752175f0aaebc6749ec3a33d2fa80370ea2c839` |
| Current archived stream | 163 | 1418256283 | `bb690f5dd203f2f950a6243c8752175f0aaebc6749ec3a33d2fa80370ea2c839` |
| Current archived stream | 242 | 274583042 | `bb690f5dd203f2f950a6243c8752175f0aaebc6749ec3a33d2fa80370ea2c839` |

Development 2006 has audit fingerprint
`b8b74200c641873dec2c07755d151d4de57d5177602c4b6a6d4836637b4a80a0`.
The audit hash omits the common task object; the study canonical hash above includes
it. Positions are `[0.0, 800.0]` metres and speed limits are
`[16.666666666666668, 19.444444444444443]` m/s.

The current stream uses no reserved numeric seed. Its first 223 reset seeds match the
archived current seed-11 checkpoint exactly, including episode 163. This is a content
collision in a finite procedural generator, not a reset-ID leak. With a 1,800-step
episode cap, 300,000 transitions require at least 167 episodes, so the historical-158
and current-163 collisions cannot be avoided by policy behavior.

## Reconstruction and initialization

- `T_HIST`: audit `historical_track_fingerprints`, using initial seed 11, continuous
  environment RNG, and used reset draws 0, 2, 4, ... . The first 100 definitions
  exactly match the audit artifact.
- `T_CURRENT`: audit/current `TrainingTrackStream`, using
  `SeedSequence([11, 0x4D564531])` and explicit reset seeds. The first 100 definitions
  exactly match the audit, and the retained 223-seed archive prefix matches exactly.
- Archived current checkpoint SHA-256:
  `994568f614c9d2428b805fafafc9526717fa2d6e4ca73783adfcb361feb4090d`.
- A/B/C/D actor SHA-256:
  `72e9207137ecd2f8e1c6506915f1348eb59f47133a3b181cbf09098888e928c5`.
- A/B/C/D combined learner SHA-256:
  `beffa5734886fc714e4eeca5ddae17fa51e12af9839a2999af7743dc5287d87a`.
- Initial alpha is 1.0, target entropy is -1.0, and both PID multipliers are 0.0.

## Unavailable causal readouts

Actual transitions and updates are zero for A/B/C/D. There are no Attempts, resumes,
Development curves, Validation outcomes, PID/lambda trajectories, alpha trajectories,
factorial contrasts, current-side D reproduction result, or historical-side A
movement result. The exact number of Validation episodes is **0**, not the planned 36.

The appropriate qualitative interpretation is **inconclusive (reproducibility/split
gate failed)**. Confidence in the collision finding is high under the controlled
current runtime. No claim is made about population effects or about the unknown exact
historical runtime.

## Next scientific step

Do not start another study automatically. The protocol owner must resolve the direct
conflict between exact-stream fidelity and content-level split protection. A revised
protocol could explicitly define split protection by numeric seed identity, or could
authorize deterministic rejection of protected content and acknowledge that this
changes both streams. Either choice changes the present frozen protocol.

Validation remained sealed. Paper tracks 4000–4017 remained sealed and were never
opened or evaluated.
