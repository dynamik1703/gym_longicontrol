# Inherited learner contract

Byte-identical sac_agent.py, masked_distribution.py and replay.py. Causal 10D observation; 21 native actions; two hidden layers of 256; twin reward critics and three cost critics; original optimizer settings, signed physical energy reward (-1000 × step kWh), speed/terminal/intervention costs, fixed coefficients, native gamma 0.99**0.1, tau .005, entropy target .98 log(valid count).

RAW samples all actions and executes directly. POST samples all actions, keeps proposal in replay and replaces inadmissible proposals by the closest certified grid index, lower index on ties. PRE restricts actor, entropy and critic target support before sampling; no normal replacement. POST/PRE both independently certify the selected native action with the same frozen V4 shield. No warm start, reward changes, progress bonus, cloning, tuning or chunking.
