# Learner/reward invariant specification

sac_agent.py, masked_distribution.py and replay.py are copied byte-for-byte from
immutable Pilot V1. Shield treatment predicates and certification are unchanged;
the declared nearest/lower-tie admission rule receives the correctness repair below. The physical shield libraries are imported
unchanged. No performance-driven amendment is made.

The categorical actor, twin reward critics and three cost critics have 256/256
ReLU hidden layers. Input is the same original eight-vector plus causal position
/1000 and elapsed time/140. Hidden track, numeric seed, oracle and held-out outcomes
are excluded. Native gamma=0.99**0.1; tau=.005; batch=256; replay capacity=300000.
Actor/critic/temperature Adam LR=3e-4; multiplier LR=1e-3. Costs: speed, terminal
failure, intervention. Speed/terminal multipliers are projected nonnegative with
zero optimization limits; intervention actor weight remains .05.

RAW and POST categorical support is all 21 actions [-1,-.9,...,.9,1]. PRE support
is the frozen safe grid. Invalid probabilities and expectation contributions are
exactly zero. Entropy target is .98 log(number of valid actions), including zero
for singleton support. Reward target is r+D sum pi(a'|x')[min(Qbar1,Qbar2)-alpha log pi].
Each cost target is c+D sum pi Qbar_cost. Actor and multiplier equations are unchanged
from V1's exact categorical implementation. Stored verified successor masks are
used for PRE targets; no mask computation occurs in optimizer updates.

Native reward=-1000*signed step_energy_kWh. Speed cost is exact positive right-endpoint
speed excess in m/s. Terminal cost=1 only at true task end for incomplete route or
time>140s. Intervention=abs(native proposal-native execution). No progress reward,
reward tuning or hidden-route dense deadline signal is added. Energy never repairs
task failure. Difficult terminal credit assignment is a research limitation.
Environment TimeLimit ends the task at 1800 steps, with no bootstrap. A pure 50k
training-budget cutoff keeps normal continuation and is not a terminal task failure.

POST stores the actor proposal and models transitions caused by its deterministic
nearest certified execution (lower value for a tie). PRE samples the mask directly
and final-certifies without ordinary replacement. Both use identical V4 predicates,
backup policy, frontier, timestep, margins and grid. Scalar penalties are no safety proof.

Frozen exploratory decision rules: PROMISING requires PRE speed=12/12, at least one
completion, median progress>=500m and RSR strictly above both matched learners.
NO_LEARNING_SIGNAL if all learned arms complete zero tasks, or PRE completes zero
with median progress<100m. Otherwise INCONCLUSIVE. These thresholds are fixed before
any new policy-performance outcome. A 12-task, one-seed pilot proves no superiority.

Pre-training correctness amendment GRID_TIE_INDEX_EXACT_V2: the old floating-point
selector maps proposal .2 between certified .1/.3 to .3 because the computed
distances are .1 and .09999999999999998. V2 uses integer grid-index distance,
then the smaller index, to implement the already-declared mathematical tie rule.
All 19 interior adjacent-neighbor ties are tested. This changes no action value,
safety predicate, learner, reward, multiplier, hyperparameter or budget. It is
documented before resource measurement, scientific freeze and all policy outcomes.
Frozen physical contextual baselines remain untouched.
