# Candidate reward API

Each candidate is one standalone UTF-8 Python file defining exactly one public
function:

```python
def compute_reward(ctx: RewardContext) -> RewardOutput: ...
```

The harness injects `RewardContext` and `RewardOutput`. A candidate may instead
import those two names from `benchmarks.llm_reward.reward_api`. `import math` is
the only other permitted import.

Return:

```python
RewardOutput(
    reward=<finite scalar>,
    components={<lower_snake_case_name>: <finite scalar>, ...},
)
```

The component dictionary is diagnostic only. Its values never affect physical
evaluation or candidate ranking directly. Component names must start with a
lowercase letter, contain only lowercase letters, digits and underscores, and
be at most 64 characters.

The function must be deterministic and free of side effects. It must not read
files, network state, process state or environment variables; call subprocesses;
use randomness; mutate global state; inspect track identities; or import other
project modules. The returned reward and every component must remain finite on
ordinary and terminal smoke contexts.

Store the short explicit design rationale separately from the source. Do not
include hidden reasoning.

