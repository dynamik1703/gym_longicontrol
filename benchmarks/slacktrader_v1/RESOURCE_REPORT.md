# Resource Report

The measured weighted planner solve time was 28.579 ms for AdaptivePrice and
27.098 ms for FixedPrice. AdaptivePrice used 445,123 exact physics predictions
across 355 solves and 3,037 native steps on four capability tasks.

Using the measured replan multiplier plus the prior corrected Hybrid SAC and
checkpoint measurements, a hypothetical authorized three-seed run with all
Diagnostics projects to 1.51 raw hours, or 3.02 hours with a 2x uncertainty
factor. Retained artifacts project to approximately 2.1 GB. Both are below the
24-hour and 12-GB gates.

Runtime and disk gates pass, but the scientific gate fails. No main-training
episodes or optimizer updates were run. Bounded correctness tests only exercised
short unreserved streams and exact-resume state; they are not performance data.
Per the stop rule, those primitives were not promoted to a canonical authorized
main-training execution, so `training_infrastructure_ready` remains false.
