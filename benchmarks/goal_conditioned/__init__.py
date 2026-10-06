"""Goal-conditioned LongiControl benchmark preparation."""

from .goal import (
    GoalScales,
    canonical_desired_goal,
    encode_achieved_goal,
    goal_success,
    goal_transition_reward,
    relabel_desired_goal,
    relabeled_terminal_mask,
)

__all__ = [
    "GoalScales",
    "canonical_desired_goal",
    "encode_achieved_goal",
    "goal_success",
    "goal_transition_reward",
    "relabel_desired_goal",
    "relabeled_terminal_mask",
]
