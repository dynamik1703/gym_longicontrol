"""No Diagnostic execution after the failed independent-evidence gate."""

from .execution import require_ready


def evaluate(*args, **kwargs):
    require_ready()
