"""SONIC task environments.

Currently this task is a thin wrapper around Whole Body Tracking (WBT) so we can
iterate in an isolated namespace without impacting existing WBT experiments.
"""

from .sonic_manager import SonicTrackingManager

__all__ = ["SonicTrackingManager"]
