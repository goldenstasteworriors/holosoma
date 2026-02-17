"""SONIC tracker manager.

This is intentionally implemented as a subclass of the existing
`WholeBodyTrackingManager` to keep behavior identical at baseline while allowing
SONIC-specific changes to be introduced incrementally.
"""

from __future__ import annotations

from holosoma.envs.wbt.wbt_manager import WholeBodyTrackingManager


class SonicTrackingManager(WholeBodyTrackingManager):
    pass
