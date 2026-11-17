"""The plugin's lab geometry kinds, as ``mime.scene`` holds MIME's: rigid tips, each with the registered shape its one
region is evaluated by (one file a kind)."""

from .cone_tip import ConeTip
from .headed_tip import HeadedTip
from .punch_tip import PunchTip

__all__ = ["ConeTip", "HeadedTip", "PunchTip"]
