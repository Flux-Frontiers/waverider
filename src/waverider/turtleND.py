"""
N-dimensional turtle. The implementation lives in the ``turtlend`` package,
https://github.com/Flux-Frontiers/turtlend. This module re-exports it so that
``waverider.turtleND`` keeps resolving.

Author: Eric G. Suchanek, PhD
"""

from turtlend.turtleND import TurtleND

__all__ = ["TurtleND"]
