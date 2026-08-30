"""orbit: an rl cluster scheduler.

a c++ discrete-event dag simulator exposed to a pytorch/gnn + reinforce
training stack.
"""

from __future__ import annotations

__version__ = "0.1.0"

# the compiled c++ extension, imported eagerly so the fast simulator types are
# available on import orbit
from . import _orbit_core as core

__all__ = ["__version__", "core"]
