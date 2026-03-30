"""Data models for CargoLoader."""

from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class Box:
    """A box to be packed into a container."""
    id: int
    name: str
    width: float       # x-axis dimension
    depth: float       # z-axis dimension
    height: float      # y-axis dimension (vertical)
    weight: float      # kg
    can_rotate: bool = True
    color: Tuple[float, float, float] = field(default=(0.5, 0.5, 0.5))

    def get_rotations(self) -> List[Tuple[float, float, float]]:
        """Return all unique (width, depth, height) rotation permutations.

        6 possible orientations when can_rotate is True, otherwise just the
        original orientation.  Duplicate dimension tuples are removed.
        """
        if not self.can_rotate:
            return [(self.width, self.depth, self.height)]

        w, d, h = self.width, self.depth, self.height
        perms = [
            (w, d, h), (w, h, d),
            (d, w, h), (d, h, w),
            (h, w, d), (h, d, w),
        ]
        seen: List[Tuple[float, float, float]] = []
        for p in perms:
            if p not in seen:
                seen.append(p)
        return seen


@dataclass
class PlacedBox:
    """A box that has been placed inside the container."""
    box: Box
    x: float           # position along width axis
    y: float           # position along height axis (vertical)
    z: float           # position along depth axis
    width: float       # placed width  (after rotation)
    depth: float       # placed depth  (after rotation)
    height: float      # placed height (after rotation)


@dataclass
class Container:
    """The shipping container."""
    width: float = 250.0    # x-axis
    depth: float = 500.0    # z-axis
    height: float = 250.0   # y-axis (vertical)
