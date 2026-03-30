"""3-D bin-packing using the Extreme-Points heuristic with gravity.

The algorithm maintains a list of candidate positions (extreme points).
Each box is placed at the best valid extreme point according to a
bottom-left-back preference (lowest y → lowest z → lowest x).

Constraints enforced:
  * Container bounds
  * No overlap between placed boxes
  * Realistic gravity -- a box must rest on the floor (y ~ 0) **or**
    the combined support area from all boxes directly below must cover
    at least 50 % of the box's bottom-face area.  A box may bridge
    across multiple boxes and the individual support areas are summed.
"""

from typing import List, Tuple

from .models import Box, Container, PlacedBox

_EPS = 0.001  # numerical tolerance


# ── helpers ──────────────────────────────────────────────────────────

def _boxes_overlap(p: PlacedBox,
                   x: float, y: float, z: float,
                   w: float, d: float, h: float) -> bool:
    """True if placed box *p* overlaps with a box at (x,y,z) sized (w,d,h)."""
    return not (
        p.x + p.width  <= x + _EPS or x + w <= p.x + _EPS or
        p.y + p.height <= y + _EPS or y + h <= p.y + _EPS or
        p.z + p.depth  <= z + _EPS or z + d <= p.z + _EPS
    )


def _gravity_ok(placed: List[PlacedBox],
                x: float, y: float, z: float,
                w: float, d: float, h: float,
                min_support: float = 0.50) -> bool:
    """Check that the position is gravity-valid (realistic physics).

    A box is stable when:
      - it sits on the floor (y ~ 0) -- 100 % supported, always OK, **or**
      - the **combined** support area from every box whose top face
        touches this box's bottom face is >= *min_support* (default 50 %)
        of the new box's base area.

    This means a box may rest across two (or more) boxes at once and
    the individual contributions are summed.
    """
    if y < 0.01:
        return True                       # on the floor -- 100 % supported

    box_area = w * d
    if box_area <= 0:
        return False

    total_support = 0.0
    for p in placed:
        if abs(p.y + p.height - y) < 0.01:  # top of p touches bottom of new
            ox = max(0.0, min(p.x + p.width, x + w) - max(p.x, x))
            oz = max(0.0, min(p.z + p.depth, z + d) - max(p.z, z))
            total_support += ox * oz

    return total_support >= min_support * box_area


def _point_inside_any(px: float, py: float, pz: float,
                      placed: List[PlacedBox]) -> bool:
    """True if the point sits strictly inside any placed box."""
    for b in placed:
        if (b.x + _EPS < px < b.x + b.width  - _EPS and
            b.y + _EPS < py < b.y + b.height - _EPS and
            b.z + _EPS < pz < b.z + b.depth  - _EPS):
            return True
    return False


# ── main packer ──────────────────────────────────────────────────────

def pack_boxes(container: Container,
               boxes: List[Box],
               order: List[int],
               rotations: List[int]) -> List[PlacedBox]:
    """Place *boxes* into *container* in the given *order* / *rotations*.

    Parameters
    ----------
    container : Container
    boxes     : full list of Box objects
    order     : permutation of box indices – the sequence to try
    rotations : rotation index for each box (index into ``box.get_rotations()``)

    Returns
    -------
    list[PlacedBox] – only the boxes that could be successfully placed.
    """
    placed: List[PlacedBox] = []
    eps: List[Tuple[float, float, float]] = [(0.0, 0.0, 0.0)]

    for idx in order:
        if idx < 0 or idx >= len(boxes):
            continue

        box = boxes[idx]
        rots = box.get_rotations()
        rot_idx = rotations[idx] % len(rots)
        w, d, h = rots[rot_idx]

        # Filter out extreme points that landed inside a placed box
        valid = [ep for ep in set(eps)
                 if not _point_inside_any(ep[0], ep[1], ep[2], placed)]
        # Prefer bottom → back → left
        valid.sort(key=lambda p: (p[1], p[2], p[0]))

        best_pos = None
        for (ex, ey, ez) in valid:
            # --- container bounds ---
            if ex + w > container.width  + 0.01:
                continue
            if ey + h > container.height + 0.01:
                continue
            if ez + d > container.depth  + 0.01:
                continue

            # --- collision ---
            if any(_boxes_overlap(p, ex, ey, ez, w, d, h) for p in placed):
                continue

            # --- gravity ---
            if not _gravity_ok(placed, ex, ey, ez, w, d, h):
                continue

            best_pos = (ex, ey, ez)
            break                          # first valid is best (sorted)

        if best_pos is not None:
            px, py, pz = best_pos
            placed.append(PlacedBox(box=box, x=px, y=py, z=pz,
                                    width=w, depth=d, height=h))

            # Four new extreme points from the placed box
            candidates = [
                (px + w, py,     pz),       # right
                (px,     py,     pz + d),   # front
                (px,     py + h, pz),       # top
                (px + w, py,     pz + d),   # corner
            ]
            for pt in candidates:
                if (pt[0] <= container.width  + 0.01 and
                    pt[1] <= container.height + 0.01 and
                    pt[2] <= container.depth  + 0.01):
                    eps.append(pt)

    return placed
