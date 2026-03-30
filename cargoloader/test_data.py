"""Sample boxes for testing the packing algorithm."""

from typing import List

from .models import Box

# Distinct colours for up to 20 boxes (R, G, B in 0-1 range)
BOX_COLORS = [
    (0.94, 0.35, 0.35),   # Red
    (0.35, 0.75, 0.40),   # Green
    (0.35, 0.55, 0.95),   # Blue
    (0.95, 0.75, 0.25),   # Gold
    (0.75, 0.40, 0.85),   # Purple
    (0.30, 0.82, 0.82),   # Cyan
    (0.95, 0.58, 0.30),   # Orange
    (0.65, 0.85, 0.35),   # Lime
    (0.90, 0.45, 0.65),   # Pink
    (0.55, 0.65, 0.90),   # Periwinkle
    (0.80, 0.65, 0.40),   # Brown
    (0.60, 0.90, 0.65),   # Mint
    (0.85, 0.55, 0.90),   # Lavender
    (0.50, 0.75, 0.55),   # Sage
    (0.90, 0.80, 0.50),   # Tan
]


def create_test_boxes() -> List[Box]:
    """Create a varied set of 15 test boxes for a 250×500×250 container.

    The total box volume is ~80 % of container volume, so the GA must
    find a smart ordering / rotation to fit as many as possible.

    Each tuple: (id, name, width, depth, height, weight_kg, can_rotate)
    """
    data = [
        (1,  "Crate Alpha",    120, 200, 100,  50.0, True),
        (2,  "Crate Bravo",    100, 150,  80,  35.0, True),
        (3,  "Pallet Charlie", 240, 300,  60,  45.0, True),
        (4,  "Box Delta",       80,  80,  80,  20.0, True),
        (5,  "Box Echo",        80, 120,  80,  15.0, True),
        (6,  "Drum Foxtrot",    80,  80, 160,  45.0, False),
        (7,  "Flat Golf",      200, 400,  50,  25.0, True),
        (8,  "Small Hotel",     50,  50,  50,  10.0, True),
        (9,  "Long India",      50, 450,  50,  30.0, True),
        (10, "Cube Juliet",    120, 120, 120,  35.0, True),
        (11, "Wide Kilo",      160, 120,  80,  25.0, True),
        (12, "Tall Lima",       60,  60, 220,  40.0, True),
        (13, "Slab Mike",      230, 350,  30,  15.0, True),
        (14, "Block November", 100, 160, 100,  22.0, True),
        (15, "Beam Oscar",      40, 400,  40,  18.0, True),
    ]

    boxes: List[Box] = []
    for i, (bid, name, w, d, h, wt, rot) in enumerate(data):
        color = BOX_COLORS[i % len(BOX_COLORS)]
        boxes.append(Box(bid, name, float(w), float(d), float(h),
                         wt, rot, color))
    return boxes
