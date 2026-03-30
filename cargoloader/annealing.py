"""Simulated Annealing for refining a 3-D packing solution.

After the GA has found a good region of the search space, SA performs
intensive local search.  At each step it applies a small random
perturbation (swap, insert, segment-reverse, or rotation flip) and
accepts worse solutions probabilistically via the Metropolis criterion,
which helps escape local optima that the GA may have converged to.

Temperature cools geometrically from *t_start* to *t_min* over the
configured number of iterations.
"""

import copy
import math
import random
from typing import Callable, List, Optional, Tuple

from .models import Box, Container, PlacedBox
from .packing import pack_boxes

Individual = Tuple[List[int], List[int]]


class SimulatedAnnealing:
    """SA that refines a single packing solution."""

    def __init__(
        self,
        container: Container,
        boxes: List[Box],
        initial: Optional[Individual] = None,
        iterations: int = 2000,
        t_start: float = 0.05,
        t_min: float = 0.0001,
    ):
        self.container = container
        self.boxes = boxes
        self.n = len(boxes)
        self.iterations = max(iterations, 1)
        self.t_start = t_start
        self.t_min = t_min

        # Geometric cooling so T goes t_start -> t_min over all iterations
        self.alpha = (t_min / t_start) ** (1.0 / self.iterations)

        # Starting solution
        if initial is not None:
            self._current: Individual = (list(initial[0]), list(initial[1]))
        else:
            order = list(range(self.n))
            random.shuffle(order)
            self._current = (order, [random.randint(0, 5) for _ in range(self.n)])

        self.best_fitness: float = 0.0
        self.best_placed: List[PlacedBox] = []

        # on_progress(iteration, total_iters, best_fitness, placed_or_None)
        self.on_progress: Optional[Callable] = None
        self._stop = False

    # ── public ────────────────────────────────────────────────────

    def stop(self) -> None:
        self._stop = True

    def run(self) -> Tuple[List[PlacedBox], float]:
        self._stop = False

        # Evaluate starting point
        cur_fit, cur_placed = self._evaluate(self._current)
        self.best_fitness = cur_fit
        self.best_placed = cur_placed

        temp = self.t_start

        for it in range(self.iterations):
            if self._stop:
                break

            neighbour = self._neighbour(self._current)
            nb_fit, nb_placed = self._evaluate(neighbour)

            delta = nb_fit - cur_fit            # positive = improvement
            if delta > 0 or random.random() < math.exp(delta / max(temp, 1e-12)):
                self._current = neighbour
                cur_fit = nb_fit
                cur_placed = nb_placed

                if cur_fit > self.best_fitness:
                    self.best_fitness = cur_fit
                    self.best_placed = cur_placed

            temp *= self.alpha

            if self.on_progress:
                send = (copy.deepcopy(self.best_placed)
                        if it % 50 == 0 or it == self.iterations - 1
                        else None)
                self.on_progress(it, self.iterations, self.best_fitness, send)

        return self.best_placed, self.best_fitness

    # ── internals ─────────────────────────────────────────────────

    def _evaluate(self, ind: Individual) -> Tuple[float, List[PlacedBox]]:
        order, rotations = ind
        placed = pack_boxes(self.container, self.boxes, order, rotations)
        cvol = self.container.width * self.container.depth * self.container.height
        used = sum(p.width * p.depth * p.height for p in placed)
        vol_ratio = used / cvol if cvol > 0 else 0.0
        count_ratio = len(placed) / self.n if self.n > 0 else 0.0
        fitness = 0.6 * vol_ratio + 0.4 * count_ratio
        return fitness, placed

    def _neighbour(self, ind: Individual) -> Individual:
        """Return a slightly-perturbed copy of *ind*."""
        order, rotations = list(ind[0]), list(ind[1])

        if self.n < 2:
            # Only rotation flips possible
            if self.n == 1:
                rotations[0] = random.randint(0, 5)
            return (order, rotations)

        move = random.random()

        if move < 0.30:
            # Swap two random positions in the order
            i, j = random.sample(range(self.n), 2)
            order[i], order[j] = order[j], order[i]

        elif move < 0.55:
            # Remove an element and re-insert elsewhere
            i = random.randrange(self.n)
            elem = order.pop(i)
            j = random.randrange(self.n)
            order.insert(j, elem)

        elif move < 0.75:
            # Reverse a small segment (2-opt style)
            i, j = sorted(random.sample(range(self.n), 2))
            order[i : j + 1] = reversed(order[i : j + 1])

        else:
            # Flip one box's rotation
            i = random.randrange(self.n)
            rotations[i] = random.randint(0, 5)

        return (order, rotations)
