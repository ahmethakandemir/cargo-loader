"""Genetic Algorithm for optimising 3-D container packing.

Chromosome
----------
Each individual is a tuple ``(order, rotations)`` where
* **order** is a permutation of box indices (placement sequence), and
* **rotations** is a list of rotation indices, one per box.

Operators
---------
* Selection  – tournament (k = 3)
* Crossover  – Order Crossover (OX) for the permutation part,
               uniform crossover for rotations.
* Mutation   – swap + segment-reversal for the permutation,
               random rotation flip for the rotation part.
* Elitism    – the two fittest individuals carry over unchanged.
"""

import copy
import random
from typing import Callable, List, Optional, Tuple

from .models import Box, Container, PlacedBox
from .packing import pack_boxes

Individual = Tuple[List[int], List[int]]


class GeneticAlgorithm:
    """GA that maximises container-fill quality."""

    def __init__(self,
                 container: Container,
                 boxes: List[Box],
                 pop_size: int = 60,
                 generations: int = 100,
                 crossover_rate: float = 0.85,
                 mutation_rate: float = 0.20):
        self.container = container
        self.boxes = boxes
        self.pop_size = max(pop_size, 6)
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.n = len(boxes)

        self.best_fitness: float = 0.0
        self.best_placed: List[PlacedBox] = []
        self.best_individual: Optional[Individual] = None

        # Called as  on_progress(gen, total_gens, fitness, placed_or_None)
        self.on_progress: Optional[Callable] = None
        self._stop = False

    # ── public ────────────────────────────────────────────────────

    def stop(self) -> None:
        self._stop = True

    def run(self) -> Tuple[List[PlacedBox], float]:
        self._stop = False
        population = [self._random_individual() for _ in range(self.pop_size)]

        for gen in range(self.generations):
            if self._stop:
                break

            # Evaluate
            results = [self._evaluate(ind) for ind in population]
            fitnesses = [r[0] for r in results]

            # Track best
            best_idx = max(range(len(fitnesses)), key=lambda i: fitnesses[i])
            if fitnesses[best_idx] > self.best_fitness:
                self.best_fitness = fitnesses[best_idx]
                self.best_placed = results[best_idx][1]
                self.best_individual = population[best_idx]

            # Report progress (send solution every 5 gens to avoid overhead)
            if self.on_progress:
                send_placed = (copy.deepcopy(self.best_placed)
                               if gen % 5 == 0 or gen == self.generations - 1
                               else None)
                self.on_progress(gen, self.generations,
                                 self.best_fitness, send_placed)

            # Build next generation
            ranked = sorted(range(len(fitnesses)),
                            key=lambda i: fitnesses[i], reverse=True)
            new_pop: List[Individual] = [
                population[ranked[0]],
                population[ranked[1]],
            ]

            while len(new_pop) < self.pop_size:
                p1 = self._tournament(population, fitnesses)
                p2 = self._tournament(population, fitnesses)
                child = (self._crossover(p1, p2)
                         if random.random() < self.crossover_rate
                         else (list(p1[0]), list(p1[1])))
                child = self._mutate(child)
                new_pop.append(child)

            population = new_pop

        return self.best_placed, self.best_fitness

    # ── internals ─────────────────────────────────────────────────

    def _random_individual(self) -> Individual:
        order = list(range(self.n))
        random.shuffle(order)
        rotations = [random.randint(0, 5) for _ in range(self.n)]
        return (order, rotations)

    def _evaluate(self, ind: Individual) -> Tuple[float, List[PlacedBox]]:
        order, rotations = ind
        placed = pack_boxes(self.container, self.boxes, order, rotations)

        cvol = (self.container.width * self.container.depth
                * self.container.height)
        used = sum(p.width * p.depth * p.height for p in placed)

        vol_ratio   = used / cvol if cvol > 0 else 0.0
        count_ratio = len(placed) / self.n if self.n > 0 else 0.0

        fitness = 0.6 * vol_ratio + 0.4 * count_ratio
        return fitness, placed

    def _tournament(self, pop: List[Individual],
                    fits: List[float], k: int = 3) -> Individual:
        idxs = random.sample(range(len(pop)), min(k, len(pop)))
        return pop[max(idxs, key=lambda i: fits[i])]

    # ── crossover ─────────────────────────────────────────────────

    def _order_crossover(self, p1: List[int], p2: List[int]) -> List[int]:
        """OX crossover for a permutation."""
        n = len(p1)
        if n < 2:
            return list(p1)
        a, b = sorted(random.sample(range(n), 2))
        child = [None] * n
        child[a:b + 1] = p1[a:b + 1]
        segment = set(p1[a:b + 1])
        pos = (b + 1) % n
        for gene in p2:
            if gene not in segment:
                child[pos] = gene
                pos = (pos + 1) % n
        return child  # type: ignore[return-value]

    def _crossover(self, pa: Individual, pb: Individual) -> Individual:
        child_order = self._order_crossover(list(pa[0]), list(pb[0]))
        child_rot = [pa[1][i] if random.random() < 0.5 else pb[1][i]
                     for i in range(self.n)]
        return (child_order, child_rot)

    # ── mutation ──────────────────────────────────────────────────

    def _mutate(self, ind: Individual) -> Individual:
        order, rotations = list(ind[0]), list(ind[1])

        # swap two positions
        if random.random() < self.mutation_rate and self.n >= 2:
            i, j = random.sample(range(self.n), 2)
            order[i], order[j] = order[j], order[i]

        # reverse a small segment
        if random.random() < self.mutation_rate and self.n >= 2:
            i, j = sorted(random.sample(range(self.n), 2))
            order[i:j + 1] = reversed(order[i:j + 1])

        # flip individual rotations
        for i in range(self.n):
            if random.random() < self.mutation_rate * 2.0 / max(self.n, 1):
                rotations[i] = random.randint(0, 5)

        return (order, rotations)
