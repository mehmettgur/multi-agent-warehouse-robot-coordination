from __future__ import annotations

from dataclasses import dataclass

from warehouse_sim.models import Position


@dataclass(frozen=True)
class GridMap:
    width: int
    height: int
    obstacles: set[Position]

    def in_bounds(self, pos: Position) -> bool:
        x, y = pos
        return 0 <= x < self.width and 0 <= y < self.height

    def passable(self, pos: Position) -> bool:
        return pos not in self.obstacles

    def is_valid(self, pos: Position) -> bool:
        return self.in_bounds(pos) and self.passable(pos)

    def neighbors4(self, pos: Position) -> list[Position]:
        x, y = pos
        candidates = [(x, y - 1), (x + 1, y), (x, y + 1), (x - 1, y)]
        return [cell for cell in candidates if self.is_valid(cell)]

    def neighbors_with_wait(self, pos: Position) -> list[Position]:
        return self.neighbors4(pos) + [pos]

    @staticmethod
    def manhattan(a: Position, b: Position) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
