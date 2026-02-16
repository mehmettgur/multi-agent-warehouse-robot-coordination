from __future__ import annotations

from warehouse_sim.models import Position


class EventEngine:
    """Evaluates dynamic events such as temporary blocks and stochastic delay."""

    def __init__(self, events: list[dict]) -> None:
        self.events = list(events)

    def active_blocked_cells(self, tick: int) -> set[Position]:
        blocked: set[Position] = set()
        for event in self.events:
            if event.get("type") != "temp_block":
                continue
            if not _is_active(event, tick):
                continue
            cell = event.get("cell")
            if isinstance(cell, list) and len(cell) == 2:
                blocked.add((int(cell[0]), int(cell[1])))
        return blocked

    def delayed_robots(
        self,
        tick: int,
        robot_ids: list[str],
        rng,
    ) -> tuple[set[str], int]:
        delayed: set[str] = set()
        delay_event_count = 0

        for event in self.events:
            if event.get("type") != "stochastic_delay":
                continue
            if not _is_active(event, tick):
                continue

            probability = float(event.get("probability", 0.0))
            target_ids = event.get("robot_ids")
            target_set = set(target_ids) if isinstance(target_ids, list) else None

            for robot_id in sorted(robot_ids):
                if target_set is not None and robot_id not in target_set:
                    continue
                if rng.random() < probability:
                    delayed.add(robot_id)
                    delay_event_count += 1

        return delayed, delay_event_count


def _is_active(event: dict, tick: int) -> bool:
    start_tick = int(event.get("start_tick", 0))
    end_tick = int(event.get("end_tick", start_tick))
    return start_tick <= tick <= end_tick
