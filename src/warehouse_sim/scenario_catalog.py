from __future__ import annotations

from pathlib import Path
from typing import Literal

PaperSuite = Literal["main", "allocator", "planner", "robustness", "all"]

REPO_ROOT = Path(__file__).resolve().parents[2]
SCENARIO_DIR = REPO_ROOT / "configs" / "scenarios"

CORE_SCENARIOS: tuple[str, ...] = (
    "narrow_corridor_swap",
    "intersection_4way_crossing",
    "bottleneck_shelves",
    "high_load_6r_30t",
)
APPENDIX_SCENARIOS: tuple[str, ...] = (
    "dynamic_obstacle",
    "stochastic_delay",
)
LEGACY_SCENARIOS: tuple[str, ...] = (
    "narrow_corridor",
    "dense_tasks",
)

SUITE_SCENARIOS: dict[str, tuple[str, ...]] = {
    "main": CORE_SCENARIOS,
    "allocator": (
        "bottleneck_shelves",
        "high_load_6r_30t",
        "stochastic_delay",
    ),
    "planner": CORE_SCENARIOS + ("dynamic_obstacle",),
    "robustness": (
        "dynamic_obstacle",
        "stochastic_delay",
    ),
}

def scenario_path(name: str) -> Path:
    return SCENARIO_DIR / f"{name}.json"


def scenario_paths(names: tuple[str, ...] | list[str]) -> list[str]:
    return [str(scenario_path(name)) for name in names]

def scenarios_for_suite(suite: PaperSuite) -> tuple[str, ...]:
    if suite == "all":
        seen: list[str] = []
        for item in ("main", "allocator", "planner", "robustness"):
            for scenario in SUITE_SCENARIOS[item]:
                if scenario not in seen:
                    seen.append(scenario)
        return tuple(seen)
    return SUITE_SCENARIOS[suite]

def list_available_scenarios(include_appendix: bool = True, include_legacy: bool = False) -> list[str]:
    selected = list(CORE_SCENARIOS)
    if include_appendix:
        selected.extend(APPENDIX_SCENARIOS)
    if include_legacy:
        selected.extend(LEGACY_SCENARIOS)
    return selected
