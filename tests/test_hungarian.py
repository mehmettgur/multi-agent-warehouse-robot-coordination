from warehouse_sim.optimization.hungarian import hungarian_assignment


def test_hungarian_finds_global_minimum_for_square_matrix() -> None:
    costs = [
        [4, 1, 3],
        [2, 0, 5],
        [3, 2, 2],
    ]
    assignment, total = hungarian_assignment(costs)

    assert assignment == [1, 0, 2]
    assert int(total) == 5


def test_hungarian_handles_rectangular_matrix() -> None:
    costs = [
        [10, 1],
        [2, 7],
        [3, 4],
    ]
    assignment, total = hungarian_assignment(costs)

    assigned = [idx for idx in assignment if idx >= 0]
    assert len(assigned) == 2
    assert total >= 0
