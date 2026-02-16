from __future__ import annotations


def hungarian_assignment(cost_matrix: list[list[float]]) -> tuple[list[int], float]:
    """Return optimal assignment for rectangular cost matrix.

    Returns:
        assignments: list of length rows, value is assigned column index or -1.
        total_cost: sum of selected costs for assigned rows.
    """

    if not cost_matrix or not cost_matrix[0]:
        return [], 0.0

    rows = len(cost_matrix)
    cols = len(cost_matrix[0])

    if rows <= cols:
        row_to_col, cost = _hungarian_rows_leq_cols(cost_matrix)
        return row_to_col, cost

    transposed = [[cost_matrix[r][c] for r in range(rows)] for c in range(cols)]
    col_to_row, cost = _hungarian_rows_leq_cols(transposed)

    row_to_col = [-1] * rows
    for col_idx, row_idx in enumerate(col_to_row):
        if row_idx >= 0:
            row_to_col[row_idx] = col_idx

    return row_to_col, cost


def _hungarian_rows_leq_cols(cost_matrix: list[list[float]]) -> tuple[list[int], float]:
    n = len(cost_matrix)
    m = len(cost_matrix[0])

    u = [0.0] * (n + 1)
    v = [0.0] * (m + 1)
    p = [0] * (m + 1)
    way = [0] * (m + 1)

    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [float("inf")] * (m + 1)
        used = [False] * (m + 1)

        while True:
            used[j0] = True
            i0 = p[j0]
            delta = float("inf")
            j1 = 0

            for j in range(1, m + 1):
                if used[j]:
                    continue
                cur = cost_matrix[i0 - 1][j - 1] - u[i0] - v[j]
                if cur < minv[j]:
                    minv[j] = cur
                    way[j] = j0
                if minv[j] < delta or (minv[j] == delta and j < j1):
                    delta = minv[j]
                    j1 = j

            for j in range(0, m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta

            j0 = j1
            if p[j0] == 0:
                break

        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break

    assignment = [-1] * n
    total_cost = 0.0

    for j in range(1, m + 1):
        if p[j] == 0:
            continue
        row = p[j] - 1
        col = j - 1
        assignment[row] = col

    for row, col in enumerate(assignment):
        if col >= 0:
            total_cost += cost_matrix[row][col]

    return assignment, total_cost
