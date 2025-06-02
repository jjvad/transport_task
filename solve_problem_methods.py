import numpy as np

def is_closed_problem(suppliers, consumers):
    return sum(suppliers) == sum(consumers)

def balance_problem(suppliers, consumers, cost_matrix):
    total_supply = sum(suppliers)
    total_demand = sum(consumers)

    if total_supply > total_demand:
        for row in cost_matrix:
            row.append(0)
        consumers.append(total_supply - total_demand)
    elif total_demand > total_supply:
        cost_matrix.append([0] * len(consumers))
        suppliers.append(total_demand - total_supply)

    return suppliers, consumers, cost_matrix

def find_loop(plan, bi, bj):
    """
    Ищет цикл пересчёта, начиная с указанной позиции.
    Возвращает список координат, образующих цикл.
    """
    m, n = plan.shape
    path = []
    visited = set()

    def dfs(i, j, direction='row', start=True):
        if not start and i == bi and j == bj:
            path.append((i, j))
            return True
        if (i, j) in visited:
            return False
        visited.add((i, j))
        path.append((i, j))

        # Переключаем направление по строкам/столбцам
        if direction == 'row':
            for nj in range(n):
                if nj != j and plan[i][nj] > 0:
                    if dfs(i, nj, 'col', False):
                        return True
        else:
            for ni in range(m):
                if ni != i and plan[ni][j] > 0:
                    if dfs(ni, j, 'row', False):
                        return True

        path.pop()
        return False

    if dfs(bi, bj):
        return path
    return []

def potential_method(suppliers, consumers, cost_matrix):
    m = len(suppliers)
    n = len(consumers)

    supply = np.array(suppliers)
    demand = np.array(consumers)
    costs = np.array(cost_matrix)
    plan = np.zeros((m, n))

    # Формирование начального опорного плана методом минимальных элементов
    indices = [(i, j) for i in range(m) for j in range(n)]
    sorted_indices = sorted(indices, key=lambda x: costs[x[0]][x[1]])

    used_cells = []

    for i, j in sorted_indices:
        if supply[i] > 0 and demand[j] > 0:
            amount = min(supply[i], demand[j])
            plan[i][j] = amount
            supply[i] -= amount
            demand[j] -= amount
            used_cells.append((i, j))
            if len(used_cells) >= m + n - 1:
                break

    # Добавляем фиктивные нулевые перевозки, если нужно
    while len(used_cells) < m + n - 1:
        for i in range(m):
            for j in range(n):
                if plan[i][j] == 0 and (i, j) not in used_cells:
                    plan[i][j] = 0.0001  # Избегаем вырожденности
                    used_cells.append((i, j))
                    break
            if len(used_cells) >= m + n - 1:
                break

    iteration = 0
    while True:
        iteration += 1

        # Вычисление потенциалов
        u = [None] * m
        v = [None] * n
        u[0] = 0

        filled_cells = [(i, j) for i in range(m) for j in range(n) if plan[i][j] > 0]

        updated = True
        while updated:
            updated = False
            for i, j in filled_cells:
                if u[i] is not None and v[j] is None:
                    v[j] = costs[i][j] - u[i]
                    updated = True
                elif v[j] is not None and u[i] is None:
                    u[i] = costs[i][j] - v[j]
                    updated = True


        # Проверка на оптимальность
        delta_ij = np.zeros((m, n))
        has_positive = False
        for i in range(m):
            for j in range(n):
                if plan[i][j] == 0:
                    delta_ij[i][j] = costs[i][j] - (u[i] + v[j])
                    if delta_ij[i][j] < 0:
                        has_positive = True

        if not has_positive:
            break

        # Цикл перераспределения
        entering_i, entering_j = np.unravel_index(np.argmin(delta_ij), delta_ij.shape)
        loop = find_loop(plan, entering_i, entering_j)

        if not loop:
            # Добавляем искусственную малую перевозку
            for i in range(m):
                for j in range(n):
                    if plan[i][j] == 0:
                        plan[i][j] = 0.0001
                        loop = find_loop(plan, entering_i, entering_j)
                        if loop:
                            break
                if loop:
                    break
            if not loop:
                raise ValueError("Цикл так и не найден. Задача слишком сложная.")

        theta = min(plan[i][j] for i, j in loop[1::2])

        for idx, (i, j) in enumerate(loop):
            if idx % 2 == 0:
                plan[i][j] += theta
            else:
                plan[i][j] -= theta

        # Обнуляем ячейки, где значение стало равно нулю
        for i in range(m):
            for j in range(n):
                if abs(plan[i][j]) < 1e-6:
                    plan[i][j] = 0

    return plan

def solve_transport_problem(suppliers, consumers, cost_matrix):

    if not is_closed_problem(suppliers, consumers):
        suppliers_balanced, consumers_balanced, cost_matrix_balanced = balance_problem(
            suppliers.copy(), consumers.copy(), [row.copy() for row in cost_matrix]
        )
    else:
        suppliers_balanced = suppliers
        consumers_balanced = consumers
        cost_matrix_balanced = cost_matrix

    solution_plan = potential_method(suppliers_balanced, consumers_balanced, cost_matrix_balanced)


    total_cost = sum(
        solution_plan[i][j] * cost_matrix_balanced[i][j]
        for i in range(len(suppliers_balanced)) for j in range(len(consumers_balanced))
    )


    return solution_plan, total_cost
