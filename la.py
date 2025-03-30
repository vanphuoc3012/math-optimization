import math
import random

import cvxpy as cp
import numpy as np
import math

def la_phuoc(input):
    # Extract input parameters
    ticket_prices = {
        'E1': input['r_E1'],  # Economy Standard
        'E2': input['r_E2'],  # Economy Flexible
        'S1': input['r_S1'],  # Special Economy Standard
        'S2': input['r_S2'],  # Special Economy Flexible
        'B1': input['r_B1']   # Business Standard
    }

    # Calculate compensation costs (200% of ticket price)
    compensation_costs = {k: input['boi_thuong'] * v * 0.01 for k, v in ticket_prices.items()}

    # Seating capacities
    capacities = {
        'E': input['S_E'],    # Economy
        'S': input['S_S'],    # Special Economy
        'B': input['S_B']     # Business
    }
    total_capacity = sum(capacities.values())

    # Seat allocation by fare level (percentage)
    capacities_by_fare = {
        'E1': math.floor(input['S_E1'] * capacities['E'] * 0.01),
        'S1': math.floor(input['S_S1'] * capacities['S'] * 0.01),
        'B1': math.floor(input['S_B1'] * capacities['B'] * 0.01)
    }
    # Remaining seats for flexible fares
    capacities_by_fare['E2'] = capacities['E'] - capacities_by_fare['E1']
    capacities_by_fare['S2'] = capacities['S'] - capacities_by_fare['S1']

    # Show-up probabilities
    show_up_probs = {
        'E1': input['p_E1'],
        'E2': input['p_E2'],
        'S1': input['p_S1'],
        'S2': input['p_S2'],
        'B1': input['p_B1']
    }

    # Overbooking factor
    alpha = input['alpha']
    T_max = math.floor(alpha * total_capacity)


    # ---- Lagrangian Relaxation Approach ----

    # Initialize Lagrangian multipliers
    lagrangian_multipliers = {
        'total_capacity': 0.0,  # For the total tickets constraint
        'E': 0.0,  # For Economy class constraint
        'S': 0.0,  # For Special Economy class constraint
        'B': 0.0,  # For Business class constraint
    }

    # Parameters for subgradient method
    max_iterations = 100
    step_size = 1.0
    best_solution = None
    best_profit = float('-inf')

    # Variables to track convergence
    prev_lagrangian = float('-inf')
    no_improvement_count = 0

    for iteration in range(max_iterations):
        # Solve the relaxed problem
        relaxed_solution, relaxed_profit = solve_relaxed_problem(
            ticket_prices, compensation_costs, capacities, capacities_by_fare,
            show_up_probs, T_max, lagrangian_multipliers
        )

        # Check if solution is feasible for the original problem
        is_feasible = check_feasibility(
            relaxed_solution, T_max, capacities_by_fare
        )

        if is_feasible and relaxed_profit > best_profit:
            best_solution = relaxed_solution
            best_profit = relaxed_profit

        # If we have a feasible solution, try to improve it
        if is_feasible:
            improved_solution = local_search(
                relaxed_solution, ticket_prices, compensation_costs,
                capacities, capacities_by_fare, show_up_probs, T_max
            )

            # Calculate profit for improved solution
            improved_profit = calculate_profit(
                improved_solution, ticket_prices, compensation_costs,
                capacities_by_fare, show_up_probs
            )

            if improved_profit > best_profit:
                best_solution = improved_solution
                best_profit = improved_profit

        # Check for convergence
        if abs(relaxed_profit - prev_lagrangian) < 1e-5:
            no_improvement_count += 1
            if no_improvement_count >= 5:
                break
        else:
            no_improvement_count = 0

        prev_lagrangian = relaxed_profit

        # Update Lagrangian multipliers using subgradient method
        update_lagrangian_multipliers(
            lagrangian_multipliers, relaxed_solution, T_max,
            capacities_by_fare, step_size, iteration
        )

        # Reduce step size gradually
        step_size = step_size * 0.95

    # If no feasible solution was found, generate one using the original CVXPY approach
    if best_solution is None:
        print('No feasible solution was found, try solve generate one using the original CVXPY approach')
        # best_solution, best_profit = solve_with_cvxpy(
        #     ticket_prices, compensation_costs, capacities, capacities_by_fare,
        #     show_up_probs, T_max
        # )
        return {}

     # Calculate expected overbooked passengers for each fare level
    variables_result = {
         'x_E1': int(best_solution['E1']),
        'x_E2': int(best_solution['E2']),
        'x_S1': int(best_solution['S1']),
        'x_S2': int(best_solution['S2']),
        'x_B1': int(best_solution['B1']),
     }
    overbooked = {
        fare: cp.pos(show_up_probs[fare] * variables_result[f'x_{fare}'] - capacities_by_fare[fare])
        for fare in ['E1', 'E2', 'S1', 'S2', 'B1']
    }

    # Calculate compensation costs
    compensation = sum(overbooked[fare] * compensation_costs[fare]
                       for fare in ['E1', 'E2', 'S1', 'S2', 'B1'])
    # Format the final solution
    result = {
        'x_E1': int(best_solution['E1']),
        'x_E2': int(best_solution['E2']),
        'x_S1': int(best_solution['S1']),
        'x_S2': int(best_solution['S2']),
        'x_B1': int(best_solution['B1']),
        'total_revenue': sum(ticket_prices[fare] * best_solution[fare] for fare in ['E1', 'E2', 'S1', 'S2', 'B1']),
        "total_tickets_sold": sum(best_solution[fare] for fare in ['E1', 'E2', 'S1', 'S2', 'B1']),
        "profit": best_profit,
        'comp_cost': sum([compensation_costs[i] * max(show_up_probs[i] * best_solution[f'{i}'] - capacities_by_fare[i], 0) for i in ['E1', 'E2', 'S1', 'S2', 'B1']])
    }

    return result

def solve_relaxed_problem(ticket_prices, compensation_costs, capacities, capacities_by_fare, show_up_probs, T_max, lagrangian_multipliers):
    """Solve the Lagrangian relaxed problem for a given set of multipliers"""
    solution = {'E1': 0, 'E2': 0, 'S1': 0, 'S2': 0, 'B1': 0}

    # For each fare type, determine the optimal number of tickets to sell
    for fare in ['E1', 'E2', 'S1', 'S2', 'B1']:
        # Calculate marginal profit for this fare type
        revenue = ticket_prices[fare]
        expected_compensation = show_up_probs[fare] * compensation_costs[fare]

        # Apply Lagrangian penalty
        if fare in ['E1', 'E2']:
            class_penalty = lagrangian_multipliers['E']
        elif fare in ['S1', 'S2']:
            class_penalty = lagrangian_multipliers['S']
        else:  # B1
            class_penalty = lagrangian_multipliers['B']

        total_penalty = lagrangian_multipliers['total_capacity'] + class_penalty

        # Determine optimal quantity: if profitable after penalties, sell up to capacity
        if revenue - expected_compensation > total_penalty:
            # Maximum possible tickets for this fare
            if fare in ['E1', 'S1', 'B1']:
                solution[fare] = math.floor(capacities_by_fare[fare] / show_up_probs[fare])
            else:  # Flexible fares
                solution[fare] = math.floor(capacities_by_fare[fare] / show_up_probs[fare]) + 5
        else:
            # Minimum required tickets
            solution[fare] = capacities_by_fare[fare]

    # Calculate profit for this solution
    profit = calculate_profit(solution, ticket_prices, compensation_costs, capacities_by_fare, show_up_probs)

    # Apply Lagrangian penalties
    total_tickets = sum(solution.values())
    penalty = lagrangian_multipliers['total_capacity'] * max(0, total_tickets - T_max)

    # Class-specific penalties
    class_penalties = {
        'E': solution['E1'] + solution['E2'] - (capacities_by_fare['E1'] + capacities_by_fare['E2']),
        'S': solution['S1'] + solution['S2'] - (capacities_by_fare['S1'] + capacities_by_fare['S2']),
        'B': solution['B1'] - capacities_by_fare['B1']
    }

    for class_type in ['E', 'S', 'B']:
        if class_penalties[class_type] > 0:
            penalty += lagrangian_multipliers[class_type] * class_penalties[class_type]

    # Lagrangian relaxed objective
    lagrangian_profit = profit - penalty

    return solution, lagrangian_profit

def calculate_profit(solution, ticket_prices, compensation_costs, capacities_by_fare, show_up_probs):
    """Calculate expected profit for a given solution"""
    # Revenue
    revenue = sum(ticket_prices[fare] * solution[fare] for fare in ['E1', 'E2', 'S1', 'S2', 'B1'])

    # Expected compensation for overbooking
    compensation = 0
    for fare in ['E1', 'E2', 'S1', 'S2', 'B1']:
        expected_passengers = show_up_probs[fare] * solution[fare]
        overbooked = max(0, expected_passengers - capacities_by_fare[fare])
        compensation += overbooked * compensation_costs[fare]

    return revenue - compensation

def check_feasibility(solution, T_max, capacities_by_fare):
    """Check if the solution satisfies all constraints"""
    # Check total tickets constraint
    total_tickets = sum(solution.values())
    if total_tickets > T_max:
        return False

    # Check minimum capacity constraints
    for fare in ['E1', 'E2', 'S1', 'S2', 'B1']:
        if solution[fare] < capacities_by_fare[fare]:
            return False

    return True

def update_lagrangian_multipliers(multipliers, solution, T_max, capacities_by_fare, step_size, iteration):
    """Update Lagrangian multipliers using subgradient method"""
    # Calculate violations
    total_tickets = sum(solution.values())
    total_violation = max(0, total_tickets - T_max)

    # Class-specific violations
    class_violations = {
        'E': solution['E1'] + solution['E2'] - (capacities_by_fare['E1'] + capacities_by_fare['E2']),
        'S': solution['S1'] + solution['S2'] - (capacities_by_fare['S1'] + capacities_by_fare['S2']),
        'B': solution['B1'] - capacities_by_fare['B1']
    }

    # Update multipliers
    multipliers['total_capacity'] = max(0, multipliers['total_capacity'] + step_size * total_violation)

    for class_type in ['E', 'S', 'B']:
        if class_violations[class_type] > 0:
            multipliers[class_type] = max(0, multipliers[class_type] + step_size * class_violations[class_type])
        else:
            # Reduce multiplier if no violation
            multipliers[class_type] = max(0, multipliers[class_type] - step_size * 0.1)

def local_search(initial_solution, ticket_prices, compensation_costs, capacities, capacities_by_fare, show_up_probs, T_max):
    """Try to improve the solution using local search"""
    current_solution = initial_solution.copy()
    current_profit = calculate_profit(current_solution, ticket_prices, compensation_costs, capacities_by_fare, show_up_probs)

    improved = True
    while improved:
        improved = False

        # Try increasing each fare type
        for fare in ['E1', 'E2', 'S1', 'S2', 'B1']:
            test_solution = current_solution.copy()
            test_solution[fare] += 1

            # Check if still feasible
            if check_feasibility(test_solution, T_max, capacities_by_fare):
                test_profit = calculate_profit(test_solution, ticket_prices, compensation_costs, capacities_by_fare, show_up_probs)

                if test_profit > current_profit:
                    current_solution = test_solution
                    current_profit = test_profit
                    improved = True

        # Try decreasing each fare type (if above minimum)
        for fare in ['E1', 'E2', 'S1', 'S2', 'B1']:
            if current_solution[fare] > capacities_by_fare[fare]:
                test_solution = current_solution.copy()
                test_solution[fare] -= 1

                test_profit = calculate_profit(test_solution, ticket_prices, compensation_costs, capacities_by_fare, show_up_probs)

                if test_profit > current_profit:
                    current_solution = test_solution
                    current_profit = test_profit
                    improved = True

    return current_solution

def solve_with_cvxpy(ticket_prices, compensation_costs, capacities, capacities_by_fare, show_up_probs, T_max):
    """Fallback method using CVXPY if no feasible solution is found with Lagrangian relaxation"""
    # Define decision variables
    variables = {}
    # Tickets to sell for each fare level
    for fare in ['E1', 'E2', 'S1', 'S2', 'B1']:
        variables[fare] = cp.Variable(integer=True, name=f"x_{fare}")

    # Total tickets by class
    variables['E'] = cp.Variable(integer=True, name="x_E")
    variables['S'] = cp.Variable(integer=True, name="x_S")
    variables['B'] = cp.Variable(integer=True, name="x_B")

    # Calculate expected revenue
    revenue = sum(ticket_prices[fare] * variables[fare] for fare in ['E1', 'E2', 'S1', 'S2', 'B1'])

    # Calculate expected overbooked passengers for each fare level
    overbooked = {
        fare: cp.pos(show_up_probs[fare] * variables[fare] - capacities_by_fare[fare])
        for fare in ['E1', 'E2', 'S1', 'S2', 'B1']
    }

    # Calculate compensation costs
    compensation = sum(overbooked[fare] * compensation_costs[fare]
                    for fare in ['E1', 'E2', 'S1', 'S2', 'B1'])

    # Calculate profit
    profit = revenue - compensation

    # Define constraints
    constraints = [
        # Fare level sums
        variables['E1'] + variables['E2'] == variables['E'],
        variables['S1'] + variables['S2'] == variables['S'],
        variables['B1'] == variables['B'],

        # Total tickets limit
        sum(variables[fare] for fare in ['E1', 'E2', 'S1', 'S2', 'B1']) <= T_max,

        # Minimum capacity constraints
        *[variables[fare] >= capacities_by_fare[fare] for fare in ['E1', 'E2', 'S1', 'S2', 'B1']],

        # Non-negativity constraints
        *[variables[fare] >= 0 for fare in ['E1', 'E2', 'S1', 'S2', 'B1']]
    ]

    # Define and solve the problem
    problem = cp.Problem(cp.Maximize(profit), constraints)
    problem.solve(solver=cp.GLPK_MI)

    # Extract solution
    solution = {fare: int(variables[fare].value) for fare in ['E1', 'E2', 'S1', 'S2', 'B1']}
    profit_value = profit.value

    return solution, profit_value