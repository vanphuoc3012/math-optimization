import cvxpy as cp
import math
print(cp.installed_solvers())

def  solve_use_MI(input, solver):
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
    alpha = input['alpha']  # Fixed to use the correct parameter
    T_max = math.floor(alpha * total_capacity)

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

    # Create and solve the problem
    problem = cp.Problem(cp.Maximize(profit), constraints)
    result = problem.solve(solver=solver, verbose=True)

    # Format the results
    solution = {
        f"x_{fare}": variables[fare].value for fare in ['E1', 'E2', 'S1', 'S2', 'B1']
    }

    total_tickets_sold = sum(solution.values())
    solution['total_tickets_sold'] = total_tickets_sold
    solution['profit'] = profit.value

    # Print results
    print("=== Optimal Solution ===")
    print(f" Economy Standard (x_E1): {solution['x_E1']}")
    print(f" Economy Flexible (x_E2): {solution['x_E2']}")
    print(f" Special Economy Standard (x_S1): {solution['x_S1']}")
    print(f" Special Economy Flexible (x_S2): {solution['x_S2']}")
    print(f" Business Standard (x_B1): {solution['x_B1']}")
    print(f" Total tickets sold: {total_tickets_sold}")
    print(f" Expected Profit: {profit.value:,.0f} VND")
    
    return {
        "x_E1": solution['x_E1'],
        "x_E2": solution['x_E2'],
        "x_S1": solution['x_S1'],
        "x_S2": solution['x_S2'],
        "x_B1": solution['x_B1'],
        "total_tickets_sold": total_tickets_sold,
        "profit": profit.value,
        "comp_cost": sum([compensation_costs[i] * max(show_up_probs[i] * solution[f'x_{i}'] - capacities_by_fare[i], 0) for i in ['E1', 'E2', 'S1', 'S2', 'B1']]),
        "total_revenue": revenue.value
    }