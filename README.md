# math-optimization

# Math Optimization for Overbooking Problem

This repository contains implementations and simulations for solving an overbooking optimization problem in the airline industry. The goal is to maximize profit by determining the optimal number of tickets to sell for different fare classes while considering constraints such as seat capacity, overbooking limits, and compensation costs for bumped passengers.

## Problem Description

Airlines often sell more tickets than the available seats on a flight to account for no-show passengers. However, overbooking can lead to situations where more passengers show up than the available seats, requiring the airline to compensate bumped passengers. The challenge is to balance the trade-off between maximizing ticket sales revenue and minimizing compensation costs.

### Key Parameters
- **Seat Capacity (S):** The number of seats available for each fare class.
- **Overbooking Factor (α):** The maximum allowable tickets sold as a percentage of total seat capacity.
- **Show-Up Probability (p):** The likelihood of a passenger showing up for the flight.
- **Compensation Costs (c):** The cost incurred for each bumped passenger, typically calculated as a percentage of the ticket price.
- **Revenue (r):** The ticket price for each fare class.

### Objective
The objective is to maximize the airline's profit, defined as:

Profit = Revenue - Compensation Costs


### Constraints
1. The total number of tickets sold must not exceed the overbooking limit.
2. Each fare class must sell at least a minimum number of tickets.
3. The number of bumped passengers is calculated based on the binomial distribution or linear approximation.

## Repository Structure
```
.
├── input.csv
├── la.py
├── mix_integer.py
├── README.md
├── Simulation.ipynb
├── solution_results.csv
├── solve_use_ga.py
├── structure.txt
└── validation
    ├── binomial.py
    ├── normal_approx.py
    ├── poisson.py
    ├── random_search.py
    └── weighted_sum.py
```


### Key Files
- **`Simulation.ipynb`:** A Jupyter Notebook containing simulations and visualizations for comparing different optimization approaches (e.g., binomial distribution vs. linear approximation).
- **`solve_use_ga.py`:** Implements a Genetic Algorithm (GA) to solve the overbooking optimization problem.
- **`mix_integer.py`:** Uses Mixed Integer Linear Programming (MILP) to solve the problem.
- **`validation/`:** Contains scripts for validating and comparing different statistical models and optimization techniques.

## Methods and Approaches

1. **Binomial Distribution:**
   - Calculates the expected number of bumped passengers using the cumulative distribution function (CDF) of the binomial distribution.
   - Provides a more accurate but computationally intensive solution.

2. **Linear Approximation:**
   - Approximates the number of bumped passengers using a linear model.
   - Faster but less accurate compared to the binomial approach.

3. **Optimization Techniques:**
   - **Mixed Integer Linear Programming (MILP):** Solves the problem using mathematical optimization libraries like `cvxpy`.
   - **Genetic Algorithm (GA):** Uses evolutionary algorithms to find near-optimal solutions.

4. **Visualization:**
   - Plots comparing the results of binomial distribution and linear approximation.
   - Visualizes the impact of overbooking on profit and compensation costs.

## How to Run

1. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run simulations in the Jupyter Notebook:
    ```bash
    jupyter notebook Simulation.ipynb
    ```

## Results
