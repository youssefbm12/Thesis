import matplotlib.pyplot as plt

def plot_bar(names, values):
    # Check if the lengths of names and values are the same
    if len(names) != len(values):
        raise ValueError("Number of names must be equal to number of values.")

    # Create a bar plot
    plt.figure(figsize=(10, 6))  # Adjust figure size if needed
    plt.bar(names, values)
    plt.xlabel('Budget')
    plt.ylabel('Values')
    plt.title('Average Of Best Values')
    plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better visibility
    plt.tight_layout()  # Adjust layout to prevent label overlapping
    plt.show()

# Example usage:
bar_names = ['100', '200', '1000', '2000', '5000', '10 000', '50 000', '100 000', '200 000', '300 000']  # Convert to strings
bar_values = [3.65, 4.3, 5.5, 4.3, 5, 4.9, 5.3, 5.35, 6.1, 6.3] # Best
#bar_values = [3.53, 3.86, 4.8, 4.3, 4.6, 4.53, 4.8, 5.03, 5.2, 5.6] # Average Of Best
#plot_bar(bar_names, bar_values)

import save_in_file as sif
from save_in_file import check_file_exist
from evaluation_functions import qaoa

eval_func = [qaoa]
N_ITER = 3
#BUDGET = [1000, 2000, 5000, 10000, 50000, 100000, 200000, 300000]#, 400000, 600000]

BUDGET = [200000, 300000]

BF = [False]

ROTYPE = 'classic'
ROSTEPS = [1]
p = {'a': 50, 'd': 15, 's': 20, 'c': 15, 'p': 0}
EPS = None
STOP = False
MAX_DEPTH = 20   # Chosen by the hardware
qubits = {'qaoa': 7}

# Cost plot: convergence via mcts, boxplot of the best via mcts, boxplot after classical optimizer, convergence via classical optimizer
plot = [False, False, False, False]
run = True
apply_gradient_descent = [True, True]

# Plots
for r in ROSTEPS:
    for f in eval_func:
        for m in BF:
            for b in BUDGET:
                if apply_gradient_descent[0]:
                    if check_file_exist(evaluation_function=f, budget=b, n_iter=N_ITER, branches=False, epsilon=EPS, roll_out_steps=r, rollout_type=ROTYPE, stop_deterministic=STOP):                    
                        # Add columns of the cost along the mcts path and the gradient descent on the best quantum circuit found
                        sif.add_columns(evaluation_function=f, budget=b, n_iter=N_ITER, branches=False, epsilon=EPS, roll_out_steps=r, rollout_type=ROTYPE, stop_deterministic=STOP, gradient=apply_gradient_descent[1])
                
                if plot[0]:
                    # plot the cost along the mcts path
                    sif.add_columns(evaluation_function=f, budget=b, n_iter=N_ITER, branches=False, epsilon=EPS, roll_out_steps=r, rollout_type=ROTYPE, stop_deterministic=STOP, gradient=apply_gradient_descent[1])
                    sif.plot_cost(evaluation_function=f, branches=m, budget=b, roll_out_steps=r, rollout_type=ROTYPE, n_iter=N_ITER, epsilon=EPS, stop_deterministic=STOP)
            if plot[1]:
                # Boxplot with the results of the best circuits at different budget on n_iter independent run
                sif.boxplot(BUDGET,evaluation_function=f, branches=m, roll_out_steps=r, rollout_type=ROTYPE, epsilon=EPS,
                            n_iter=N_ITER, gradient=False, stop_deterministic=STOP)
            if plot[2]:
                # Boxplot with the results after the fine-tuning at different budget on n_iter independent run
                sif.boxplot(BUDGET,evaluation_function=f, branches=m, roll_out_steps=r, rollout_type=ROTYPE, epsilon=EPS,
                            n_iter=N_ITER, gradient=True, stop_deterministic=STOP)
            if plot[3]:
                # Plot of the gradient descent on the best run for different budgets
                sif.plot_gradient_descent(evaluation_function=f, branches=m, budget=BUDGET, roll_out_steps=r,
                                          rollout_type=ROTYPE, n_iter=N_ITER, epsilon=EPS, stop_deterministic=STOP)
