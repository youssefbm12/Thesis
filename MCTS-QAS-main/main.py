import save_in_file as sif
from save_in_file import check_file_exist
from evaluation_functions import qaoa

eval_func = [qaoa]
N_ITER = 3
BUDGET = [1000, 2000, 5000, 10000, 50000]# 100000, 200000, 300000]#, 400000, 600000]
#BUDGET = [1000]

BF = [False]

ROTYPE = 'classic'
ROSTEPS = [1]
p = {'a': 50, 'd': 10, 's': 20, 'c': 20, 'p': 0}
EPS = None
STOP = False
MAX_DEPTH = 100    # Chosen by the hardware
qubits = {'qaoa': 7}

# Cost plot: convergence via mcts, boxplot of the best via mcts, boxplot after classical optimizer, convergence via classical optimizer
plot = [True, True, True, True]
run = True
apply_gradient_descent = [True, True]

# Run Experiments
for r in ROSTEPS:    
    for f in eval_func:    
        for m in BF:            
            for b in BUDGET:                
                for i in range(N_ITER):
                    if run:
                        print('run')
                        sif.run_and_savepkl(evaluation_function=f, variable_qubits=qubits[f.__name__], ancilla_qubits=0, gate_set='continuous',
                                            rollout_type=ROTYPE, budget=b, branches=m, roll_out_steps=r, iteration=i, max_depth=MAX_DEPTH,
                                            choices=p, epsilon=EPS, stop_deterministic=STOP, verbose=True)



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
