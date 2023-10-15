import numpy as np
import random
from math import exp


class SA:
    def __init__(self, fit_function, num_parameters):
        self.fit_function = fit_function
        self.num_parameters = num_parameters
        self.parameters = [1.5] * num_parameters  # Initialize parameters with default values
        self.thermalization_cycles = 50
        self.alfa = 0.2
        self.delta_tilde = []
        self.delta = []

    def evaluate(self, data, *params):
        loss = 0.0
        func_params = params

        for i in range(len(data)):
            loss += (data[i] - self.fit_function(i, *func_params)) ** 2
        loss *= 1.0 / (len(data))
        loss = np.sqrt(loss)
        return loss

    def fit(self, n, *params):
        func_params = params

        fit_data = [None] * n
        for i in range(n):
            fit_data[i] = self.fit_function(i, *func_params)
        return fit_data

    def report_progress(self, j, i, params, best_eval):
        # Create a string representation of all parameters
        params_str = ', '.join(str(param) for param in params)

        # Report progress
        print(f'>{j}-{i}: Parameters: {params_str}, Evaluation: {best_eval}')

    def simulated_annealing(self, data, n_iterations, step_size):
        # Starting point
        params = self.parameters
        s = step_size
        # Evaluate the initial point
        best_eval = self.evaluate(data, *params)
        # Current working solution
        curr_params, curr_eval = params, best_eval
        scores = []
        fit_data = self.fit(len(data), *params)  # Initialize fit_data here

        # Training Loop
        for j in range(self.thermalization_cycles):
            for i in range(n_iterations):
                # Take a step
                candidate_params = [curr_params[k] + (1 - 2 * random.uniform(0, 1)) * curr_params[k] * s[k]
                                    for k in range(len(params))]

                # Evaluate candidate point
                candidate_eval = self.evaluate(data, *candidate_params)

                # Calculate temperature for current epoch
                t = self.alfa * candidate_eval

                # Check for new best solution
                if candidate_eval < best_eval:
                    # Store new best point
                    params, best_eval = candidate_params, candidate_eval
                    fit_data = self.fit(len(data), *params)
                    # Report progress
                    self.report_progress(j, i, params, best_eval)
                else:
                    # Difference between candidate and current point evaluation
                    diff = candidate_eval - curr_eval
                    # Calculate Metropolis acceptance criterion
                    metropolis = exp(-diff / t)
                    # Check if we should keep the new point
                    if random.random() < metropolis:
                        # Store the new current point
                        curr_params, curr_eval = candidate_params, candidate_eval
                        fit_data = self.fit(len(data), *curr_params)
                        scores.append(curr_eval)
                        # Report progress
                        self.report_progress(j, i, curr_params, curr_eval)
            n_iterations += 10

        return params, best_eval, scores, fit_data
