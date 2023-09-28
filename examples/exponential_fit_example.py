import numpy as np
import sys
import os
import matplotlib.pyplot as plt

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(parent_dir)

from pysa.simulated_annealing import SA


# Define the custom exponential function
def custom_exponential_function(x, A, B):
    return A * np.exp(B * x)


if __name__ == "__main__":

    # Create an instance of the SA class for fitting the custom function
    num_parameters = 2  # Number of parameters for the custom function
    sa = SA(custom_exponential_function, num_parameters)

    # Generate some example data (replace this with your actual data)
    x_data = np.array([0, 1, 2, 3, 4, 5])
    y_data = np.array([0.5, 2.0, 4.5, 8.0, 12.5, 18.0])

    # Initial guess for the parameters
    initial_parameters = [2.5, 1.5]  # A and B

    # Set up SA parameters
    n_iterations = 4000
    # Adjust the step size as needed, including step sizes for to, tc, and t_fin
    step_size = [0.01, 0.01]  # Adjust the step sizes as needed

    # Perform simulated annealing to fit the custom function
    optimized_params, best_eval, scores, optimized_fit = sa.simulated_annealing(y_data, n_iterations, step_size)

    print("Optimized Parameters:", optimized_params)
    print("Best Evaluation:", best_eval)

    # Plot the original data and the fitted curve
    plt.scatter(x_data, y_data, label="Data")
    plt.plot(x_data, optimized_fit, label="Fitted Curve", color='red')
    plt.legend()
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Custom Exponential Function Fitting")
    plt.show()
