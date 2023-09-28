import numpy as np
import sys
import os
import matplotlib.pyplot as plt

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(parent_dir)

from pysa.simulated_annealing import SA


# Define the custom mixed function with trigonometric, polynomial, and logarithmic components
def custom_mixed_function(x, A, B, C, D, E, F, G):
    return A * np.sin(B * x + C) + D * x**2 + E * np.log(F * x + G)


if __name__ == "__main__":
    # Create an instance of the SA class for fitting the custom mixed function
    num_parameters = 7  # Number of parameters for the custom function
    sa = SA(custom_mixed_function, num_parameters)

    # Generate some example data (replace this with your actual data)
    x_data = np.linspace(0.1, 10, 100)
    y_data = (1.5 * np.sin(1.2 * x_data + 0.5) + 0.3 * x_data**2 +
              0.2 * np.log(0.5 * x_data + 1.0) + np.random.normal(0, 0.2, len(x_data)))

    # Initial guess for the parameters
    initial_parameters = [1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0]  # A, B, C, D, E, F, G

    # Set up SA parameters
    n_iterations = 1000
    step_size = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]  # Adjust the step sizes as needed

    # Perform simulated annealing to fit the custom mixed function
    optimized_params, best_eval, scores, optimized_fit = sa.simulated_annealing(y_data, n_iterations, step_size)

    print("Optimized Parameters:", optimized_params)
    print("Best Evaluation:", best_eval)

    # Plot the original data and the fitted curve
    plt.scatter(x_data, y_data, label="Data")
    plt.plot(x_data, optimized_fit, label="Fitted Curve", color='red')
    plt.legend()
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Custom Function Fitting")
    plt.show()
