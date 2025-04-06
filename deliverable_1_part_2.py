import sys

sys.setrecursionlimit(10000)  # Increase recursion limit to allow deeper recursions

import random
import time
import matplotlib.pyplot as plt
import numpy as np

from part_1_quicksort_pivots import quicksort

# Code from Faezeh, then Barak added some, then Andy added the theoretical experiment and plot.


# ===============================
# Step 1: Create and perturb arrays
# ===============================
def generate_sorted_array(n):
    """Generates a sorted array of integers from 0 to n-1."""
    return list(range(n))


def perturb_array(arr, percent):
    """
    Perturbs the array by swapping a number of randomly chosen element pairs.
    'percent' is the fraction of the total number of elements to swap.
    """
    arr = arr.copy()
    n = len(arr)
    num_swaps = int(n * percent)
    for _ in range(num_swaps):
        i, j = random.sample(range(n), 2)
        arr[i], arr[j] = arr[j], arr[i]
    return arr


# ===============================
# Step 2: Running Experiments
# ===============================
def run_experiment(n, noise_levels, trials=10):
    results = {"noise": [], "runtime": [], "max_depth": [], "avg_balance": []}

    for noise in noise_levels:
        trial_runtimes = []
        trial_depths = []
        trial_balances = []

        for _ in range(trials):
            sorted_arr = generate_sorted_array(n)
            test_arr = perturb_array(sorted_arr, noise)

            start_time = time.perf_counter()
            _, _, current_max_depth, current_avg_balance = quicksort(
                test_arr, pivot_strategy="first"
            )
            end_time = time.perf_counter()

            trial_runtimes.append(end_time - start_time)
            trial_depths.append(current_max_depth)
            trial_balances.append(current_avg_balance)

        results["noise"].append(noise * 100)
        results["runtime"].append(np.mean(trial_runtimes))
        results["max_depth"].append(np.mean(trial_depths))
        results["avg_balance"].append(np.mean(trial_balances))

        print(
            f"Noise: {noise * 100:.1f}%, Avg. Runtime: {results['runtime'][-1]:.5f}s, "
            f"Avg. Max Depth: {results['max_depth'][-1]:.2f}, "
            f"Avg. Pivot Balance: {results['avg_balance'][-1]:.3f}"
        )

    return results


# ===============================
# Step 3: Running Experiments
# ===============================
def run_theoretical_experiment(n_vals, noise_levels, trials=10):
    run_times_for_each_noise_level = []
    for noise in noise_levels:
        run_times_for_a_noise = []
        for n_val in n_vals:
            trial_runtimes = []
            for _ in range(trials):
                sorted_arr = generate_sorted_array(n_val)
                test_arr = perturb_array(sorted_arr, noise)

                start_time = time.perf_counter()
                _, _, current_max_depth, current_avg_balance = quicksort(
                    test_arr, pivot_strategy="first"
                )
                end_time = time.perf_counter()

                trial_runtimes.append(end_time - start_time)
            run_times_for_a_noise.append(np.mean(trial_runtimes))
        print(f"completing theoretical computation for noise level: {noise}%")
        run_times_for_each_noise_level.append(run_times_for_a_noise)
    return run_times_for_each_noise_level



# ===============================
# Step 4: Plot the Results
# ===============================
def plot_results(results):
    noise = results["noise"]

    plt.figure(figsize=(14, 4))

    plt.subplot(1, 3, 1)
    plt.plot(noise, results["runtime"], marker="o")
    plt.xlabel("Noise Level (%)")
    plt.ylabel("Average Runtime (s)")
    plt.title("Runtime vs. Noise Level")
    plt.grid(True)

    plt.subplot(1, 3, 2)
    plt.plot(noise, results["max_depth"], marker="o", color="green")
    plt.xlabel("Noise Level (%)")
    plt.ylabel("Average Max Recursion Depth")
    plt.title("Recursion Depth vs. Noise Level")
    plt.grid(True)

    plt.subplot(1, 3, 3)
    plt.plot(noise, results["avg_balance"], marker="o", color="red")
    plt.xlabel("Noise Level (%)")
    plt.ylabel("Average Pivot Balance")
    plt.title("Pivot Balance vs. Noise Level")
    plt.grid(True)

    plt.tight_layout()
    plt.show()


# ===============================
# Step 5: Plot the Theoretical comparison results
# ===============================
def plot_theoretical(run_times_for_each_noise_level, noise_levels, n_vals_for_sims, early_n):
    plt.figure()

    all_x = np.arange(1, n_vals_for_sims[-1] + 1)
    n_logn_data = all_x * np.log(all_x)
    n_squared_data = all_x * all_x

    # NOTE: must contain value n=early_n in sims
    early_n_index = n_vals_for_sims.index(early_n)

    #n_log_n and n^2 comparison adjustments:
    n_log_n_adjustment = n_logn_data[early_n] / run_times_for_each_noise_level[-1][early_n_index]
    n_squared_adjustment = n_squared_data[early_n] / run_times_for_each_noise_level[0][early_n_index]

    plt.plot(all_x, n_logn_data / n_log_n_adjustment, label="O(n*log(n))")
    plt.plot(all_x, n_squared_data / n_squared_adjustment, label="O(n^2)")
    plt.xlabel("Samples (n)")
    plt.ylabel("Average Runtime (s)")
    plt.yscale("log")
    plt.title("Nearly-Sorted Runtime to Theoertical Comparison")
    plt.grid(True)
    for ix, a_noise_runtime_curve in enumerate(run_times_for_each_noise_level):
        plt.plot(n_vals_for_sims, a_noise_runtime_curve, marker='o', linestyle="--", label=f"result for noise level {noise_levels[ix]}%")
    plt.legend()
    plt.show()


# ===============================
# Main: Set parameters and run the experiment
# ===============================
if __name__ == "__main__":
    n = 1000  # Size of the array
    noise_levels = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
    trials = 10

    #results = run_experiment(n, noise_levels, trials)
    #plot_results(results)

    # theoretical analysis:

    n_vals = [50, 100, 200, 400, 800, 1600, 3200]
    early_n = n_vals[0]
    run_times_for_each_noise_level = run_theoretical_experiment(n_vals, noise_levels, trials)

    plot_theoretical(run_times_for_each_noise_level, noise_levels, n_vals, early_n)
