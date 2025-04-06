import math
import random
import matplotlib.pyplot as plt
import numpy as np

#CS 5080 Deliverable 2
#Contributions: distribution base and secretary trial's from Faezeh. Various edits, the "estimator" algorithm and additional plotting
# by Andy W.

# --------------------------------
# Candidate Generator
# --------------------------------
def generate_candidates(n, distribution="uniform"):
    """
    Generate n candidate scores based on the specified distribution.
    Supported distributions:
      - "uniform": Uniformly distributed in [0, 1].
      - "normal": Gaussian with mean 0.5 and std 0.15 (clamped to [0, 1]).
      - "exponential": Exponentially distributed (order matters).
      - "beta": Beta distribution with parameters (2, 5).
    """
    if distribution == "uniform":
        return [random.random() for _ in range(n)]
    elif distribution == "normal":
        return [min(max(random.gauss(0.5, 0.15), 0), 1) for _ in range(n)]
    elif distribution == "exponential":
        return [random.expovariate(1) for _ in range(n)]
    elif distribution == "beta":
        return [random.betavariate(2, 5) for _ in range(n)]
    else:
        raise ValueError("Unknown distribution specified.")


# --------------------------------
# Single Trial Simulation with Estimator Evaluation
# --------------------------------
def secretary_trial(n, k, distribution="uniform", estimator_method="max"):
    """
    Run one trial of the secretary simulation with a given rejection threshold k.

    Parameters:
      - n: Total number of candidates.
      - k: Number of candidates to reject in the observation phase.
      - distribution: Distribution to generate candidate scores.

    Process:
      1. Generate candidate scores.
      2. Record the true best candidate and its score.
      3. In the observation phase (first k candidates), compute observed_max.
      4. In the selection phase (from k to n-1), select the first candidate with score > observed_max.
         If none qualify, select the last candidate.

    Also, the estimator here is the observed_max (the best seen in the rejection phase).

    Returns:
      - success (bool): True if the selected candidate is the best overall.
      - estimator_error (float): Difference (true_best - observed_max).
      - selected_value (float): The score of the selected candidate.
      - true_best (float): The maximum candidate score.
    """

    candidates = generate_candidates(n, distribution)
    true_best = max(candidates)
    best_index = candidates.index(true_best)

    # Observation phase: reject first k candidates
    if k > 0:
        observed_max = max(candidates[:k])
        seen_scores = candidates[:k]
    else:
        # With k = 0, no observation; set observed_max to a very low value
        observed_max = -float('inf')
        seen_scores = []

    # Selection phase: select the first candidate exceeding observed_max
    estimator_values = []
    selected_index = None
    classic_completed = False
    for i in range(k, n):

        # First updating simple estimator values:

        # Update the seen scores (only used for alternative estimator)
        seen_scores.append(candidates[i])
        # Update the estimator based on the chosen method
        if estimator_method == 'max':
            current_estimator = max(seen_scores)
        elif estimator_method == 'avg_top3':
            # Consider the top three scores seen so far (or fewer if not available)
            top_scores = sorted(seen_scores, reverse=True)[:3]
            current_estimator = sum(top_scores) / len(top_scores)
        else:
            raise ValueError("Unsupported estimator method. Use 'max' or 'avg_top3'.")

        #print(f"current estimator: {current_estimator}")
        estimator_values.append(current_estimator)
        # next, updating selection for classic algorithm:
        if candidates[i] > observed_max and not classic_completed:
            selected_index = i
            classic_completed = True
    if selected_index is None:
        selected_index = n - 1  # if no candidate qualifies, select the last one

    selected_value = candidates[selected_index]
    success = (selected_index == best_index)
    estimator_error = true_best - observed_max  # basic estimator error

    return success, estimator_error, selected_value, true_best, estimator_values


# --------------------------------
# Run Experiment Over a Range of Thresholds
# --------------------------------
def run_threshold_experiment(n, trials, distribution="uniform", threshold_percentages=None):
    """
    Runs the secretary simulation for a range of rejection thresholds.

    Parameters:
      - n: Number of candidates.
      - trials: Number of trials per threshold.
      - distribution: Distribution to generate candidate scores.
      - threshold_percentages: List of rejection thresholds (as fraction of n)
          e.g., [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

    Returns:
      - thresholds (list): Thresholds in percentage (0 to 100).
      - success_rates (list): Average success rate for each threshold.
      - avg_estimation_errors (list): Average estimation error for each threshold.
    """
    if threshold_percentages is None:
        threshold_percentages = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

    thresholds = []
    success_rates = []
    avg_estimation_errors = []
    averaged_estimator_alg_curves = []
    for thresh_frac in threshold_percentages:
        k = int(n * thresh_frac)
        successes = []
        errors = []
        all_estimator_alg = []
        for _ in range(trials):
            success, est_error, _, _, estimator_values = secretary_trial(n, k, distribution)
            successes.append(success)
            errors.append(est_error)
            all_estimator_alg.append(estimator_values)
        #for an_estimate in all_estimator_alg:
        #    print(len(an_estimate))
        averaged_estimator_alg_curves.append(np.average(all_estimator_alg, axis=0))
        thresholds.append(thresh_frac * 100)
        success_rates.append(np.mean(successes))
        avg_estimation_errors.append(np.mean(errors))
        print(f"Threshold: {thresh_frac * 100:.1f}% (k = {k}), Success Rate: {np.mean(successes):.4f}, Estimation Error: {np.mean(errors):.4f}")

    return thresholds, success_rates, avg_estimation_errors, averaged_estimator_alg_curves


# --------------------------------
# Plotting Results
# --------------------------------
def plot_results(thresholds, success_rates, avg_estimation_errors, averaged_estimator_alg_curves, threshold_percentages, n, distribution):
    plt.figure(figsize=(14, 5))

    # Plot success rate vs. threshold
    plt.subplot(1, 2, 1)
    plt.plot(thresholds, success_rates, marker='o', linestyle='-')
    plt.xlabel("Rejection Threshold (% of n)")
    plt.ylabel("Success Rate")
    plt.title(f'Success Rate vs. Rejection Threshold for distribution: "{distribution}"')
    plt.grid(True)

    # Plot estimation error vs. threshold
    plt.subplot(1, 2, 2)
    plt.plot(thresholds, avg_estimation_errors, marker='o', linestyle='-', color='red')
    plt.xlabel("Rejection Threshold (% of n)")
    plt.ylabel("Average Estimation Error")
    plt.title(f'Error vs. Rejection Threshold for distribution: "{distribution}"')
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    main_x = np.arange(1,n + 1)
    plt.figure()
    for ix, curve in enumerate(averaged_estimator_alg_curves):
        plt.plot(main_x[len(main_x) - len(curve):], curve, label=f"Threshold curve for: {threshold_percentages[ix] * 100}%")
    plt.grid(True)
    plt.title(f'simple "max" estimations for distribution: "{distribution}"')
    plt.xlabel(f"Sequence's sample indicies (n = {n})")
    plt.ylabel("Average Estimator alg value byeond threadhold.")
    plt.legend()
    plt.show()

def final_error_plot(all_thresholds, all_secretary_errors, all_n, distribution):
    for ix, curve in enumerate(all_secretary_errors):
        plt.plot(all_thresholds[ix], curve, marker='o', label=f"Threshold curve for n = {all_n[ix]}")
    plt.xlabel("Rejection Threshold (% of n)")
    plt.ylabel("Average Estimation Error")
    plt.title(f'Error vs. Rejection TH accross "n" for distribution: "{distribution}"')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

# --------------------------------
# Main Execution
# --------------------------------
if __name__ == "__main__":
    all_n = [10, 100, 1000]  # Number of candidates
    trials = 500  # Number of trials per threshold
    distribution = "uniform"  # Change this if you want to test other distributions
    # Define rejection thresholds as percentages (fraction of n)
    threshold_percentages = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    all_secretary_errors = []
    all_thresholds = []
    for n in all_n:
        print(f"Trials for samples (n) = {n}:")
        # Run experiment for a range of thresholds
        thresholds, success_rates, avg_estimation_errors, averaged_estimator_alg_curves = run_threshold_experiment(n, trials, distribution,
                                                                                    threshold_percentages)
        # Plot the results
        plot_results(thresholds, success_rates, avg_estimation_errors, averaged_estimator_alg_curves, threshold_percentages, n, distribution)

        all_secretary_errors.append(avg_estimation_errors)
        all_thresholds.append(thresholds)
        print()
    final_error_plot(all_thresholds, all_secretary_errors, all_n, distribution)

    # Find the threshold that gives the maximum success rate
    best_index = np.argmax(success_rates)
    best_threshold = thresholds[best_index]
    best_success_rate = success_rates[best_index]
    print(f"\nMaximum success rate is {best_success_rate:.4f} at a rejection threshold of {best_threshold:.2f}% of n.")