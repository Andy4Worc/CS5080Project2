import random
import math


def simulate_candidates(n, dist='uniform', **kwargs):
    """
    Simulate the arrival of n candidates with scores drawn from a specified distribution.
    - dist: 'uniform' (default) or 'normal'
    - For normal distribution, use kwargs 'mu' (mean, default 0.5) and 'sigma' (std, default 0.15)
    Returns:
      candidates: list of candidate scores.
      best_index: index of the candidate with the highest score.
    """
    candidates = []
    if dist == 'uniform':
        candidates = [random.uniform(0, 1) for _ in range(n)]
    elif dist == 'normal':
        mu = kwargs.get('mu', 0.5)
        sigma = kwargs.get('sigma', 0.15)
        candidates = [random.gauss(mu, sigma) for _ in range(n)]
    else:
        raise ValueError("Unsupported distribution type. Use 'uniform' or 'normal'.")

    # Determine the true best candidate (highest score)
    best_index = max(range(n), key=lambda i: candidates[i])
    print(f"{best_index}, and: {len(candidates)}, and{max(candidates)}")
    return candidates, best_index


def classical_strategy(candidates, k, estimator_method='max'):
    """
    Implement the classical secretary strategy:
      - Reject the first k candidates.
      - Then select the first candidate whose score is higher than all previously seen.
    Also records the estimator value at each candidate evaluation after the rejection phase.

    estimator_method:
      'max'       : Use the maximum score seen so far as the estimator.
      'avg_top3'  : Use the average of the top 3 scores seen so far as the estimator.

    Returns:
      selected_index: index of the candidate selected (or None if no selection made)
      selected_score: the candidate's score (or None)
      estimator_values: list of estimator values at each step after the rejection phase.
    """
    n = len(candidates)
    estimator_values = []

    # Use the first k candidates as the rejection phase and determine the threshold
    if k > 0:
        best_in_reject = max(candidates[:k])
        # For alternative estimator, keep a list of scores seen so far.
        seen_scores = candidates[:k]
    else:
        best_in_reject = float('-inf')
        seen_scores = []

    # Starting from candidate k, evaluate each candidate
    selected_index = None
    selected_score = None
    for i in range(k, n):
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

        estimator_values.append(current_estimator)

        # Apply the classical selection rule:
        if candidates[i] > best_in_reject:
            selected_index = i
            selected_score = candidates[i]
            break  # select the candidate and stop

    return selected_index, selected_score, estimator_values


def run_trial(n, k, dist='uniform', estimator_method='max', **kwargs):
    """
    Run a single trial of the Secretary Problem.
    Returns:
      candidates: the list of candidate scores.
      best_index: the index of the true best candidate.
      selected_index: the index of the candidate selected by the strategy (or None).
      success: True if the best candidate was selected, else False.
      estimator_values: the sequence of estimator values recorded after the rejection phase.
    """
    candidates, best_index = simulate_candidates(n, dist, **kwargs)
    selected_index, selected_score, estimator_values = classical_strategy(candidates, k, estimator_method)
    success = (selected_index is not None and selected_index == best_index)
    return candidates, best_index, selected_index, success, estimator_values


def run_simulation(num_trials=1000, n=100, k=None, dist='uniform', estimator_method='max', **kwargs):
    """
    Run multiple trials of the Secretary Problem simulation and compute the empirical success rate.
    - If k is not provided, it is set to round(n/e) (~0.37 * n).
    Returns:
      success_rate: proportion of trials where the best candidate was selected.
      details: list of dictionaries with trial results for further analysis.
    """
    if k is None:
        k = round(n / math.e)

    successes = 0
    details = []
    for trial in range(num_trials):
        candidates, best_index, selected_index, success, estimator_values = run_trial(
            n, k, dist, estimator_method, **kwargs)
        if success:
            successes += 1
        trial_detail = {
            "trial": trial,
            "candidates": candidates,
            "best_index": best_index,
            "selected_index": selected_index,
            "success": success,
            "estimator_values": estimator_values
        }
        details.append(trial_detail)
    success_rate = successes / num_trials
    return success_rate, details


# Example usage:
if __name__ == "__main__":
    # Simulation parameters
    num_trials = 1000
    n = 100
    k = round(n / math.e)  # classical threshold approximately n/e
    distribution = 'uniform'  # Change to 'normal' for normal distribution
    estimator_method = 'max'  # Change to 'avg_top3' to use the average of top 3 seen so far

    # Run simulation
    success_rate, simulation_details = run_simulation(
        num_trials=num_trials,
        n=n,
        k=k,
        dist=distribution,
        estimator_method=estimator_method
    )

    print(f"Secretary Problem Simulation ({num_trials} trials, n = {n}, k = {k}):")
    print(f"Empirical success rate (best candidate selected): {success_rate:.3f}")
