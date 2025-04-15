**Class: CS 5080**
**Date: 04/14/2025**
**Author: Anderson (Andy) Worcester**

# Project 2

In this project, I built code, copied code and added to others' code also.
Here are some relevant pieces that can be run:

## Deliverable 1: Quicksort Analysis

My goal and contribution: To see and fully understand the case(s) when Quicksort behaves poorly, compared to the
  average case when it performs well.

- Run `random_arrays_and_plotting.py` to see how I made a plot that clearly shows a bad pivot with various array
  distribution types. On sorted array, the run time is clearly polynomial at `O(n^2)` compared to `O(n log(n))` for the
  other cases.

- Run `deliverable_1_all_parts.py` to see noise affects results (not built by me) and the theoretical comparison results:
  this comparison auto-scales an `n^2` curve and an `n*log(n)` curve and compares them to sorted arrays and nearly sorted
  arrays which have incremental amounts of noise added to them. It does this with both a good and a bad pivot selection.

## Deliverable 2: Secretary Problem / Online Selection Analysis

My goal and main contribution: Here I had two parts, first was to see how different distributions and the sizes of `n`
  would affect the error of difference from the candidate chosen by the classic strategy to the best candidate. Then, I
  to try and maximize the expected value, I make several strategies to compete with the classic `37%` strategy, some of 
  which do better than it.

- Run `secretary_prob_parts_1_and_2.py` to see both of my goals here. First, it iterates through different thresholds
  and finds the errors for the classic strategy (not build by me), then, I have it compare in the same plot different
  errors for different values of `n`.
  - Secondly here, I show the results for comparing different strategies I came up with to try and achieve the best
    expected value/reward from an input, by averaging results. I do this for various input distributions.