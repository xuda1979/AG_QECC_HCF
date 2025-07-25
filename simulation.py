"""
Simulation tools for AG-inspired quantum error-correcting codes.

This module provides simple functions to estimate logical block
error probabilities for CSS-type quantum codes under a depolarizing
error model. It includes both an approximate calculation based on
binomial tails and a naive Monte Carlo simulation that samples
random error patterns and checks whether they exceed the code's
correctable threshold.

Note: These tools use very simple decoding models: a code with
parameters [[n,k,d]] is assumed to correct up to t=floor((d-1)/2)
Pauli X and Z errors independently. Correlations between error
types and degeneracy are ignored. For more accurate decoding,
replace the threshold check with a full decoder.
"""

import math
import random
from typing import Tuple


def approximate_block_error(n: int, t: int, p: float) -> float:
    """Approximate block error probability for a CSS code.

    Args:
        n: Number of physical qubits in the code.
        t: Maximum number of correctable errors of each type.
        p: Depolarizing error rate (probability of error per qubit).

    Returns:
        Approximate probability that either bit-flip or phase-flip
        errors exceed the correctable threshold.

    This function assumes X and Z errors occur independently with
    probability p/2. The probability of an uncorrectable X-error pattern
    is the tail of a Binomial(n, p/2) distribution above t. The same
    applies to Z errors. We sum the two contributions to bound the
    overall logical error rate.
    """
    if not (0.0 <= p <= 1.0):
        raise ValueError("p must be a probability between 0 and 1")
    p_xz = p / 2.0
    # compute tail probability for X or Z errors
    tail = 0.0
    for k in range(t + 1, n + 1):
        tail += math.comb(n, k) * (p_xz ** k) * ((1 - p_xz) ** (n - k))
    return 2 * tail


def monte_carlo_block_error(n: int, t: int, p: float, trials: int = 10000) -> float:
    """Estimate block error probability via Monte Carlo sampling.

    Args:
        n: Number of physical qubits.
        t: Maximum number of correctable errors of each type.
        p: Depolarizing error rate per qubit.
        trials: Number of random samples to generate.

    Returns:
        Estimated probability of uncorrectable error patterns.
    """
    if trials <= 0:
        raise ValueError("trials must be positive")
    failures = 0
    # probability of an X or Z error on a single qubit under depolarizing noise
    p_xz = p / 2.0
    for _ in range(trials):
        # sample number of X errors and Z errors independently
        x_errors = sum(random.random() < p_xz for _ in range(n))
        z_errors = sum(random.random() < p_xz for _ in range(n))
        if x_errors > t or z_errors > t:
            failures += 1
    return failures / trials


def demo():
    """Run a simple demonstration comparing several codes at multiple p values."""
    codes = [
        (255, 33, 21),  # AG-inspired code
        (25, 1, 5),     # surface code
        (128, 32, 8),   # BCH-based CSS code
    ]
    p_values = [0.01, 0.03, 0.05]
    trials = 5000
    for n, k, d in codes:
        t = (d - 1) // 2
        print(f"\nCode [[{n},{k},{d}]], t={t}")
        for p in p_values:
            approx = approximate_block_error(n, t, p)
            mc = monte_carlo_block_error(n, t, p, trials)
            print(f"  p={p:.2%}: approx P_L={approx:.3e}, MC estimate={mc:.3e}")


if __name__ == "__main__":
    demo()
