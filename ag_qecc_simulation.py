# ag_qecc_simulation.py

import math
from typing import Tuple

def block_error(n: int, t: int, p: float) -> float:
    """
    Approximate block error probability for a CSS code correcting up to t errors of each type.

    :param n: Number of physical qubits in the code.
    :param t: Maximum number of correctable errors per error type.
    :param p: Depolarizing error rate (probability of error per qubit).
    :return: Approximate logical block error probability.
    """
    p_xz = p / 2  # assume X and Z errors are equally likely
    prob_x = sum(math.comb(n, k) * (p_xz ** k) * ((1 - p_xz) ** (n - k))
                 for k in range(t + 1, n + 1))
    # Approximate total logical error rate by summing probabilities for X and Z
    return 2 * prob_x

def code_summary(n: int, k: int, d: int, p_values: Tuple[float, ...]):
    t = (d - 1) // 2
    print(f"Code parameters: [[{n},{k},{d}]] (t = {t})")
    for p in p_values:
        p_L = block_error(n, t, p)
        fidelity = 1 - p_L
        print(f"  p = {p:.2%}:   P_L ≈ {p_L:.3e},   F ≈ {fidelity:.3f}")

if __name__ == "__main__":
    # Example: our AG-inspired code, surface code, and BCH-based CSS code
    codes = [
        (255, 33, 21),   # AG-inspired
        (25, 1, 5),      # Surface code
        (128, 32, 8)     # BCH-based CSS
    ]
    p_values = (0.01, 0.03, 0.05)
    for n, k, d in codes:
        code_summary(n, k, d, p_values)
        print()
