#!/usr/bin/env python3
"""Compute a one-sided exact Clopper-Pearson image-quality lower bound."""

from __future__ import annotations

import argparse
import math


def _binomial_upper_tail(n: int, k: int, p: float) -> float:
    """P[X >= k] for X~Binomial(n,p), stable when k is above the mode."""

    if k <= 0:
        return 1.0
    if k > n or p <= 0.0:
        return 0.0
    if p >= 1.0:
        return 1.0
    log_term = (
        math.lgamma(n + 1)
        - math.lgamma(k + 1)
        - math.lgamma(n - k + 1)
        + k * math.log(p)
        + (n - k) * math.log1p(-p)
    )
    term = math.exp(log_term)
    total = term
    ratio_p = p / (1.0 - p)
    for i in range(k, n):
        term *= ((n - i) / (i + 1.0)) * ratio_p
        total += term
        if term <= max(total, 1e-300) * 1e-16:
            break
    return min(max(total, 0.0), 1.0)


def one_sided_lower_bound(successes: int, trials: int, confidence: float = 0.95) -> float:
    if trials < 1:
        raise ValueError("trials must be positive")
    if not 0 <= successes <= trials:
        raise ValueError("successes must be in [0, trials]")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0,1)")
    if successes == 0:
        return 0.0
    alpha = 1.0 - confidence
    low = 0.0
    high = successes / trials
    for _ in range(100):
        midpoint = (low + high) / 2.0
        if _binomial_upper_tail(trials, successes, midpoint) < alpha:
            low = midpoint
        else:
            high = midpoint
    return (low + high) / 2.0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Exact one-sided lower confidence bound for image-level annotation correctness"
    )
    parser.add_argument("--audited", type=int, required=True, help="Number of independently audited images")
    parser.add_argument("--incorrect", type=int, required=True, help="Number of incorrect audited images")
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--target", type=float, default=0.95)
    args = parser.parse_args()

    if args.audited < 1:
        parser.error("audited must be positive")
    if not 0 <= args.incorrect <= args.audited:
        parser.error("incorrect must be between 0 and audited")
    if not 0.0 < args.confidence < 1.0:
        parser.error("confidence must be between 0 and 1")
    if not 0.0 < args.target < 1.0:
        parser.error("target must be between 0 and 1")
    correct = args.audited - args.incorrect
    estimate = correct / args.audited
    lower = one_sided_lower_bound(correct, args.audited, confidence=args.confidence)
    accepted = lower > args.target
    print(f"audited_images={args.audited}")
    print(f"correct_images={correct}")
    print(f"incorrect_images={args.incorrect}")
    print(f"observed_correctness={estimate:.8f}")
    print(f"one_sided_{args.confidence:.1%}_exact_lower_bound={lower:.8f}")
    print(f"target={args.target:.8f}")
    print(f"accepted={str(accepted).lower()}")
    return 0 if accepted else 2


if __name__ == "__main__":
    raise SystemExit(main())
