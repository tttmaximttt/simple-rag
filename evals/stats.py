"""Bootstrap confidence intervals for the eval metrics. Pure functions, so they can be tested on hand-made cases.

Every metric here is a mean of per-question values: hit@k (1 or 0) for Recall@k, reciprocal rank for MRR. A
bootstrap resamples the questions with replacement many times and reads the spread of the resampled means.

paired_bootstrap compares two configs on the same questions by resampling the per-question differences. With
66 questions an unpaired interval for each config is wide, but most questions are easy or hard for every config,
so the differences vary much less than the scores do and a paired test can see a change the two intervals hide.
"""
import math

import numpy as np

N = 10_000
SEED = 0
LEVEL = 0.95


# Each metric and the per-question value it is the mean of.
METRICS = {"recall@1": "hit@1", "recall@5": "hit@5", "mrr@10": "rr@10"}


def per_question(rank: int | None) -> dict:
    """The values the metrics average for one question whose first expected page is at `rank` (None: missed)."""
    hit = lambda n: 1.0 if rank is not None and rank <= n else 0.0
    return {"hit@1": hit(1), "hit@5": hit(5), "rr@10": 1 / rank if rank is not None and rank <= 10 else 0.0}


def intervals(ranks, **kw) -> dict:
    """{metric: (low, high)} for Recall@1, Recall@5 and MRR@10 over questions with these first-hit ranks."""
    values = [per_question(r) for r in ranks]
    return {m: bootstrap_ci([v[field] for v in values], **kw) for m, field in METRICS.items()}


def _resampled_means(values: np.ndarray, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return values[rng.integers(0, len(values), size=(n, len(values)))].mean(axis=1)


def bootstrap_ci(per_question_values, n: int = N, seed: int = SEED, level: float = LEVEL) -> tuple[float, float]:
    """A percentile bootstrap interval for the mean: (low, high). Both equal the mean when every value is the same,
    or there is one value; (nan, nan) for no values."""
    values = np.asarray(per_question_values, dtype=float)
    if not len(values):
        return float("nan"), float("nan")
    means = _resampled_means(values, n, seed)
    tail = (1 - level) / 2 * 100
    lo, hi = np.percentile(means, [tail, 100 - tail])
    return float(lo), float(hi)


def paired_bootstrap(a, b, n: int = N, seed: int = SEED, level: float = LEVEL) -> dict:
    """B minus A on the same questions, in the same order.

    diff   mean of the per-question differences b[i] - a[i]
    ci     percentile bootstrap interval of that mean, resampling questions (so each keeps its pair)
    p      two-sided p-value for "no difference": of the resampled means, shifted so their centre is 0, the share
           at least as far from 0 as the observed difference, as (count + 1) / (n + 1) so it is never 0. 1.0
           when no question differs.

    When only a few questions differ, and all in one direction, the percentile interval can exclude 0 while p is
    above 0.05: both are approximations there. sign_test is exact; read it, and how many questions moved.
    """
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"paired values need the same questions: {len(a)} vs {len(b)}")
    if not len(a):
        raise ValueError("no questions to compare")
    d = b - a
    observed = float(d.mean())
    means = _resampled_means(d, n, seed)
    tail = (1 - level) / 2 * 100
    lo, hi = np.percentile(means, [tail, 100 - tail])
    if not d.any():
        p = 1.0
    else:
        # Under "no difference" the resampled means centre on 0: shift them there and count the extreme ones.
        null = means - observed
        p = float((np.sum(np.abs(null) >= abs(observed) - 1e-12) + 1) / (n + 1))
    return {"diff": observed, "ci": (float(lo), float(hi)), "p": p}


def sign_test(a, b) -> dict:
    """Exact two-sided sign test on paired values (McNemar's exact test when they are 0 or 1): of the questions
    where A and B differ, how many B wins and loses, and the binomial probability of a split at least that uneven
    if each were a coin flip. Ties are left out; p is 1.0 when no question differs."""
    d = np.asarray(b, dtype=float) - np.asarray(a, dtype=float)
    wins, losses = int((d > 0).sum()), int((d < 0).sum())
    n = wins + losses
    tail = sum(math.comb(n, i) for i in range(min(wins, losses) + 1)) / 2 ** n if n else 0.5
    return {"wins": wins, "losses": losses, "p": min(1.0, 2 * tail)}
