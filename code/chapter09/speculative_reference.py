"""CPU probability reference, not a parallel inference engine (Python 3.10+)."""

import math
import random


def distribution(values):
    values = list(values)
    if not values or any(not math.isfinite(x) or x < 0 for x in values):
        raise ValueError("Probabilities must be finite and nonnegative")
    total = sum(values)
    if not math.isclose(total, 1.0, abs_tol=1e-12):
        raise ValueError("Expected a normalized distribution")
    return [x / total for x in values]


def sample(probabilities, rng):
    draw = rng.random()
    cumulative = 0.0
    for token, probability in enumerate(probabilities):
        cumulative += probability
        if draw < cumulative:
            return token
    return max(i for i, probability in enumerate(probabilities) if probability > 0)


def residual(p, q):
    if len(p) != len(q):
        raise ValueError("Target and draft vocabularies must align")
    weights = [max(0.0, a - b) for a, b in zip(p, q)]
    total = sum(weights)
    if total <= 0:
        raise ValueError("Identical distributions cannot reach rejection")
    return [weight / total for weight in weights]


def speculative_step(target, draft, prefix, length, rng=None):
    """Return accepted tokens plus one correction/bonus token and acceptance count.

    target/draft(prefix) return normalized distributions over the same vocabulary.
    The real engine batches target verification; this reference calls it serially.
    EOS, logits processors, KV storage and output-length truncation are caller tasks.
    """
    if length < 1:
        raise ValueError("length must be positive")
    rng = rng if rng is not None else random.Random()
    candidates, proposals = [], []
    for _ in range(length):
        q = distribution(draft(list(prefix) + candidates))
        candidates.append(sample(q, rng))
        proposals.append(q)
    verified = []
    for token, q in zip(candidates, proposals):
        p = distribution(target(list(prefix) + verified))
        if len(p) != len(q):
            raise ValueError("Target and draft vocabularies must align")
        if rng.random() < min(1.0, p[token] / q[token]):
            verified.append(token)
        else:
            return verified + [sample(residual(p, q), rng)], len(verified)
    p = distribution(target(list(prefix) + verified))
    if len(p) != len(proposals[-1]):
        raise ValueError("Target and draft vocabularies must align")
    return verified + [sample(p, rng)], len(verified)
