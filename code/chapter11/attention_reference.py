"""Single-head CPU attention reference; no dependencies or GPU speed claims.

Rows represent tokens, columns represent features. No dropout or custom masks.
The causal path includes the diagonal, so every row has at least one valid key.
"""

import math


def _validate(q, k, v, causal):
    if not q or not k or len(k) != len(v) or not q[0] or not v[0]:
        raise ValueError("Expected nonempty Q/K/V with matching K/V lengths")
    d, dv = len(q[0]), len(v[0])
    if any(len(row) != d for row in q + k) or any(len(row) != dv for row in v):
        raise ValueError("Inconsistent feature dimensions")
    if any(not math.isfinite(x) for row in q + k + v for x in row):
        raise ValueError("Expected finite inputs")
    if causal and len(q) != len(k):
        raise ValueError("Causal reference requires aligned self-attention")
    return d, dv


def standard_attention(q, k, v, causal=False):
    d, dv = _validate(q, k, v, causal)
    output = []
    for i, query in enumerate(q):
        stop = i + 1 if causal else len(k)
        scores = [sum(a * b for a, b in zip(query, key)) / math.sqrt(d) for key in k[:stop]]
        maximum = max(scores)
        weights = [math.exp(s - maximum) for s in scores]
        normalizer = sum(weights)
        output.append([sum(w * row[c] for w, row in zip(weights, v[:stop])) / normalizer
                       for c in range(dv)])
    return output


def tiled_attention(q, k, v, block_size=64, causal=False):
    """Online softmax with unnormalized output accumulator, normalized at the end.

    This demonstrates FlashAttention's recurrence, not its fused GPU execution.
    """
    if block_size < 1:
        raise ValueError("block_size must be positive")
    d, dv = _validate(q, k, v, causal)
    output = []
    for i, query in enumerate(q):
        maximum, normalizer = -math.inf, 0.0
        accumulator = [0.0] * dv
        stop = i + 1 if causal else len(k)
        for start in range(0, stop, block_size):
            end = min(start + block_size, stop)
            scores = [sum(a * b for a, b in zip(query, key)) / math.sqrt(d)
                      for key in k[start:end]]
            new_maximum = max(maximum, max(scores))
            rescale = math.exp(maximum - new_maximum)
            weights = [math.exp(s - new_maximum) for s in scores]
            normalizer = rescale * normalizer + sum(weights)
            accumulator = [rescale * old + sum(w * row[c] for w, row in zip(weights, v[start:end]))
                           for c, old in enumerate(accumulator)]
            maximum = new_maximum
        output.append([value / normalizer for value in accumulator])
    return output
