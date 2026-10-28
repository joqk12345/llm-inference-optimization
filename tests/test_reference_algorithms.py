import importlib.util
import math
from pathlib import Path
import random
import unittest


ROOT = Path(__file__).resolve().parents[1]


def module(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


attention = module("attention", "code/chapter11/attention_reference.py")
speculation = module("speculation", "code/chapter09/speculative_reference.py")


class AttentionTests(unittest.TestCase):
    def test_tiling_matches_dense_across_shapes_masks_and_blocks(self):
        rng = random.Random(7)
        for n in (1, 3, 7, 17):
            for d, dv in ((1, 1), (3, 5)):
                q = [[rng.uniform(-4, 4) for _ in range(d)] for _ in range(n)]
                k = [[rng.uniform(-4, 4) for _ in range(d)] for _ in range(n)]
                v = [[rng.uniform(-2, 2) for _ in range(dv)] for _ in range(n)]
                for causal in (False, True):
                    expected = attention.standard_attention(q, k, v, causal)
                    for block in (1, 2, 4, 64):
                        with self.subTest(n=n, d=d, causal=causal, block=block):
                            actual = attention.tiled_attention(q, k, v, block, causal)
                            for row, target in zip(actual, expected):
                                for value, reference in zip(row, target):
                                    self.assertAlmostEqual(value, reference, places=11)

    def test_large_scores_and_cross_attention(self):
        q, k, v = [[1.0]], [[1000.0], [1001.0], [-1000.0]], [[2.0], [4.0], [999.0]]
        actual = attention.tiled_attention(q, k, v, 1)[0][0]
        self.assertAlmostEqual(actual, (2 + 4 * math.e) / (1 + math.e), places=12)

    def test_causal_does_not_read_future_values(self):
        output = attention.tiled_attention([[1], [1]], [[1], [1000]], [[3], [999]], 1, True)
        self.assertEqual(output[0], [3])

    def test_reject_invalid_shapes(self):
        with self.assertRaises(ValueError):
            attention.tiled_attention([[1]], [[1, 2]], [[1]])


class SpeculationTests(unittest.TestCase):
    def test_accept_and_residual_masses_reconstruct_target(self):
        for p, q in (([.2, .3, .5], [.6, .3, .1]), ([0, 1], [1, 0]), ([.5, .5], [1, 0])):
            accepted = [min(a, b) for a, b in zip(p, q)]
            rejected = 1 - sum(accepted)
            correction = speculation.residual(p, q)
            for expected, kept, extra in zip(p, accepted, correction):
                self.assertAlmostEqual(kept + rejected * extra, expected)

    def test_all_accepted_includes_bonus_with_updated_prefix(self):
        def conditional(prefix):
            return [0, 1] if len(prefix) % 2 else [1, 0]
        tokens, count = speculation.speculative_step(conditional, conditional, [], 3, random.Random(1))
        self.assertEqual(tokens, [0, 1, 0, 1])
        self.assertEqual(count, 3)

    def test_first_rejection_discards_suffix_and_adds_correction(self):
        tokens, count = speculation.speculative_step(lambda _: [0, 1], lambda _: [1, 0], [], 4, random.Random(1))
        self.assertEqual((tokens, count), ([1], 0))

    def test_later_rejection_preserves_accepted_prefix(self):
        target = lambda prefix: [1, 0] if not prefix else [0, 1]
        tokens, count = speculation.speculative_step(target, lambda _: [1, 0], [], 4, random.Random(2))
        self.assertEqual((tokens, count), ([0, 1], 1))

    def test_seeded_sampling_matches_target_first_token_distribution(self):
        rng = random.Random(31)
        counts = [0, 0, 0]
        for _ in range(20000):
            tokens, _ = speculation.speculative_step(lambda _: [.2, .3, .5], lambda _: [.6, .3, .1], [], 2, rng)
            counts[tokens[0]] += 1
        for count, expected in zip(counts, [.2, .3, .5]):
            self.assertLess(abs(count / 20000 - expected), .015)

    def test_mismatched_vocab_rejected(self):
        with self.assertRaises(ValueError):
            speculation.speculative_step(lambda _: [1], lambda _: [.5, .5], [], 1)


if __name__ == "__main__":
    unittest.main()
