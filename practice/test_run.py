"""GPU regression tests: python -m unittest discover -s practice -p 'test_*.py'."""

import ctypes
import unittest

import torch
from run import materialize, verify


class AddChallenge:
    atol = 1e-5
    rtol = 1e-5

    def reference_impl(self, a, b, out, n):
        torch.add(a, b, out=out)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.signature = {
            "a": (ctypes.POINTER(ctypes.c_float), "in"),
            "b": (ctypes.POINTER(ctypes.c_float), "in"),
            "out": (ctypes.POINTER(ctypes.c_float), "out"),
            "n": (ctypes.c_int, "in"),
        }
        self.case = {
            "a": torch.ones(4, device="cuda"),
            "b": -torch.ones(4, device="cuda"),
            "out": torch.zeros(4, device="cuda"),
            "n": 4,
        }

    def test_noop_rejected_even_when_expected_output_is_zero(self):
        with self.assertRaises(AssertionError):
            verify(self.case, AddChallenge(), lambda *args: None, self.signature, torch)

    def test_read_only_input_mutation_rejected(self):
        def solve(a, b, out, n):
            torch.add(a, b, out=out)
            a.zero_()

        with self.assertRaises(AssertionError):
            verify(self.case, AddChallenge(), solve, self.signature, torch)

    def test_inout_preserves_initial_value_and_original_case(self):
        class Accumulate:
            atol = 1e-5
            rtol = 1e-5

            def reference_impl(self, a, b, out, n):
                out.add_(a + b)

        signature = dict(self.signature)
        signature["out"] = (ctypes.POINTER(ctypes.c_float), "inout")
        self.case["out"].fill_(7)

        def solve(a, b, out, n):
            out.add_(a + b)

        verify(self.case, Accumulate(), solve, signature, torch)
        torch.testing.assert_close(self.case["out"], torch.full_like(self.case["out"], 7))

    def test_input_descriptors_are_materialized(self):
        from core.challenge_base import (
            FullTensor,
            OutTensor,
            RandIntTensor,
            RandnTensor,
            RandTensor,
        )

        result = materialize(
            {
                "uniform": RandTensor((8,), -2, 3, "float16"),
                "normal": RandnTensor((8,), 2, 0),
                "integer": RandIntTensor((8,), 2, 3),
                "full": FullTensor((8,), 5),
                "out": OutTensor((8,)),
                "n": 8,
            },
            torch,
            "cuda",
        )
        self.assertEqual(result["uniform"].dtype, torch.float16)
        self.assertTrue(bool(((result["uniform"] >= -2) & (result["uniform"] < 3)).all()))
        self.assertTrue(bool((result["normal"] == 2).all()))
        self.assertTrue(bool((result["integer"] == 2).all()))
        self.assertTrue(bool((result["full"] == 5).all()))
        self.assertEqual(result["out"].shape, (8,))
        self.assertEqual(result["n"], 8)


if __name__ == "__main__":
    unittest.main()
