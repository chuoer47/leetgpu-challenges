"""Regression tests for explicit GPU selection overriding a sourced environment."""

import os
import unittest
from unittest.mock import patch

from run import select_gpu


class GpuSelectionTests(unittest.TestCase):
    def test_explicit_gpu_overrides_existing_visibility(self):
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "1"}):
            self.assertEqual(select_gpu(3), "3")

    def test_explicit_zero_is_respected(self):
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "3"}):
            self.assertEqual(select_gpu(0), "0")

    def test_environment_is_preserved_when_gpu_is_omitted(self):
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "GPU-example"}):
            self.assertEqual(select_gpu(None), "GPU-example")

    def test_missing_environment_defaults_to_gpu_one(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(select_gpu(None), "1")


if __name__ == "__main__":
    unittest.main()
