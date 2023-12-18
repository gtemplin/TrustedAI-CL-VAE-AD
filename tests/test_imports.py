#!/usr/bin/env python3

import unittest

class TestImports(unittest.TestCase):

    def test_tensorflow_import(self):
        import tensorflow as tf
        self.assertIsNotNone(tf)

    def test_opencv_import(self):
        import cv2
        self.assertIsNotNone(cv2)

    def test_numpy_import(self):
        import numpy as np
        self.assertIsNotNone(np)

    def test_fuzzy_vae_import(self):
        from src.fuzzy_vae import FuzzyVAE
        self.assertIsNotNone(FuzzyVAE)

    def test_plotly_import(self):
        import plotly
        self.assertIsNotNone(plotly)

    def test_kaleido_import(self):
        import kalaido
        self.assertIsNotNone(kalaido)