import sys
import unittest

import numpy as np

sys.path.append('./')

from source.matrix_operations import khatri_rao_row

class TestFeatures(unittest.TestCase):
    def test_khatri_rao_row(self):
        # prepare the data:
        a = np.arange(1, 5).reshape(2, 2)
        b = np.arange(1, 7).reshape(2, 3)

        expected = np.array(
            [
                [ 1,  2,  3,  2,  4,  6],
                [12, 15, 18, 16, 20, 24],
            ]
        )
        actual = khatri_rao_row(a, b)
        self.assertTrue(np.allclose(actual, expected))
