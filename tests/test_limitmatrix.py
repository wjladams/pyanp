import unittest
import numpy as np
import pyanp.limitmatrix as lm

class MyTestCase(unittest.TestCase):
    def test_calculus(self):
        mat = np.array([
            [0.3, 0.2, 0.4],
            [0.1, 0.5, 0.5],
            [0.6, 0.3, 0.1]
        ])
        result = lm.calculus(mat)
        expected = np.array([[ 0.29411765,  0.29411765,  0.29411765],
                             [ 0.38235294,  0.38235294,  0.38235294],
                             [ 0.32352941,  0.32352941,  0.32352941]])
        np.testing.assert_allclose(result, expected)

    def test_hier(self):
        mat = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0]
        ])
        result = lm.hiearhcy_formula(mat)
        expected = np.array([
            [0, 0, 0],
            [0.5, 0, 0],
            [0.5, 1, 0]
        ])
        np.testing.assert_allclose(result, expected)

if __name__ == '__main__':
    unittest.main()
