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

    def test_limit_sinks(self):
        mat = np.array([
            [0.3, 0.2, 0.4],
            [0.1, 0.5, 0.5],
            [0.6, 0.3, 0.1]
        ])
        result = lm.limit_sinks(mat)
        expected = np.array([[ 0.29411765,  0.29411765,  0.29411765],
                             [ 0.38235294,  0.38235294,  0.38235294],
                             [ 0.32352941,  0.32352941,  0.32352941]])
        np.testing.assert_allclose(result, expected)
        # Actual matrix with hierarchy part
        mat = np.array([
            [0.3, 0.2, 0.0],
            [0.1, 0.5, 0.0],
            [0.6, 0.3, 0.0]
        ])
        result = lm.limit_sinks(mat)
        expected = np.array([[0.296223, 0.296223, 0.      ],
                             [0.404648, 0.404648, 0.      ],
                             [0.299128, 0.299128, 0.      ]])
        np.testing.assert_allclose(result, expected, atol=1e-5)

    def test_limit_newhierarchy(self):
        mat = np.array([
            [0.3, 0.2, 0.4],
            [0.1, 0.5, 0.5],
            [0.6, 0.3, 0.1]
        ])
        result = lm.limit_newhierarchy(mat)
        expected = np.array([[ 0.29411765,  0.29411765,  0.29411765],
                             [ 0.38235294,  0.38235294,  0.38235294],
                             [ 0.32352941,  0.32352941,  0.32352941]])
        np.testing.assert_allclose(result, expected)
        # Actual matrix with hierarchy part
        mat = np.array([
            [0.3, 0.2, 0.0],
            [0.1, 0.5, 0.0],
            [0.6, 0.3, 1.0]
        ])
        result = lm.limit_newhierarchy(mat)
        expected = np.array([[0.0, 0.0, 0.      ],
                             [0.0, 0.0, 0.      ],
                             [1.0, 1.0, 1.      ]])
        np.testing.assert_allclose(result, expected, atol=1e-5)

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

    def test_hiernodes(self):
        mat = np.array([
            [0, 1, 0],
            [1, 0, 0],
            [1, 0, 0]
        ])
        result = lm.hierarchy_nodes(mat)
        np.testing.assert_equal([2], result)
        mat = np.array([
            [1, 0, 0],
            [1, 0, 0],
            [1, 1, 0]
        ])
        result = lm.hierarchy_nodes(mat)
        np.testing.assert_equal([1,2], result)


    def test_two_two_breakdown(self):
        mat = np.array([
            [0, 1, 0],
            [1, 0, 0],
            [1, 0, 0]
        ])
        result = lm.two_two_breakdown(mat, [0])
        np.testing.assert_equal((np.array([[0.]]), np.array([[1., 0.]]), np.array([[1.],[1.]]), np.array([[0., 0.],[0., 0.]])),
                                 result)

        result = lm.two_two_breakdown(mat, [1])
        np.testing.assert_equal((np.array([[0.]]), np.array([[1., 0.]]), np.array([[1.],[0.]]), np.array([[0., 0.],[1., 0.]])),
                                 result)

    def test_normalize(self):
        mat = np.array([
            [1., 1, 0.],
            [1, 0, 1],
            [1, 1, 3]
        ])
        lm.normalize(mat, inplace=True)
        expected = np.array([
            [1/3, 1/2, 0],
            [1/3, 0, 1/4],
            [1/3, 1/2, 3/4]
        ])
        np.testing.assert_almost_equal(mat, expected)

if __name__ == '__main__':
    unittest.main()
