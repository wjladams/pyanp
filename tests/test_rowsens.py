import unittest
import pyanp.rowsens as rs
import numpy as np

class MyTestCase(unittest.TestCase):
    def test_rowsens(self):
        mat = np.array([
            [0.3, 0.2, 0.0],
            [0.1, 0.5, 0.0],
            [0.6, 0.3, 1.0]
        ])
        adj = rs.row_adjust(mat, 0, 0.95, None)
        print(adj)

if __name__ == '__main__':
    unittest.main()
