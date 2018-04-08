import unittest
import pyanp.rowsens as rs
import numpy as np
from pyanp.limitmatrix import normalize

class MyTestCase(unittest.TestCase):
    def test_rowsens(self):
        mat = np.array([
            [0.3, 0.2, 0.0],
            [0.1, 0.5, 0.0],
            [0.6, 0.3, 1.0]
        ])
        adj = rs.row_adjust(mat, 0, 0.95, None)
        print(adj)

    def test_calc_p0(self):
        mat = np.array([
            [0.2, 0.25, 0.05, 0.18],
            [0.3, 0.3, 0.25, 0.07],
            [0.4, 0.3, 0.5, 0.3],
            [0.1, 0.15, 0.2, 0.45]
        ])
        info = rs.calcp0(mat, 0, cluster_nodes=None, orig=0.2, p0mode=1)
        print(info)
        info = rs.calcp0(mat, 0, cluster_nodes=None, orig=0.2, p0mode=2)
        print(info)
        info = rs.calcp0(mat, 0, cluster_nodes=None, orig=0.2, p0mode=3)
        print(info)

    def test_rowadjust_cluster(self):
        mat = np.array([
            [0.2, 0.25, 0.05, 0.18],
            [0.3, 0.3, 0.25, 0.07],
            [0.4, 0.3, 0.5, 0.3],
            [0.1, 0.15, 0.2, 0.45]
        ])
        mat = normalize(mat)
        #print(mat)
        #Adjust very near 1, so row has most of cluster weight
        adj = rs.row_adjust(mat, 0, 0.99999, [0, 1], False)
        np.testing.assert_almost_equal(adj[0,0], 0.5, decimal=4)
        np.testing.assert_almost_equal(adj[1,0], 0.0, decimal=4)
        np.testing.assert_almost_equal(adj[2,0]/adj[3,0], mat[2,0]/mat[3,0], decimal=12)

        np.testing.assert_almost_equal(adj[0,1], 0.55, decimal=4)
        np.testing.assert_almost_equal(adj[1,1], 0.0, decimal=4)
        np.testing.assert_almost_equal(adj[2,1]/adj[3,1], mat[2,1]/mat[3,1], decimal=12)

        np.testing.assert_almost_equal(adj[0,2], 0.3, decimal=4)
        np.testing.assert_almost_equal(adj[1,2], 0.0, decimal=4)
        np.testing.assert_almost_equal(adj[2,2]/adj[3,2], mat[2,2]/mat[3,2], decimal=12)

        np.testing.assert_almost_equal(adj[0,3], 0.25, decimal=4)
        np.testing.assert_almost_equal(adj[1,3], 0.0, decimal=4)
        np.testing.assert_almost_equal(adj[2,3]/adj[3,3], mat[2,3]/mat[3,3], decimal=12)


        #Now Adjust very near 0, so row has none of cluster weight
        adj = rs.row_adjust(mat, 0, 0.000001, [0, 1], False)
        np.testing.assert_almost_equal(adj[0,0], 0.0, decimal=4)
        np.testing.assert_almost_equal(adj[1,0], 0.5, decimal=4)
        np.testing.assert_almost_equal(adj[2,0]/adj[3,0], mat[2,0]/mat[3,0], decimal=12)

        np.testing.assert_almost_equal(adj[0,1], 0.0, decimal=4)
        np.testing.assert_almost_equal(adj[1,1], 0.55, decimal=4)
        np.testing.assert_almost_equal(adj[2,1]/adj[3,1], mat[2,1]/mat[3,1], decimal=12)

        np.testing.assert_almost_equal(adj[0,2], 0.0, decimal=4)
        np.testing.assert_almost_equal(adj[1,2], 0.3, decimal=4)
        np.testing.assert_almost_equal(adj[2,2]/adj[3,2], mat[2,2]/mat[3,2], decimal=12)

        np.testing.assert_almost_equal(adj[0,3], 0.0, decimal=4)
        np.testing.assert_almost_equal(adj[1,3], 0.25, decimal=4)
        np.testing.assert_almost_equal(adj[2,3]/adj[3,3], mat[2,3]/mat[3,3], decimal=12)

    def test_influence_rank(self):
        mat = np.array([
            [0.2, 0.25, 0.05, 0.18],
            [0.3, 0.3, 0.25, 0.07],
            [0.4, 0.3, 0.5, 0.3],
            [0.1, 0.15, 0.2, 0.45]
        ])
        influence_info = rs.influence_rank(mat, 0)
        np.testing.assert_almost_equal(influence_info, 0.9841769522094728)

if __name__ == '__main__':
    unittest.main()
