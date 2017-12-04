'''
Created on Dec 3, 2017

@author: wjadams
'''
import unittest
import numpy as np
import numpy.testing as npt
import pyanp.priority as priority

class Test(unittest.TestCase):


    def setUp(self):
        pass


    def tearDown(self):
        pass

    def test_geom_avg(self):
        vals = [2, 3, 4]
        result = priority.geom_avg(vals)
        expected = (2*3*4)**(1/3)
        npt.assert_almost_equal(result, expected, decimal=7)

    def test_pri_eigen(self):
        mat = priority.utmrowlist_to_npmatrix([2, 6, 3])
        eig = priority.pri_eigen(mat)
        npt.assert_almost_equal(eig, [0.6, 0.3, 0.1], 7)

    def test_geom_avg_mat(self):
        mat = np.array([[1, 2], [3, 5]])
        avg_col = priority.geom_avg_mat(mat)
        npt.assert_almost_equal(avg_col, [2**(1/2), 15**(1/2)], decimal=7)
        coeffs=[2, 3]
        avg_col = priority.geom_avg_mat(mat, coeffs)
        npt.assert_almost_equal(avg_col, [(1*2 * 2*3)**(1/2), (3*2 * 5*3)**(1/2)], decimal=7)

    def test_expeigen(self):
        #Try a standard completely consistent matrix
        mat = priority.utmrowlist_to_npmatrix([2, 6, 3])
        result = priority.pri_expeigen(mat)
        npt.assert_almost_equal(result, [0.6, 0.3, 0.1], 7)
        #Try slightly inconsistent 3x3
        mat = priority.utmrowlist_to_npmatrix([2, 4, 3])
        result = priority.pri_expeigen(mat)
        expected = [0.5584245,  0.3196183,  0.1219572]
        npt.assert_almost_equal(result, expected, 7)
        #Now this agrees with expeigen for 3x3, let's try a 4x4
        mat = priority.utmrowlist_to_npmatrix([2, 3, 4, 2, 5, 3])
        result = priority.pri_expeigen(mat)
        expected = [ 0.4508867,  0.3046191,  0.1712999,  0.0731942]
        npt.assert_almost_equal(result, expected, 7)
        #Try a 5x5
        mat = priority.utmrowlist_to_npmatrix([1/2, 1/3, 1/4, 1/5, 2, 3, 4, 2, 5, 3])
        result = priority.pri_expeigen(mat)
        expected = [ 0.0636715,  0.3597739,  0.2851016,  0.1905405,  0.1009125]
        npt.assert_almost_equal(result, expected, 7)
        self.fail("This is matching llsm, which it shouldn't, something is wrong with geom_avg")

    def test_llsm(self):
        #Try a standard completely consistent matrix
        mat = priority.utmrowlist_to_npmatrix([2, 6, 3])
        result = priority.pri_llsm(mat)
        npt.assert_almost_equal(result, [0.6, 0.3, 0.1], 7)
        #Try slightly inconsistent 3x3
        mat = priority.utmrowlist_to_npmatrix([2, 4, 3])
        result = priority.pri_llsm(mat)
        expected = [0.5584245,  0.3196183,  0.1219572]
        npt.assert_almost_equal(result, expected, 7)
        #Now this agrees with expeigen for 3x3, let's try a 4x4
        mat = priority.utmrowlist_to_npmatrix([2, 3, 4, 2, 5, 3])
        result = priority.pri_llsm(mat)
        expected = [ 0.4508867,  0.3046191,  0.1712999,  0.0731942]
        npt.assert_almost_equal(result, expected, 7)
        #Try a 5x5
        mat = priority.utmrowlist_to_npmatrix([1/2, 1/3, 1/4, 1/5, 2, 3, 4, 2, 5, 3])
        result = priority.pri_llsm(mat)
        expected = [ 0.0636715,  0.3597739,  0.2851016,  0.1905405,  0.1009125]
        npt.assert_almost_equal(result, expected, 7)

    def test_harker_fix(self):
        mat = priority.utmrowlist_to_npmatrix([2, 6, 0])
        fixed = priority.harker_fix(mat)
        npt.assert_almost_equal(fixed, [[1, 2, 6], [0.5, 2, 0], [1/6, 0, 2]], decimal=7)
        mat = priority.utmrowlist_to_npmatrix([3, 6, 1/7])
        fixed = priority.harker_fix(mat)
        npt.assert_almost_equal(fixed, [[1, 3, 6], [1/3, 1, 1/7], [1/6, 7, 1]], decimal=7)

    def test_pri_eigen(self):
        mat = priority.utmrowlist_to_npmatrix([2, 6, 3])
        eig = priority.pri_eigen(mat)
        npt.assert_almost_equal(eig, [0.6, 0.3, 0.1], 7)


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.test_pri_eigen']
    unittest.main()