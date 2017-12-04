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

    def test_geom_avg_mat(self):
        mat = np.array([[1, 2], [3, 5]])
        avg_col = priority.geom_avg_mat(mat)
        npt.assert_almost_equal(avg_col, [2**(1/2), 15**(1/2)], decimal=7)
        coeffs=[2, 3]
        avg_col = priority.geom_avg_mat(mat, coeffs)
        npt.assert_almost_equal(avg_col, [(1**(2/5) * 2**(3/5)), (3**(2/5) * 5**(3/5))], decimal=7)

    def test_pri_expeigen(self):
        #Try a standard completely consistent matrix
        mat = priority.utmrowlist_to_npmatrix([2, 6, 3])
        result = priority.pri_expeigen(mat)
        npt.assert_almost_equal(result, [0.6, 0.3, 0.1], 7)
        #Try slightly inconsistent 3x3
        mat = priority.utmrowlist_to_npmatrix([2, 4, 3])
        result = priority.pri_expeigen(mat)
        expected = [ 0.5722136,  0.3011777,  0.1266087]
        npt.assert_almost_equal(result, expected, 7)
        #Let's try a 4x4
        mat = priority.utmrowlist_to_npmatrix([2, 3, 4, 2, 5, 3])
        result = priority.pri_expeigen(mat)
        expected = [ 0.4842086,  0.273452 ,  0.1595522,  0.0827873]
        npt.assert_almost_equal(result, expected, 7)
        #Try a 5x5
        mat = priority.utmrowlist_to_npmatrix([1/2, 1/3, 1/4, 1/5, 2, 3, 4, 2, 5, 3])
        result = priority.pri_expeigen(mat)
        expected = [ 0.0940351,  0.4047186,  0.2545017,  0.1592964,  0.0874481]
        npt.assert_almost_equal(result, expected, 7)

    def test_pri_llsm(self):
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
        val = priority.pri_eigen(mat, return_eigenval=True)
        npt.assert_almost_equal(val, 3.0, decimal=7)

    def test_inconsistency_divisor(self):
        sizes = list(range(2, 16))
        incon_divs = [priority.inconsistency_divisor(i)/(i-1) for i in sizes]
        actuals = [1,  .52 , .89, 1.12, 1.25, 1.35, 1.40, 1.45, 1.49, 1.51, 1.54,
                   1.56, 1.57, 1.58]
        npt.assert_almost_equal(incon_divs, actuals, decimal=7)
        npt.assert_equal(priority.inconsistency_divisor(0), 1)
        npt.assert_equal(priority.inconsistency_divisor(1), 1)
        npt.assert_equal(priority.inconsistency_divisor(2), 1)

    def test_incon_std(self):
        mat = priority.utmrowlist_to_npmatrix([2, 6, 3])
        npt.assert_almost_equal(priority.incon_std(mat), 0, decimal=7)
        mat = priority.utmrowlist_to_npmatrix([2, 1/6, 3])
        actual = 1.5430583470604067
        npt.assert_almost_equal(priority.incon_std(mat), actual, decimal=7)

    def test_prerr_euclidratio(self):
        mat = priority.utmrowlist_to_npmatrix([2, 6, 3])
        vec = np.array([1, 1, 1])
        err = priority.prerr_euclidratio(mat, vec)
        expected = ((2-1)**2+(1/2-1)**2+(6-1)**2+(1/6-1)**2+(3-1)**2+(1/3-1)**2)**(1/2)
        npt.assert_almost_equal(err, expected, decimal=7)

    def test_prerr_ratio_avg(self):
        mat = priority.utmrowlist_to_npmatrix([2, 6, 3])
        vec = np.array([1, 1, 1])
        err = priority.prerr_ratio_avg(mat, vec)
        expected = ((2-1)*2+(6-1)*2+(3-1)*2)/6
        npt.assert_almost_equal(err, expected, decimal=7)

    def test_ratio_greater_1(self):
        a=2
        b=5
        npt.assert_almost_equal(priority.ratio_greater_1(a, b), 2.5, decimal=7)
        npt.assert_almost_equal(priority.ratio_greater_1(b, a), 2.5, decimal=7)
        npt.assert_almost_equal(priority.ratio_greater_1(0, b), 1, decimal=7)
        npt.assert_almost_equal(priority.ratio_greater_1(a, 0), 1, decimal=7)
        npt.assert_almost_equal(priority.ratio_greater_1(0, 0), 1, decimal=7)

    def test_prerr_ratio_prod(self):
        mat = priority.utmrowlist_to_npmatrix([2, 6, 3])
        # Try against unit vector
        vec = np.array([1, 1, 1])
        err = priority.prerr_ratio_prod(mat, vec)
        expected = 36**(1/3)
        npt.assert_almost_equal(err, expected, decimal=7)
        # Try against a slightly more complicated vector
        vec = np.array([5, 2, 10])
        err = priority.prerr_ratio_prod(mat, vec)
        expected = 6.0822019955733992
        npt.assert_almost_equal(err, expected, decimal=7)

    def test_ratio_mat(self):
        mat = priority.ratio_mat([6, 3, 1])
        expected = [[1, 2, 6], [1/2, 1, 3], [1/6, 1/3, 1]]
        npt.assert_almost_equal(mat, expected, decimal=7)
        mat = priority.ratio_mat([6, 0, 1])
        expected = [[1, 0, 6], [0, 1, 0], [1/6, 0, 1]]
        npt.assert_almost_equal(mat, expected, decimal=7)

    def test_utmrowlist_to_npmatrix(self):
        mat = priority.utmrowlist_to_npmatrix([2, 6, 3])
        npt.assert_almost_equal(mat, [[1, 2, 6], [1/2, 1, 3], [1/6, 1/3, 1]], decimal=7)
        mat = priority.utmrowlist_to_npmatrix([2, 6, 0])
        npt.assert_almost_equal(mat, [[1, 2, 6], [1/2, 1, 0], [1/6, 0, 1]], decimal=7)


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.test_pri_eigen']
    unittest.main()