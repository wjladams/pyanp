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


    def test_pri_eigen(self):
        mat = priority.utmrowlist_to_npmatrix([2, 6, 3])
        eig = priority.pri_eigen(mat)
        npt.assert_almost_equal(eig, [0.6, 0.3, 0.1], 7)


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.test_pri_eigen']
    unittest.main()