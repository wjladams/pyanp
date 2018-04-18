import unittest
import numpy as np
from pyanp.limitmatrix import normalize
from pyanp.exmats import supermatrix_ex, pairwisematrix_ex
from pyanp.priority import pri_eigen
from numpy.testing import *


class MyTestCase(unittest.TestCase):
    def test_bill1(self):
        mat = supermatrix_ex("4x4ex1")
        #print(mat)
        self.assertEqual(len(mat), 4)
        self.assertEqual(mat[0,0], 0.2)



    def test_elena1_1(self):
        mat = supermatrix_ex("SD_BigBurger30x30-1.1")
        print(mat)
        self.assertEqual(len(mat), 30)
        self.assertEqual(mat[0,0], 0.)

if __name__ == '__main__':
    unittest.main()
