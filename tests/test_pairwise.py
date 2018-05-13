import unittest
from pyanp.pairwise import Pairwise, geom_avg_mats, add_place
from numpy.testing import assert_array_equal, assert_allclose
import numpy as np

class MyTestCase(unittest.TestCase):
    def test_crud(self):
        alts = ['alt1', 'alt2', 'alt3']
        a1, a2, a3 = alts
        pw = Pairwise(alts=alts)
        u1 = 'Bill'
        pw.add_user(u1)
        self.assertTrue(pw.is_user(u1))
        mat = pw.matrix(u1)
        assert_array_equal(mat, np.identity(3))
        pw.vote('Bill', a1, a2, 3)
        assert_array_equal(mat, [[1, 3, 0], [1./3, 1, 0], [0, 0, 1]])
        pw.unvote('Bill', a1, a2)
        assert_array_equal(mat, np.identity(3))
        u2 = 'Leanne'
        pw.add_user(u2)
        pw.vote(u2, a1, a2, 5)
        pw.vote(u1, a1, a2, 3)
        group = pw.matrix(None)
        assert_allclose(group, [[1, np.sqrt(15), 0], [1/np.sqrt(15), 1, 0], [0, 0, 1]])
        pw.vote(u1, a2, a3, 2)
        pw.vote(u1, a1, a3, 6)
        pri = pw.priority(u1)
        assert_allclose(pri, [6/9, 2/9, 1/9])

    def test_geom_avg(self):
        m1 = np.array([[1, 2, 3], [4, 5, 6]])
        m2 = np.array([[1, 1/2, 1/3], [4, 5, 3]])
        avg = geom_avg_mats([m1, m2])
        assert_allclose(avg, [[1, 1, 1,], [4, 5, np.sqrt(18)]])

    def test_addalt(self):
        pw = Pairwise()
        a1, a2, a3 = ["alt1", "a2", "a3"]
        u1,u2 = ["Bill", "Leanne"]
        pw.add_alt(a1)
        pw.add_user(u1)
        pw.add_alt(a2)
        m = pw.matrix(u1)
        assert_array_equal(m, np.identity(2))
        pw.vote(u1, a1, a2, 5)
        assert_allclose(m, [[1, 5.0], [1/5, 1]])

    def test_addplace(self):
        m1 = add_place(None)
        m2 = add_place(m1)
        assert_array_equal(m2, np.identity(2))

if __name__ == '__main__':
    unittest.main()
