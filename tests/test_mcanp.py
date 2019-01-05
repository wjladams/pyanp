from unittest import TestCase
from pyanp.mcanp import *
from pyanp.pairwise import Pairwise
import numpy as np
import pandas as pd
import numpy.testing as npt

class TestMCAnp(TestCase):

    def test_amscale(self):
        """
        Testing our conversion functions between additive and
        multiplicative scales.
        :return:
        """
        # Convert multiplicative 0 to additive should give None
        self.assertIsNone(mscale_ascale(0))
        # Convert multiplicative 1 should be additive 0
        npt.assert_almost_equal(0, mscale_ascale(1))
        # Convert multiplicative 2 should be additive 1
        npt.assert_almost_equal(1, mscale_ascale(2))
        # Convert multiplicative 1/2 should be additive -1
        npt.assert_almost_equal(-1, mscale_ascale(1/2))
        # Convert multiplicative 1/9 should be additive -8
        npt.assert_almost_equal(-8, mscale_ascale(1/9))

        # Move on to additive to multiplicative conversions
        # Convert additive None to multiplicative should be 0
        npt.assert_almost_equal(0, ascale_mscale(None), 0)
        npt.assert_almost_equal(1, ascale_mscale(0), 0)
        npt.assert_almost_equal(2, ascale_mscale(1), 0)
        npt.assert_almost_equal(3, ascale_mscale(2), 0)
        npt.assert_almost_equal(9, ascale_mscale(8), 0)
        npt.assert_almost_equal(1/2, ascale_mscale(-1), 0)
        npt.assert_almost_equal(1/3, ascale_mscale(-2), 0)
        npt.assert_almost_equal(1/9, ascale_mscale(-8), 0)

    def test_createsim(self):
        alts = ['alt'+str(i) for i in range(3)]
        sims =[pd.Series(data=[1+i, 2+i*i, 3+i*i-i], index=alts) for i in range(10)]
        sim = PrioritySim(sims)
        print(sim.df)

    def test_sim_pwmat(self):
        mc = MCAnp()
        pw = np.array([
            [1, 2, 6],
            [1/2, 1, 3],
            [1/4, 1/3, 1]
        ])
        s1 = mc.sim(pw)
        #print(s1)
        err = 0.5
        npt.assert_allclose([0.6, 0.3, 0.1], s1, rtol=err)
        # Next I need to test with a full sim with multiple count
        sim1=mc.sim(pw, count=20)
        self.assertEqual(20, len(sim1.df))
        #print(sim1)

    def test_sim_pw(self):
        pw = np.array([
            [1, 2, 4],
            [1/2, 1, 2],
            [1/4, 1/3, 1]
        ])
        mc = MCAnp()
        pwobj = Pairwise(alts=['alt ' + str(i) for i in range(3)])
        pwobj.vote_matrix(user_name='u1', val=pw)
        s1 = mc.sim(pwobj)
        #print(s1)
        err = 0.5
        npt.assert_allclose([4/7, 2/7, 1/7], s1, rtol=err)
        # Next I need to test with a full sim with multiple count
        sim1=mc.sim(pw, count=20)
        self.assertEqual(20, len(sim1.df))
        print(sim1)
