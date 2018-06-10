from unittest import TestCase

from pyanp.anp import ANPNetwork
import numpy as np
from  numpy.testing import assert_array_equal, assert_allclose

class TestANPNetwork(TestCase):


    def test_crud(self):
        net = ANPNetwork()
        clusters = ["c1", "c2", "c3"]
        cl1, cl2, cl3 = clusters
        net.add_cluster(clusters)
        self.assertEqual(4, net.nclusters())
        nodes = ["n1", "n2", "n3"]
        net.add_node(cl1, nodes)
        self.assertIsNotNone(net._get_node("n1"))
        self.assertIsNotNone(net._get_node(2))
        self.assertIsNone(net._get_node(3))
        print(net.nnodes())
        net.add_alt("alt1")
        self.assertTrue(net.is_alt("alt1"))
        net.add_user("Bill")
        self.assertEqual(1, net.nusers())
        net.node_connect("n1", "n2")
        nnames = net.node_names()
        self.assertEqual(nnames, ["alt1", "n1", "n2", "n3"])
        mat = net.node_connections()
        should = np.array([
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 0]
        ])
        assert_allclose(should, mat)
        net.node_connect("n1", ["n1", "n2", "n3", "alt1"])
        net.node_connect("n2", ["n1", "n2", "n3", "alt1"])
        net.node_connect("n3", ["n1", "n2", "n3", "alt1"])
        mat = net.unscaled_supermatrix()
        should = np.matrix([
            [0.,  1., 1., 1.],
            [0., 0.33333333,  0.33333333, 0.33333333],
            [0., 0.33333333, 0.33333333, 0.33333333],
            [0., 0.33333333, 0.33333333 ,0.33333333]
        ])
        assert_allclose(should, mat)
        should = [0.500000, 1./6, 1/6, 1/6]
        gp = net.global_priorities()
        assert_allclose(should, gp)
        pri = net.priority(username=None)
        assert_allclose(pri, [1.0])
