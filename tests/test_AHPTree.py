from unittest import TestCase

from pyanp.ahptree import AHPTree, ahptree_fromdf
from numpy.testing import assert_array_equal, assert_allclose
import pandas as pd

from pyanp.priority import incon_std


class TestAHPTree(TestCase):

    def test_crud(self):
        tree = AHPTree()
        a1, a2, a3 = ("Bill", "Dan", "John")
        tree.add_alt(a1)
        tree.add_alt(a2)
        assert_array_equal(tree.root.alt_names, (a1, a2))
        assert_array_equal(tree.alt_names, (a1, a2))
        self.assertEqual(tree.nalts(), 2)

    def test_synthesize(self):
        tree = AHPTree()
        u1, u2 = ("Bill", "Lee")
        n1, n2 = ("Node 1", "Node 2")
        a1, a2 = ("Alt1", "Alt2")
        tree.add_alt(a1)
        tree.add_alt(a2)
        tree.root.add_child(n1)
        tree.root.add_child(n2)
        node1 = tree.get_node(n1)
        node1.set_alt_scores({a1:1, a2:0.5})
        node2 = tree.get_node(n2)
        node2.set_alt_scores({a1:0.25, a2:1.0})
        rval = tree.priority()
        print(rval)

    def test_read(self):
        fname = "AHPTreeData.xlsx"
        tree = ahptree_fromdf(fname)
        nodes = ['Goal', 'A1', 'A2', 'A3', 'A', 'B', 'B1', 'B2', 'C', 'C1', 'C2']
        alts = ['Alt1', "Alt2", "Alt3"]
        self.assertEqual(set(nodes), set(tree.nodenames()))
        self.assertEqual(alts, tree.alt_names)
        info = tree.priority()
        #print(info)
        pris = [0.356952, 0.645187, 0.533711]
        assert_allclose(info, pris, rtol=1e-5)
        info = tree.priority(username="Bill")
        pris = [0.400781, 0.604409, 0.4501 ]
        assert_allclose(info, pris, rtol=1e-5)
        gl = tree.global_priority()
        pris = [1.      , 0.241823, 0.061971, 0.078277, 0.101575, 0.3214,
                0.154726, 0.166674, 0.436777, 0.116491, 0.320286]
        assert_allclose(gl, pris, rtol=1e-5)
        gl = tree.global_priority(username="Bill")
        pris = [1.      , 0.19167 , 0.011303, 0.037579, 0.142788, 0.054852,
                0.009142, 0.04571 , 0.753478, 0.251159, 0.502318]
        assert_allclose(gl, pris, atol=1e-5)
        #Trying inconsisentcy
        username = 'Bill'
        wrt = 'Goal'
        ic = tree.incon_std(username, wrt)
        self.assertAlmostEqual(ic, 0.698287, places=5)
        self.assertAlmostEqual(tree.incon_std(username, 'B'), 0, places=10)
        tree.global_priority_table()
        print(tree.incon_std_series(username))
        print(tree.incond_std_table())
        mat = tree.node_pwmatrix('Bill', 'A')
        print(mat)
        print(incon_std(mat))