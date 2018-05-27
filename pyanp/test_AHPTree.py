from unittest import TestCase

from pyanp.ahptree import AHPTree
from numpy.testing import assert_array_equal

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