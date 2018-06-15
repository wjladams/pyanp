from unittest import TestCase

from pyanp.anp import ANPNetwork, anp_from_excel
import numpy as np
from  numpy.testing import assert_array_equal, assert_allclose

from pyanp.pairwise import Pairwise


class TestANPNetwork(TestCase):


    def test_crud(self):
        net = ANPNetwork()
        clusters = ["c1", "c2", "c3"]
        cl1, cl2, cl3 = clusters
        net.add_cluster(clusters)
        self.assertEqual(4, net.nclusters())
        nodes = ["n1", "n2", "n3"]
        net.add_node(cl1, nodes)
        self.assertIsNotNone(net.node_obj("n1"))
        self.assertIsNotNone(net.node_obj(2))
        self.assertIsNone(net.node_obj(3))
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
        data_names = net.data_names()
        should = ['n2 vs n1 wrt n1', 'n2 vs n3 wrt n1', 'n1 vs n3 wrt n1', 'n1 vs n2 wrt n2', 'n1 vs n3 wrt n2', 'n2 vs n3 wrt n2', 'n1 vs n2 wrt n3', 'n1 vs n3 wrt n3', 'n2 vs n3 wrt n3']
        assert_array_equal(should, data_names)
        mat= net.node_connection_matrix()
        should = np.array([
            [0., 1., 1., 1.],
            [0., 1., 1., 1.],
            [0., 1., 1., 1.],
            [0., 1., 1., 1.]
            ])
        assert_allclose(should, mat)

    def random_pw_matrix(self, size=3):
        rval = np.identity(size)
        for row in range(size):
            for col in range(row+1, size):
                vote = np.random.randint(1, 10)
                if np.random.random() < 0.5:
                    vote = 1/vote
                rval[row, col]=vote
                rval[col, row]=1/vote
        return rval


    def random_network(self, seed=None, rval=None, maxclusters=10, maxnodes=10,
                       nodeClusterConnectProb=0.75, nusers=2)->ANPNetwork:
        if seed is not None:
            np.random.seed(seed)
        if rval is None:
            rval = ANPNetwork(create_alts_cluster=False)
        usernames = ["User "+str(i) for i in range(nusers)]
        rval.add_user(usernames)
        nclusters = np.random.randint(2, maxclusters+1)
        for clusterp in range(nclusters):
            cluster = "Cluster "+str(clusterp)
            rval.add_cluster(cluster)
            nnodes = np.random.randint(2, maxnodes+1)
            for nodep in range(nnodes):
                node = "Node "+str(clusterp)+"x"+str(nodep)
                rval.add_node(cluster, node)
        rval.set_alts_cluster(0)
        #Now do connections
        clusters = rval.cluster_names()
        nodes = rval.node_names()
        for wrtnode in nodes:
            for cluster in clusters:
                if np.random.random() < nodeClusterConnectProb:
                    #Okay we should do this connection
                    for destnode in rval.node_names(cluster):
                        rval.node_connect(wrtnode, destnode)
                    # Alright we have the connections, let's pairwise for each
                    # user
                    pw:Pairwise = rval.node_prioritizer(wrtnode, cluster)
                    nnodes = rval.nnodes(cluster)
                    for username in usernames:
                        mat = self.random_pw_matrix(size=nnodes)
                        pw.vote_matrix(username, mat)
        return rval

    def test_read_cluster_cmp(self):
        anp = anp_from_excel("data/anp_data_cluster_cmps.xlsx")
        print(anp.nalts())
        print(anp.scaled_supermatrix())

    def test_subnetwork_random(self):
        net = ANPNetwork(create_alts_cluster=False)
        net.add_cluster("goal")
        net.add_node("goal", ["n1", "n2"])
        subnet1 = net.subnet("n1")
        subnet2 = net.subnet("n2")
        np.random.seed(0)
        self.random_network(rval=subnet1)
        np.random.seed(1)
        self.random_network(rval=subnet2)
        self.assertEqual(2, subnet1.nalts())
        self.assertEqual(10, net.nalts())
        print("before priorities")
        pris = net.priority()
        print("after priorities")
        print(pris)
