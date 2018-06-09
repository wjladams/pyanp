
from pyanp.pairwise import Pairwise
from pyanp.prioritizer import Prioritizer, PriorityType
from pyanp.general import islist, unwrap_list
from typing import Union
import pandas as pd
from copy import deepcopy
import numpy as np

class ANPNode:
    '''
    A node in an ANPCLuster, which is in an ANP model.
    '''
    def __init__(self, network, cluster, name:str):
        self.name = name
        self.cluster = cluster
        self.network = network
        self.node_prioritizers = {}

    def is_node_cluster_connection(self, dest_cluster:str)->bool:
        '''
        Is this node connected to a cluster.

        :param dest_cluster: The name of the cluster

        :return:
        '''
        if dest_cluster in self.node_prioritizers:
            return True
        else:
            return False

    def node_connect(self, dest_node)->None:
        ''''
        Make a connection

        :param dest_node: The destination node as a str, int, or ANPNode
        '''
        if islist(dest_node):
            for dn in dest_node:
                self.node_connect(dn)
        else:
            prioritizer = self.get_node_prioritizer(dest_node, create=True)
            prioritizer.add_alt(dest_node, ignore_existing=True)

    def get_node_prioritizer(self, dest_node, create=False)->Prioritizer:
        '''
        Gets the node prioritizer for the other_node

        :param other_node:

        :return: The prioritizer if it exists, or None
        '''
        dest_cluster = self.network._get_node_cluster(dest_node)
        dest_name = dest_cluster.name
        if dest_name not in self.node_prioritizers:
            if create:
                prioritizer = Pairwise()
                self.node_prioritizers[dest_name] = prioritizer
                return prioritizer
            else:
                return None
        else:
            return self.node_prioritizers[dest_name]

    def is_node_node_connection(self, dest_node):
        pri = self.get_node_prioritizer(dest_node)
        if pri is None:
            return False
        elif not pri.is_alt(dest_node):
            return False
        else:
            return True

    def get_unscaled_column(self, username=None)->pd.Series:
        '''
        Returns the column in the unscaled supermatrix for this node.

        :param username: The user/users to do this for

        :return: A pandas series indexed by the node names.
        '''
        nnodes = self.network.nnodes()
        rval = pd.Series(data=[0.0]*nnodes, index=self.network.node_names())
        prioritizer:Prioritizer
        for prioritizer in self.node_prioritizers.values():
            vals = prioritizer.priority(username, PriorityType.NORMALIZE)
            for alt, val in vals.iteritems():
                rval[alt] = val
        return rval


class ANPCluster:
    '''
    A cluster in an ANP object
    '''
    def __init__(self, network, name:str):
        '''
        Creates a new cluster with the given name

        :param name:
        '''
        self.name = name
        self.network = network
        # The list of ANP nodes in this cluster
        self.nodes = {}

    def add_node(self, *nodes):
        '''
        Adds one or more nodes

        :param nodes:
        :return:
        '''
        nodes = unwrap_list(nodes)
        if islist(nodes):
            for node in nodes:
                self.add_node(node)
        else:
            self.nodes[nodes] = ANPNode(self.network, self, nodes)


    def nnodes(self):
        return len(self.nodes)

    def is_node(self, node_name):
        '''
        Does a node by that name exist in this cluster

        :param node_name: The name of the node to look for

        :return:
        '''
        return node_name in self.nodes

    def _get_node(self, node_name):
        if isinstance(node_name, ANPNode):
            return node_name
        elif isinstance(node_name, int):
            if node_name < self.nnodes():
                return get_item(self.nodes, node_name)
            else:
                return None
        elif node_name in self.nodes:
            return self.nodes[node_name]
        else:
            return None

    def node_names(self):
        '''
        :return: List of the names of the nodes in this cluster
        '''
        return list(self.nodes.keys())

    def node_objs(self):
        return self.nodes.values()

def get_item(tbl:dict, index:int):
    if index < 0:
        return None
    elif index >= len(tbl):
        return None
    else:
        count = 0
        for rval in tbl.values():
            if count == index:
                return rval
            count+=1
        #Should never make it here
        raise ValueError("Shouldn't happen in anp.get_item")

class ANPNetwork(Prioritizer):
    '''
    Represents an ANP prioritizer.  Has clusters/nodes, comparisons, etc.
    '''

    def __init__(self):
        '''
        Creates an empty ANP prioritizer
        '''
        self.clusters = {}
        cl = self.add_cluster("Alternatives")
        self.alts_cluster = cl
        self.users=[]

    def add_cluster(self, *args)->ANPCluster:
        '''
        Adds one or more clusters to a network

        :param args: Can be either a single string, or a list of strings

        :return: ANPCluster object
        '''
        clusters = unwrap_list(args)
        if islist(clusters):
            rval = []
            for cl in clusters:
                rval.append(self.add_cluster(cl))
            return rval
        else:
            #Adding a single cluster
            cl = ANPCluster(self, clusters)
            self.clusters[clusters] = cl
            return cl

    def nclusters(self)->int:
        '''
        :return: The number of clusters in the network.
        '''
        return len(self.clusters)

    def _get_cluster(self, cluster_info:Union[ANPCluster,str])->ANPCluster:
        '''
        Returns the cluster with given information

        :param cluster_info: Either the name of the cluster object to get
            or the cluster object

        :return: The ANPCluster object
        '''
        if isinstance(cluster_info, ANPCluster):
            return cluster_info
        else:
            return self.clusters[cluster_info]

    def add_node(self, cl, *nodes):
        '''
        Adds nodes to a cluster

        :param cl: The cluster name or object

        :param nodes: The name or names of the nodes

        :return: Nothing
        '''
        cluster = self._get_cluster(cl)
        cluster.add_node(nodes)

    def nnodes(self, cluster=None):
        if cluster is None:
            rval = pd.Series()
            for cname, cluster in self.clusters.items():
                rval[cname] = cluster.nnodes()
            return sum(rval)
        else:
            clobj = self._get_cluster(cluster)
            return clobj.nnodes()

    def add_alt(self, alt_name:str):
        self.add_node(self.alts_cluster, alt_name)

    def is_user(self, uname)->bool:
        '''
        Checks if a user exists

        :param uname: The name of the user to check for

        :return: bool
        '''
        return uname in self.users

    def is_alt(self, altname)->bool:
        '''
        Checks if an alternative exists

        :param altname: The alterantive name to look for

        :return: bool
        '''
        return self.alts_cluster.is_node(altname)

    def add_user(self, uname):
        '''
        Adds a user to the system

        :param uname: The name of the new user

        :return: Nothing
        '''
        if self.is_user(uname):
            raise ValueError("User by the name "+uname+" already existed")
        self.users.append(uname)

    def nusers(self):
        '''
        :return: The number of users
        '''
        return len(self.users)

    def ussernames(self):
        '''
        :return: List of names of the users
        '''
        return deepcopy(self.users)

    def _get_node(self, node_name)->ANPNode:
        '''
        Gets the ANPNode object of the node with the given name

        :param node_name:  The name of the node to get

        :return: The ANPNode if it exists, or None
        '''
        if isinstance(node_name, ANPNode):
            return node_name
        elif isinstance(node_name, int):
            #Reference by integer
            node_pos = node_name
            node_count = 0
            for cluster in self.clusters.values():
                rel_pos = node_pos - node_count
                if rel_pos < cluster.nnodes():
                    return cluster._get_node(rel_pos)
            #If we make it here, we were out of bounds
            return None
        #Okay handle string node name
        cluster: ANPCluster
        for cname, cluster in self.clusters.items():
            rval = cluster._get_node(node_name)
            if rval is not None:
                return rval

        #Made it here, the node didn't exist
        return None

    def _get_node_cluster(self, node)->ANPCluster:
        '''
        Gets the ANPCluster object a node lives in
        :param node:
        :return:
        '''
        n = self._get_node(node)
        return n.cluster

    def node_connect(self, src_node, dest_node):
        '''
        connects 2 nodes

        :param src_node:

        :param dest_node:

        :return: Nothing
        '''
        src = self._get_node(src_node)
        src.node_connect(dest_node)

    def node_names(self)->list:
        '''
        Returns a list of nodes in this network, organized by cluster

        :return: List of strs of node names
        '''
        rval = []
        cl:ANPCluster
        for cl in self.clusters.values():
            cnodes = cl.node_names()
            for name in cnodes:
                rval.append(name)
        return rval

    def node_objs(self)->list:
        '''
        Returns a list of ANPNodes in this network, organized by cluster

        :return: List of strs of node names
        '''
        rval = []
        cl:ANPCluster
        for cl in self.clusters.values():
            cnodes = cl.node_objs()
            for name in cnodes:
                rval.append(name)
        return rval

    def clusters(self)->list:
        return self.clusters.values()

    def node_connections(self):
        nnodes = self.nnodes()
        nnames = self.node_names()
        rval = np.zeros([nnodes, nnodes])
        src_node:ANPNode
        for src in range(nnodes):
            srcname = nnames[src]
            src_node = self._get_node(srcname)
            for dest in range(nnodes):
                dest_name = nnames[dest]
                if src_node.is_node_node_connection(dest_name):
                    rval[dest,src]=1
        return rval

    def unscaled_supermatrix(self):
        '''
        :return: The unscaled supermatrix
        '''
        nnodes = self.nnodes()
        rval = np.zeros([nnodes, nnodes])
        nodes = self.node_objs()
        col = 0
        node:ANPNode
        for node in nodes:
            rval[:,col] = node.get_unscaled_column()
            col += 1
        return rval
