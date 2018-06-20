'''
Group enabled ANPNetwork class and supporting classes.

'''
from pyanp.pairwise import Pairwise
from pyanp.prioritizer import Prioritizer, PriorityType
from pyanp.general import islist, unwrap_list, get_matrix, matrix_as_df
from typing import Union
import pandas as pd
from copy import deepcopy
from pyanp.limitmatrix import normalize, calculus, priority_from_limit
import numpy as np
import re

from pyanp.rating import Rating

class ANPNode:
    '''
    A node inside a cluster, inside a netowrk.  The basic building block of
    an ANP netowrk.

    :param network: An ANPNetwork object that this node lives inside.

    :param cluster: An ANPCluster object that this node lives inside.

    :param name: The name of this node.
    '''
    def __init__(self, network, cluster, name:str):
        self.name = name
        self.cluster = cluster
        self.network = network
        self.node_prioritizers = {}
        self.subnetwork = None
        self.invert = False

    def is_node_cluster_connection(self, dest_cluster:str)->bool:
        '''
        Is this node connected to a cluster.

        :param dest_cluster: The name of the cluster

        :return: True/False
        '''
        if dest_cluster in self.node_prioritizers:
            return True
        else:
            return False

    def node_connect(self, dest_node)->None:
        ''''
        Make a node connection from this node to dest_node

        :param dest_node: The destination node as a str, int, or ANPNode.  It
            can be a list of nodes, and then we will coonect each node from
            this node.  The dest_node should be in any format accepted by
            ANPNetwork._get_node()
        '''
        if islist(dest_node):
            for dn in dest_node:
                self.node_connect(dn)
        else:
            prioritizer = self.get_node_prioritizer(dest_node, create=True)
            prioritizer.add_alt(dest_node, ignore_existing=True)
            #Make sure parent clusters are connected
            src_cluster = self.cluster
            dest_cluster = self.network._get_node_cluster(dest_node)
            src_cluster.cluster_connect(dest_cluster)

    def get_node_prioritizer(self, dest_node, create=False,
                             create_class=Pairwise,  dest_is_cluster=False)->Prioritizer:
        '''
        Gets the node prioritizer for the other_node

        :param dest_node: The node as a int, str, or ANPNode object.

        :return: The prioritizer if it exists, or None
        '''
        if dest_is_cluster:
            dest_cluster = self.network.cluster_obj(dest_node)
            dest_name = dest_cluster.name
        else:
            dest_cluster = self.network._get_node_cluster(dest_node)
            dest_name = dest_cluster.name
        if dest_name not in self.node_prioritizers:
            if create:
                prioritizer = create_class()
                self.node_prioritizers[dest_name] = prioritizer
                return prioritizer
            else:
                return None
        else:
            return self.node_prioritizers[dest_name]

    def is_node_node_connection(self, dest_node)->bool:
        '''
        Checks if there is a node connection from this node to dest_node

        :param dest_node: The node as a int, str, or ANPNode object.

        :return:
        '''
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

        :param username: The user/users to do this for.  Typical Prioritizer
            calculation usage, i.e. None means do for all group average.

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

    def data_names(self, append_to=None):
        '''
        Used when exporting an Excel header for a network, for its data.

        :param append_to: If not None, append header strings to this list.
            Otherwise we create a new list to append to.

        :return: List of strings of comparison name headers.  If append_to is not
            None, we return append_to with the new string headers appended.
        '''
        if append_to is None:
            append_to = []
        pri:Prioritizer
        for pri in self.node_prioritizers.values():
            pri.data_names(append_to, post_pend="wrt "+self.name)
        return append_to

    def set_node_prioritizer_type(self, destNode, prioritizer_class):
        '''
        Sets the node prioritizer type

        :param destNode: An ANPNode object, string, or integer location

        :param prioritizer_class: The new type

        :return: None
        '''
        pri = self.get_node_prioritizer(destNode, create_class=prioritizer_class)
        if not isinstance(pri, prioritizer_class):
            #Wrong type, get alts from this one, and create correct one
            rval = prioritizer_class()
            rval.add_alt(pri.alt_names())
            dest_cluster = self.network._get_node_cluster(destNode)
            dest_name = dest_cluster.name
            self.node_prioritizers[dest_name] = rval
        else:
            pass


class ANPCluster:
    '''
    A cluster in an ANP object

    :param network: The ANPNetowrk object this cluster is in.

    :param name: The name of the cluster to create.
    '''
    def __init__(self, network, name:str):
        self.prioritizer = Pairwise()
        self.name = name
        self.network = network
        # The list of ANP nodes in this cluster
        self.nodes = {}

    def add_node(self, *nodes)->None:
        """
        Adds one or more nodes

        :param nodes: A vararg list of node names to add to this cluster.
            The names should all be strings.

        :return: Nonthing
        """
        nodes = unwrap_list(nodes)
        if islist(nodes):
            for node in nodes:
                if isinstance(node, str):
                    self.add_node(node)
        else:
            self.nodes[nodes] = ANPNode(self.network, self, nodes)

    def nnodes(self)->int:
        """
        :return: The number of nodes in this cluster.
        """
        return len(self.nodes)

    def is_node(self, node_name:str)->bool:
        '''
        Does a node by that name exist in this cluster

        :param node_name: The name of the node to look for

        :return: True/False
        '''
        return node_name in self.nodes

    def node_obj(self, node_name):
        """
        Get a node in this cluster.

        :param node_name: The node as either a string name, integer position, or
            simply the ANPObject, in which case there is nothing to do except
            return it.

        :return: ANPNode object.  If it wasn't found, None is returned.
        """
        if isinstance(node_name, ANPNode):
            return node_name
        else:
            return get_item(self.nodes, node_name)

    def node_names(self)->list:
        '''
        :return: List of the string names of the nodes in this cluster
        '''
        return list(self.nodes.keys())

    def node_objs(self)->list:
        '''
        :return: List of the ANPNode objects in this cluster.
        '''
        return self.nodes.values()

    def cluster_connect(self, dest_cluster)->None:
        """
        Make a cluster->cluster connection from this node to the destination.

        :param dest_cluster: Either the ANPCluster object to connect to, or
            the name of the destination cluster.

        :return:
        """
        if isinstance(dest_cluster, ANPCluster):
            dest_cluster_name = dest_cluster.name
        else:
            dest_cluster_name = dest_cluster
        self.prioritizer.add_alt(dest_cluster_name, ignore_existing=True)

    def set_prioritizer_type(self, prioritizer_class)->None:
        '''
        Sets the cluster prioritizer type

        :param prioritizer_class: The new type

        :return: None
        '''
        pri = self.prioritizer
        if not isinstance(pri, prioritizer_class):
            #Wrong type, get alts from this one, and create correct one
            rval = prioritizer_class()
            rval.add_alt(pri.alt_names())
            self.prioritizer = rval
        else:
            pass

    def data_names(self, append_to=None):
        '''
        Used when exporting an Excel header for a network, for its data.

        :param append_to: If not None, append header strings to this list.
            Otherwise we create a new list to append to.

        :return: List of strings of comparison name headers.  If append_to is not
            None, we return append_to with the new string headers appended.
        '''
        if append_to is None:
            append_to = []
        if self.prioritizer is not None:
            self.prioritizer.data_names(append_to, post_pend="wrt "+self.name)
        return append_to



def get_item(tbl:dict, key):
    """
    Looks up an item in a dictionary by key first, assuming the key is in the
    dictionary.  Otherwise, it checks if the key is an integer, and returns
    the item in that position.

    :param tbl: The dictionary to look in

    :param key: The key, or integer position to get the item of

    :return: The item, or it not found, None
    """
    if key in tbl:
        return tbl[key]
    elif not isinstance(key, int):
        return None
    # We have an integer key by this point
    if key < 0:
        return None
    elif key >= len(tbl):
        return None
    else:
        count = 0
        for rval in tbl.values():
            if count == key:
                return rval
            count+=1
        #Should never make it here
        raise ValueError("Shouldn't happen in anp.get_item")

__CLEAN_SPACES_RE = re.compile('\\s+')

def clean_name(name:str)->str:
    """
    Cleans up a string for usage by:

    1. stripping off begging and ending spaces
    2. All spaces convert to one space
    3. \t and \n are treated like a space

    :param name: The string name to be cleaned

    :return: The cleaned name.
    """
    rval = name.strip()
    return __CLEAN_SPACES_RE.sub(string=rval, repl=' ')

def sum_subnetwork_formula(priorities:pd.Series, dict_of_series:dict):
    """
    A function that takes the weighted sum of values.  Used for synthesis.

    :param priorities: Series whose index are the nodes with subnetworks and
        values are their weights.

    :param dict_of_series: A dictionary whose keys are the same as the keys of
        priorities, i.e. the nodes with subnetworks.  The values are Series
        whose keys are alternative names and values are the synthesized
        alternative scores under that subnetwork.
    :return:
    """
    subpriorities = priorities[dict_of_series.keys()]
    if sum(subpriorities) != 0:
        subpriorities /= sum(subpriorities)
    rval = pd.Series()
    counts = pd.Series(dtype=int)
    for subnet_name, vals in dict_of_series.items():
        priority = subpriorities[subnet_name]
        for alt_name, val in vals.iteritems():
            if alt_name in rval:
                rval[alt_name] += val * priority
                counts[alt_name] += priority
            else:
                rval[alt_name] = val
                counts[alt_name] = priority
    # Now let's calculate the averages
    for alt_name, val in rval.iteritems():
        if counts[alt_name] > 0:
            rval[alt_name] /= counts[alt_name]
    return rval


class ANPNetwork(Prioritizer):
    '''
    Represents an ANP prioritizer.  Has clusters/nodes, comparisons, etc.

    :param create_alts_cluster: If True (which is the default) we start with a
        cluster that is the alternatives cluster.  Otherwise the model starts
        empty.
    '''

    def __init__(self, create_alts_cluster=True):
        self.clusters = {}
        if create_alts_cluster:
            cl = self.add_cluster("Alternatives")
            self.alts_cluster = cl
        self.users=[]
        self.limitcalc = calculus
        self.subnet_formula = sum_subnetwork_formula
        self.default_priority_type = None

    def add_cluster(self, *args)->ANPCluster:
        '''
        Adds one or more clusters to a network

        :param args: Can be either a single string, or a list of strings

        :return: ANPCluster object or list of ANPCluster objects
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

    def cluster_names(self)->list:
        '''
        :return: List of string names of the clusters
        '''
        return list(self.clusters.keys())

    def nclusters(self)->int:
        '''
        :return: The number of clusters in the network.
        '''
        return len(self.clusters)

    def cluster_obj(self, cluster_info:Union[ANPCluster, str])->ANPCluster:
        '''
        Returns the cluster with given information

        :param cluster_info: Either the name of the cluster object to get
            or the cluster object, or its int position

        :return: The ANPCluster object
        '''
        if isinstance(cluster_info, ANPCluster):
            return cluster_info
        else:
            return get_item(self.clusters, cluster_info)

    def add_node(self, cl, *nodes):
        '''
        Adds nodes to a cluster

        :param cl: The cluster name or object

        :param nodes: The name or names of the nodes

        :return: Nothing
        '''
        cluster = self.cluster_obj(cl)
        cluster.add_node(nodes)

    def nnodes(self, cluster=None)->int:
        """
        Returns the number of nodes in the network, or a cluster.

        :param cluster: If None, we return the number of nodes in the network.
            Otherwise this is the integer position, string name, or ANPCluster
            object of the cluster to get the node count within.

        :return: The count.
        """
        if cluster is None:
            rval = pd.Series()
            for cname, cluster in self.clusters.items():
                rval[cname] = cluster.nnodes()
            return sum(rval)
        else:
            clobj = self.cluster_obj(cluster)
            return clobj.nnodes()

    def add_alt(self, alt_name:str):
        """
        Adds an alternative to the model:
        1. Adds the altenrative to alts_cluster if not None
        2. For each node with a subnetwork, we add the alternative to that subnetwork.

        :param alt_name: The name of the alternative to add

        :return: Nothing
        """
        if self.alts_cluster is not None:
            self.add_node(self.alts_cluster, alt_name)

        # We should add this alternative to each subnetwork
        for node in self.node_objs_with_subnet():
            node.subnetwork.add_alt(alt_name)

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

    def add_user(self, uname, ignore_dupe=False):
        '''
        Adds a user to the system

        :param uname: The name of the new user

        :return: Nothing

        :raise ValueError If the user already existed
        '''
        if islist(uname):
            for un in uname:
                self.add_user(un, ignore_dupe=ignore_dupe)
            return
        if self.is_user(uname):
            if not ignore_dupe:
                raise ValueError("User by the name "+uname+" already existed")
            else:
                return
        self.users.append(uname)

    def nusers(self)->int:
        '''
        :return: The number of users
        '''
        return len(self.users)

    def user_names(self)->list:
        '''
        :return: List of names of the users
        '''
        return deepcopy(self.users)

    def node_obj(self, node_name)->ANPNode:
        '''
        Gets the ANPNode object of the node with the given name

        :param node_name:  The name of the node to get, or it's overall integer
            position, or the ANPNode object itself

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
                    return cluster.node_obj(rel_pos)
            #If we make it here, we were out of bounds
            return None
        #Okay handle string node name
        cluster: ANPCluster
        for cname, cluster in self.clusters.items():
            rval = cluster.node_obj(node_name)
            if rval is not None:
                return rval

        #Made it here, the node didn't exist
        return None

    def _get_node_cluster(self, node)->ANPCluster:
        '''
        Gets the ANPCluster object a node lives in

        :param node: The name/integer positions, or ANPNode object itself.  See
            node_obj() method for more details.

        :return: The ANPCluster object this node lives in, or None if it doesn't
            exist.
        '''
        n = self.node_obj(node)
        if n is None:
            # Could not find the node
            return None
        return n.cluster

    def node_connect(self, src_node, dest_node):
        '''
        connects 2 nodes

        :param src_node: Source node as prescribed by node_object() function

        :param dest_node: Destination  node as prescribed by node_object() function

        :return: Nothing
        '''
        src = self.node_obj(src_node)
        src.node_connect(dest_node)


    def node_names(self, cluster=None)->list:
        '''
        Returns a list of nodes in this network, organized by cluster

        :param cluster: If None, we get all nodes in network, else we get nodes
            in that cluster, otherwise format as specified by cluster_obj() function.

        :return: List of strs of node names
        '''
        if cluster is not None:
            cl = self.cluster_obj(cluster)
            return cl.node_names()
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

    def cluster_objs(self)->list:
        """
        :return: List of ANPCluster objects in the network
        """
        return list(self.clusters.values())

    def node_connections(self)->np.ndarray:
        """
        Returns the node conneciton matrix for this network.
        :return: A numpy array of shape [nnode, nnodes] where item [row, col]
            1 means there is a node connection from col -> row, and 0 means
            no connection.
        """
        nnodes = self.nnodes()
        nnames = self.node_names()
        rval = np.zeros([nnodes, nnodes])
        src_node:ANPNode
        for src in range(nnodes):
            srcname = nnames[src]
            src_node = self.node_obj(srcname)
            for dest in range(nnodes):
                dest_name = nnames[dest]
                if src_node.is_node_node_connection(dest_name):
                    rval[dest,src]=1
        return rval

    def unscaled_supermatrix(self, username=None, as_df=False)->np.array:
        '''
        :param username: If None, gets it for all users.  Otherwise gets it for
            the user specified.  It can also be a list of users, in which case
            we combine them, as per the theory.

        :param as_df: If True, returns as a dataframe with index and column
            names as the names of the nodes in the network. Otherwise just
            returns the array.

        :return: The unscaled supermatrix as a numpy.array of shape [nnode, nnodes]
        '''
        nnodes = self.nnodes()
        rval = np.zeros([nnodes, nnodes])
        nodes = self.node_objs()
        col = 0
        node:ANPNode
        for node in nodes:
            rval[:,col] = node.get_unscaled_column(username)
            col += 1
        if not as_df:
            return rval
        else:
            return matrix_as_df(rval, self.node_names())

    def scaled_supermatrix(self, username=None, as_df=False)->np.ndarray:
        '''
        :param username: If None, gets it for all users.  Otherwise gets it for
            the user specified.  It can also be a list of users, in which case
            we combine them, as per the theory.

        :param as_df: If True, returns as a dataframe with index and column
            names as the names of the nodes in the network. Otherwise just
            returns the array.

        :return: The scaled supermatrix
        '''
        rval = self.unscaled_supermatrix(username)
        # Now I need to normalized by cluster weights
        clusters = self.cluster_objs()
        nclusters = len(clusters)
        col = 0
        for col_cp in range(nclusters):
            col_cluster:ANPCluster = clusters[col_cp]
            row_nnodes = col_cluster.nnodes()
            cluster_pris = col_cluster.prioritizer.priority(username, PriorityType.NORMALIZE)
            row_offset = 0
            for col_node in col_cluster.node_objs():
                row=0
                for row_cp in range(nclusters):
                    row_cluster:ANPCluster = clusters[row_cp]
                    row_cluster_name = row_cluster.name
                    if row_cluster_name in cluster_pris:
                        priority = cluster_pris[row_cluster_name]
                    else:
                        priority = 0
                    for row_node in row_cluster.node_objs():
                        rval[row, col] *= priority
                        row += 1
                col += 1
        normalize(rval, inplace=True)
        if not as_df:
            return rval
        else:
            return matrix_as_df(rval, self.node_names())

    def global_priority(self, username=None)->pd.Series:
        '''
        :param username: If None, gets it for all users.  Otherwise gets it for
            the user specified.  It can also be a list of users, in which case
            we combine them, as per the theory.

        :return: The global priorities Series, index by node name
        '''
        lm = self.limit_matrix(username)
        rval = priority_from_limit(lm)
        node_names = self.node_names()
        return pd.Series(data=rval, index=node_names)

    def global_priority_df(self, user_infos=None)->pd.DataFrame:
        '''
        :param user_infos: A list of users to do this for, if None is a part
            of this list, it means group average.  If None, it defaults to
            None plus all users.

        :return: The global priorities dataframe.  Rows are the nodes and
            columns are the users.  The first user/column is the Group Average
        '''
        if user_infos is None:
            user_infos = list(self.user_names())
            user_infos.insert(0, None)
        rval = pd.DataFrame()
        for user in user_infos:
            if user is None:
                uname = "Group Average"
            else:
                uname = user
            rval[uname] = self.global_priority(user)
        return rval

    def limit_matrix(self, username=None, as_df=False):
        '''
        :param username: If None, gets it for all users.  Otherwise gets it for
            the user specified.  It can also be a list of users, in which case
            we combine them, as per the theory.

        :param as_df: If True, returns as a dataframe with index and column
            names as the names of the nodes in the network. Otherwise just
            returns the array.

        :return: The limit supermatrix
        '''
        sm = self.scaled_supermatrix(username)
        rval = self.limitcalc(sm)
        if not as_df:
            return rval
        else:
            return matrix_as_df(rval, self.node_names())

    def alt_names(self)->list:
        '''
        :return: List of alt names in this ANP model
        '''
        if self.has_subnet():
            # We have some v1 subnetworks, we get alternative names by looking
            # there.
            rval = []
            node: ANPNode
            for node in self.node_objs_with_subnet():
                alts = node.subnetwork.alt_names()
                for alt in alts:
                    if alt not in rval:
                        rval.append(alt)
            return rval
        else:
            return self.alts_cluster.node_names()

    def priority(self, username=None, ptype:PriorityType=None)->pd.Series:
        '''
        Synthesize and return the alternative scores

        :param username: If None, gets it for all users.  Otherwise gets it for
            the user specified.  It can also be a list of users, in which case
            we combine them, as per the theory.

        :param ptype: The priority type to use

        :return: A pandas.Series indexed on alt names, values are the score
        '''
        if ptype is None:
            # Use the default priority type for this network
            ptype = self.default_priority_type

        if self.has_subnet():
            # Need to synthesize using subnetworks
            return self.subnet_synthesize(username=username, ptype=ptype)
        else:
            gp = self.global_priority(username)
            alt_names = self.alt_names()
            rval = gp[alt_names]
            if sum(rval) != 0:
                rval /= sum(rval)
            if ptype is not None:
                rval = ptype.apply(rval)
            return rval

    def data_names(self):
        '''
        Returns the column headers needed to fill in the data for this model

        :return: A list of strings that would be usable in excel for parsing
            headers
        '''
        node:ANPNode
        rval = []
        cluster: ANPCluster
        for cluster in self.cluster_objs():
            cluster.data_names(rval)
        for node in self.node_objs():
            node.data_names(rval)
        return rval

    def node_connection_matrix(self, new_mat:np.ndarray=None):
        '''
        Returns the current node conneciton matrix if new_mat is None.
        Otherwise, for each item [row, col] in the matrix with a value of 1
        we connect from node[row] to node[col].

        :param new_mat: The new node connection matrix.  If None, we return
            the current one.

        :return: Current connection matrix.
        '''
        src_node:ANPNode
        nnodes = self.nnodes()
        nodes = self.node_objs()
        node_names = self.node_names()
        if new_mat is not None:
            for src_node_pos in range(nnodes):
                src_node = nodes[src_node_pos]
                for dest_node_pos in range(nnodes):
                    if new_mat[dest_node_pos, src_node_pos] == 1:
                        src_node.node_connect(node_names[dest_node_pos])
        rval = np.zeros([nnodes, nnodes])
        for src_node_pos in range(nnodes):
            src_node = nodes[src_node_pos]
            for dest_node_pos in range(nnodes):
                if src_node.is_node_node_connection(node_names[dest_node_pos]):
                    rval[dest_node_pos, src_node_pos] = 1
        return rval

    def import_pw_series(self, series:pd.Series)->None:
        '''
        Takes in a well titled series of data, and pushes it into the right
        node's prioritizer (or cluster).
        The name should be A vs B wrt C, where A, B, C are node or cluster names.

        :param series: The series of data for each user.  Index is usernames.
            Values are the votes.

        :return: Nothing
        '''
        name = series.name
        name = clean_name(name)
        info = name.split(' wrt ')
        if len(info) < 2:
            # We cannot do anything with this, we need a wrt
            raise ValueError("No wrt in "+name)
        wrt = info[1].strip()
        wrtNode:ANPNode
        wrtNode = self.node_obj(wrt)
        info = info[0].split( ' vs ')
        if len(info) < 2:
            raise ValueError(" vs was not present in "+name)
        row, col = info
        rowNode = self.node_obj(row)
        colNode = self.node_obj(col)
        npri: Pairwise
        if (wrtNode is not None) and (rowNode is not None) and (colNode is not None):
            # Node pairwise
            npri = wrtNode.get_node_prioritizer(rowNode, create=True)
            #print("Node comparison "+name)
            if not isinstance(npri, Pairwise):
                raise ValueError("Node prioritizer was not pairwise")
            npri.vote_series(series, row, col, createUnknownUser=True)
            self.add_user(series.index, ignore_dupe=True)
        else:
            # Try cluster pairwise
            wrtcluster = self.cluster_obj(wrt)
            rowcluster = self.cluster_obj(row)
            colcluster = self.cluster_obj(col)
            if wrtcluster is None:
                raise ValueError("wrt="+wrt+" was not a cluster, and the group was not a node comparison")
            if rowcluster is None:
                raise ValueError("row="+row+" was not a cluster, and the group was not a node comparison")
            if colcluster is None:
                raise ValueError("col="+col+" was not a cluster, and the group was not a node comparison")
            npri = self.cluster_prioritizer(wrtcluster)
            npri.vote_series(series, row, col, createUnknownUser=True)
            self.add_user(series.index, ignore_dupe=True)
            #print("Cluster comparison "+name)

    def set_alts_cluster(self, new_cluster):
        '''
        Sets the new alternatives cluster

        :param new_cluster: Cluster specified as cluster_obj() expects.

        :return: Nothing
        '''
        cl = self.cluster_obj(new_cluster)
        self.alts_cluster = cl

    def import_rating_series(self, series:pd.Series):
        '''
        Takes in a well titled series of data, and pushes it into the right
        node's prioritizer as ratings (or cluster).
        Title should be A wrt B, where A and B are either both node names or
        both column names.

        :param series: The series of data for each user.  Index is usernames.
            Values are the votes.

        :return: Nothing
        '''
        name = series.name
        name = clean_name(name)
        info = name.split(' wrt ')
        if len(info) < 2:
            # We cannot do anything with this, we need a wrt
            raise ValueError("No wrt in "+name)
        wrt = info[1].strip()
        dest = info[0].strip()
        wrtNode:ANPNode
        destNode:ANPNode
        wrtNode = self.node_obj(wrt)
        destNode = self.node_obj(dest)
        npri:Rating
        if (wrtNode is not None) and (destNode is not None):
            # Node ratings
            npri = wrtNode.get_node_prioritizer(destNode, create=True, create_class=Rating)
            if not isinstance(npri, Rating):
                wrtNode.set_node_prioritizer_type(destNode, Rating)
                npri = wrtNode.get_node_prioritizer(destNode, create=True)
            npri.vote_column(votes=series, alt_name=dest, createUnknownUsers=True)
        else:
            # Trying cluster ratings
            wrtcluster = self.cluster_obj(wrt)
            destcluster = self.cluster_obj(dest)
            if wrtcluster is None:
                raise ValueError("Ratings: wrt is not a cluster wrt="+wrt+" and wasn't a node either")
            if destcluster is None:
                raise ValueError("Ratings: dest is not a cluster dest="+dest+" and wasn't a node either")
            npri = wrtcluster.prioritizer
            if not isinstance(npri, Rating):
                wrtcluster.set_prioritizer_type(Rating)
                npri = wrtcluster.prioritizer
            npri.vote_column(votes=series, alt_name=dest, createUnknownUsers=True)

    def node_prioritizer(self, wrtnode=None, cluster=None):
        '''
        Gets the prioritizer for node->cluster connection

        :param wrtnode: The node as understood by node_obj() function.

        :param cluster: Cluster as understood by cluster_obj() function.

        :return: If both wrtnode and cluster are specified, a single node prioritizer
            is returned for that comparison (or None if there was nothing there).
            Otherwise it returns a dictionary indexed by [wrtnode, cluster] and
            whose values are the prioritizers for that (only the non-None ones).
        '''
        if wrtnode is not None and cluster is not None:
            node = self.node_obj(wrtnode)
            cl_obj = self.cluster_obj(cluster)
            cluster_name = cl_obj.name
            return node.get_node_prioritizer(dest_node=cluster_name, dest_is_cluster=True)
        elif wrtnode is not None:
            # Have wrtnode, do not have cluster
            rval = {}
            for cluster in self.cluster_names():
                pri = self.node_prioritizer(wrtnode, cluster)
                if pri is not None:
                    rval[(wrtnode, cluster)] = pri
            return rval
        elif cluster is not None:
            # Have cluster, but not wrtnode
            rval = {}
            for wrtnode in self.node_names():
                pri = self.node_prioritizer(wrtnode, cluster)
                if pri is not None:
                    rval[(wrtnode, cluster)] = pri
            return rval
        else:
            # Both wrtnode and cluster are none, want all
            rval = {}
            for wrtnode in self.node_names():
                for cluster in self.cluster_names():
                    pri = self.node_prioritizer(wrtnode, cluster)
                    if pri is not None:
                        rval[(wrtnode, cluster)] = pri
            return rval


    def subnet(self, wrtnode):
        '''
        Makes wrtnode have a subnetwork if it did not already.

        :param wrtnode: The node to give a subnetwork to, or get the subnetwork
            of.  Node specified as node_obj() function expects.

        :return: The ANPNetwork that is the subnet of this node
        '''
        node = self.node_obj(wrtnode)
        if node.subnetwork is not None:
            return node.subnetwork
        else:
            rval = ANPNetwork(create_alts_cluster=False)
            node.subnetwork = rval
            rval.default_priority_type = PriorityType.IDEALIZE
            return rval

    def node_invert(self, node, value=None):
        '''
        Either sets, or tells if a node is inverted

        :param node: The node to do this on, as expected by node_obj() function

        :param value: If None, we return the boolean about if this node is
            inverted.  Otherwise specifies the new value.

        :return: T/F if value=None, telling if the node is inverted.  Otherwise
            returns nothing.
        '''
        node = self.node_obj(node)
        if value is None:
            return node.invert
        else:
            node.invert = value

    def has_subnet(self)->bool:
        '''
        :return: True/False telling if some node had a subentwork
        '''
        for node in self.node_objs():
            if node.subnetwork is not None:
                return True
        return False

    def subnet_synthesize(self, username=None, ptype:PriorityType=None):
        '''
        Does the standard V1 subnetowrk synthesis.

        :param username: The user/users to synthesize for.  If None, we group
            synthesize across all.  If a single user, we sythesize for that user
            across all.  If it is a list, we synthesize for the group that is that
            list of users.

        :return: Nothing
        '''
        # First we need our global priorities
        pris = self.global_priority(username)
        # Next we need the alternative priorities from each subnetwork
        subnets = {}
        node:ANPNode
        for node in self.node_objs_with_subnet():
            p = node.subnetwork.priority(username, ptype)
            if node.invert:
                p = self.invert_priority(p)
            subnets[node.name]=p
        rval = self.synthesize_combine(pris, subnets)
        if ptype is not None:
            rval = ptype.apply(rval)
        return rval

    def node_objs_with_subnet(self):
        """
        :return: List of ANPNode objects in this network that have v1 subnets
        """
        return [node for node in self.node_objs() if node.subnetwork is not None]

    def invert_priority(self, p):
        """
        Makes a copy of the list like element p, and inverts.  The current
        standard inversion is 1-p.  There could be others implemented later.

        :param p: The list like to invert

        :return: New list-like of same type as p, with inverted priorities
        """
        rval = deepcopy(p)
        for i in range(len(p)):
            rval[i] = 1 - rval[i]
        return rval

    def synthesize_combine(self, priorities:pd.Series, alt_scores:dict):
        """
        Performs the actual sythesis step from anp v1 synthesis.

        :param priorities: Priorities of the subnetworks

        :param alt_scores: Alt scores as dictionary, keys are subnetwork names
            values are Series whose keys are alt names.

        :return: Series whose keys are alt names, and whose values are the
            synthesized scores.
        """
        return self.subnet_formula(priorities, alt_scores)

    def cluster_prioritizer(self, wrtcluster=None):
        """
        Gets the prioritizer for the clusters wrt a given cluster.

        :param wrtcluster: WRT cluster identifier as expected by cluster_obj() function.
            If None, then we return a dictionary indexed by cluster names and values
            are the prioritizers

        :return: THe prioritizer for that cluster, or a dictionary of all cluster
            prioritizers
        """
        if wrtcluster is not None:
            cluster = self.cluster_obj(wrtcluster)
            return cluster.prioritizer
        else:
            rval = {}
            for cluster in self.cluster_objs():
                rval[cluster.name] = cluster.prioritizer
            return rval

    def to_excel(self, fname):
        struct = pd.DataFrame()
        cluster:ANPCluster
        writer = pd.ExcelWriter(fname, engine='openpyxl')
        for cluster in self.cluster_objs():
            cluster_name = cluster.name
            if cluster == self.alts_cluster:
                cluster_name = "*"+str(cluster_name)
            struct[cluster_name] = cluster.node_names()
        struct.to_excel(writer, sheet_name="struct", index=False)
        # Now the node connections
        mat = self.node_connection_matrix()
        pd.DataFrame(mat).to_excel(writer, sheet_name="connection", index=False, header=False)
        # Lastly let's write just the comparison structure
        cmp = self.data_names()
        pd.DataFrame({"":cmp}).to_excel(writer, sheet_name="votes", index=False, header=True)
        writer.save()
        writer.close()

    def cluster_incon_std_df(self, user_infos=None) -> pd.DataFrame:
        """
        :param user_infos: A list of users to do this for, if None is a part
            of this list, it means group average.  If None, it defaults to
            None plus all users.

        :return: DataFrame whose columns are clusters, rows
            are users (as controlled by user_infos params) and the value is
            the inconsistency for the given user on the given comparison.
        """
        if user_infos is None:
            user_infos = list(self.user_names())
            user_infos.insert(0, None)
        rval = pd.DataFrame()
        # We need the name for the group (i.e. None) to be something useful)
        for cluster, pw in self.cluster_prioritizer().items():
            if isinstance(pw, Pairwise):
                incon = [pw.incon_std(user) for user in user_infos]
                rval[cluster] = pd.Series(incon, index=user_infos)
        if None in rval.index:
            rval = rval.rename(
                lambda x: x if x is not None else "Group Average")
        return rval

    def node_incon_std_df(self, user_infos=None)->pd.DataFrame:
        """
        :param user_infos: A list of users to do this for, if None is a part
            of this list, it means group average.  If None, it defaults to
            None plus all users.

        :return: DataFrame whose columns are (node,cluster) pairs, rows
            are users (as controlled by user_infos params) and the value is
            the inconsistency for the given user on the given comparison.
        """
        if user_infos is None:
            user_infos = list(self.user_names())
            user_infos.insert(0, None)
        rval = pd.DataFrame()
        # We need the name for the group (i.e. None) to be something useful)
        for info, pw in self.node_prioritizer().items():
            if isinstance(pw, Pairwise):
                incon = [pw.incon_std(user) for user in user_infos]
                rval[info] = pd.Series(incon, index=user_infos)
        if None in rval.index:
            rval = rval.rename(lambda x: x if x is not None else "Group Average")
        return rval

__PW_COL_REGEX = re.compile('\\s+vs\\s+.+\\s+wrt\\s+')

def is_pw_col_name(col:str)->bool:
    """
    Checks to see if the name matches the naming convention for a pairwise
    comparison, i.e. A vs B wrt C

    :param col: The title of the column to check

    :return: T/F
    """
    if col is None:
        return False
    elif isinstance(col, (float, int)) and np.isnan(col):
        return False
    else:
        return __PW_COL_REGEX.search(col) is not None


__RATING_COL_REGEX = re.compile('\\s+wrt\\s+')

def is_rating_col_name(col:str)->bool:
    """
    Checks to see if the name matches the naming convention for a rating
    column of data, i.e. A wrt B

    :param col: The name of the column
    :return: T/F
    """
    if col is None:
        return False
    elif isinstance(col, (float, int)) and np.isnan(col):
        return False
    elif is_pw_col_name(col):
        return False
    else:
        return __RATING_COL_REGEX.search(col) is not None


def anp_manual_scales_from_excel(anp:ANPNetwork, excel_fname):
    """
    Parses manual rating scales from an Excel file

    :param anp: The model to put the scale values in.

    :param excel_fname: The string file name of the excel file with the data

    :return: Nothing
    """
    xl = pd.ExcelFile(excel_fname)
    if "scales" not in xl.sheet_names:
        # We have no scales, do nothing
        return
    # Scales exist, read in
    df = xl.parse(sheet_name="scales")
    for scale_info in df:
        # See if it has a wrt and whatnot
        pieces = scale_info.split(" wrt ")
        if len(pieces) == 2:
            # Found one
            cluster = pieces[0].strip()
            wrtnode = pieces[1].strip()
            scale_data = {}
            for item in df[scale_info]:
                name, val = str(item).split("=")
                name = name.lower().strip()
                val = float(val)
                scale_data[name]=[val]
            rating:Rating
            rating = anp.node_prioritizer(wrtnode, cluster)
            #print(scale_data)
            rating.set_word_eval(scale_data)
    # We are done!


def anp_from_excel(excel_fname:str)->ANPNetwork:
    """
    Parses an excel file to get an ANPNetwork

    :param excel_fname: The name of the excel file

    :return: The newly created ANPNetwork object
    """
    ## Structure first
    df = pd.read_excel(excel_fname, sheet_name=0)
    anp = ANPNetwork(create_alts_cluster=False)
    for col in df:
        if col.startswith("*"):
            is_alt = True
            cname = col[1:len(col)]
        else:
            is_alt = False
            cname = col
        anp.add_cluster(cname)
        anp.add_node(cname, df[col])
        if is_alt:
            anp.set_alts_cluster(cname)
    ## Now conneciton data
    conn_mat = get_matrix(excel_fname, sheet=1)
    #print(conn_mat)
    #print(conn_mat)
    #print(conn_mat.shape)
    anp.node_connection_matrix(conn_mat)
    ## Now pairwise data
    df = pd.read_excel(excel_fname, sheet_name=2)
    row_names_with_vs = [1.0 if " vs " in name else 0.0 for name in df.index]
    col_names_with_vs = [1.0 if " vs " in name else 0.0 for name in df.columns]
    if len(row_names_with_vs) > 0:
        row_percent = sum(row_names_with_vs) / len(row_names_with_vs)
    else:
        row_percent = 0
    if len(col_names_with_vs) > 0:
        col_percent = sum(col_names_with_vs) / len(col_names_with_vs)
    else:
        col_percent = 0
    if row_percent > col_percent:
        df = df.transpose()
    #display(df)
    for col in df:
        # print(col)
        if is_pw_col_name(col):
            anp.import_pw_series(df[col])
        elif is_rating_col_name(col):
            # print("Rating column "+col)
            anp.import_rating_series(df[col])
        else:
            print("Unknown column "+str(col)+" ignored")
    # Now let's setup manual rating scales
    anp_manual_scales_from_excel(anp, excel_fname)
    return anp

def anp_from_dict(cluster_dict:dict)->ANPNetwork:
    """
    Creates an ANPNetwork from a dictionary whose keys are cluster names
    and whose values are list of node names in that cluster

    :param cluster_dict: Keys are cluster names.  If the cluster name starts
        with *, that is the alternatives cluster (and the asterisk is removed
        from the name).  The values are list of strings that are the names
        of the nodes in that network

    :return: The ANPNetwork with that structure
    """
    rval = ANPNetwork(create_alts_cluster=False)
    for cluster, nodes in cluster_dict.items():
        if cluster.startswith("*"):
            # We need to trim that off
            cluster = cluster[1:len(cluster)]
            rval.add_cluster(cluster)
            rval.set_alts_cluster(cluster)
        else:
            rval.add_cluster(cluster)
        rval.add_node(cluster, nodes)
    return rval