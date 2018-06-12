
from pyanp.pairwise import Pairwise
from pyanp.prioritizer import Prioritizer, PriorityType
from pyanp.general import islist, unwrap_list, get_matrix
from typing import Union
import pandas as pd
from copy import deepcopy
from pyanp.limitmatrix import normalize, calculus, priority_from_limit
import numpy as np
import re

from pyanp.rating import Rating


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

    def get_node_prioritizer(self, dest_node, create=False, create_class=Pairwise)->Prioritizer:
        '''
        Gets the node prioritizer for the other_node

        :param other_node:

        :return: The prioritizer if it exists, or None
        '''
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

    def data_names(self, append_to=None):
        '''
        :return: String of comparison name headers
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
                if isinstance(node, str):
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

__CLEAN_SPACES_RE = re.compile('\\s+')

def clean_name(name:str)->str:
    '''
    Cleans up a string for usage by:

    1. stripping off begging and ending spaces
    2. All spaces convert to one space
    3. \t and \n are treated like a space

    :param name:
    :return:
    '''
    rval = name.strip()
    return __CLEAN_SPACES_RE.sub(string=rval, repl=' ')


class ANPNetwork(Prioritizer):
    '''
    Represents an ANP prioritizer.  Has clusters/nodes, comparisons, etc.
    '''

    def __init__(self, create_alts_cluster=True):
        '''
        Creates an empty ANP prioritizer
        '''
        self.clusters = {}
        if create_alts_cluster:
            cl = self.add_cluster("Alternatives")
            self.alts_cluster = cl
        self.users=[]
        self.limitcalc = calculus

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
        elif isinstance(cluster_info, int):
            count = 0
            for cl in self.clusters.values():
                if count == cluster_info:
                    return cl
                count+=1
            #Made it here, was too big
            raise ValueError("Cluster int was too big")
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

    def user_names(self):
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

    def unscaled_supermatrix(self, username=None):
        '''
        :return: The unscaled supermatrix
        '''
        nnodes = self.nnodes()
        rval = np.zeros([nnodes, nnodes])
        nodes = self.node_objs()
        col = 0
        node:ANPNode
        for node in nodes:
            rval[:,col] = node.get_unscaled_column(username)
            col += 1
        return rval

    def scaled_supermatrix(self, username=None):
        '''
        :return: The scaled supermatrix
        '''
        rval = self.unscaled_supermatrix(username)
        normalize(rval, inplace=True)
        return rval

    def global_priorities(self, username=None):
        '''

        :param username:
        :return: The global priorities Series, index by node name
        '''
        lm = self.limit_matrix(username)
        rval = priority_from_limit(lm)
        node_names = self.node_names()
        return pd.Series(data=rval, index=node_names)

    def limit_matrix(self, username=None):
        sm = self.scaled_supermatrix(username)
        rval = self.limitcalc(sm)
        return rval

    def alt_names(self):
        '''
        :return: List of alt names in this ANP model
        '''
        return self.alts_cluster.node_names()

    def priority(self, username=None, ptype:PriorityType=None):
        '''
        Synthesize and return the alternative scores

        :param username: The user/users to synthesize for

        :param ptype: The priority type to use

        :return: A pandas.Series indexed on alt names, values are the score
        '''
        gp = self.global_priorities(username)
        alt_names = self.alt_names()
        rval = gp[alt_names]
        if sum(rval) != 0:
            rval /= sum(rval)
        return rval

    def data_names(self):
        '''
        Returns the column headers needed to fill in the data for this model

        :return: A list of strings that would be usable in excel for parsing
            headers
        '''
        node:ANPNode
        rval = []
        for node in self.node_objs():
            node.data_names(rval)
        return rval

    def node_connection_matrix(self, new_mat=None):
        '''
        Returns the current node conneciton matrix if new_mat is None
        otherwise sets it.

        :param new_mat:
        :return:
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
        wrtNode = self._get_node(wrt)
        info = info[0].split( ' vs ')
        if len(info) < 2:
            raise ValueError(" vs was not present in "+name)
        row, col = info
        rowNode = self._get_node(row)
        colNode = self._get_node(col)
        if rowNode.cluster.name != colNode.cluster.name:
            raise ValueError(" comparing nodes not exiting in same cluster")
        npri:Pairwise
        npri = wrtNode.get_node_prioritizer(rowNode, create=True)
        if not isinstance(npri, Pairwise):
            raise ValueError("Node prioritizer was not pairwise")
        npri.vote_series(series, row, col, createUnknownUser=True)

    def set_alts_cluster(self, new_cluster):
        '''
        Sets the new alternatives cluster

        :param new_cluster:
        :return:
        '''
        cl = self._get_cluster(new_cluster)
        self.alts_cluster = cl

    def import_rating_series(self, series:pd.Series):
        '''
        Takes in a well titled series of data, and pushes it into the right
        node's prioritizer as ratings (or cluster).

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
        wrtNode = self._get_node(wrt)
        destNode = self._get_node(dest)
        npri:Rating
        npri = wrtNode.get_node_prioritizer(destNode, create=True, create_class=Rating)
        if not isinstance(npri, Rating):
            wrtNode.set_node_prioritizer_type(destNode, Rating)
            npri = wrtNode.get_node_prioritizer(destNode, create=True)
        npri.vote_column(votes=series, alt_name=dest, createUnknownUsers=True)

    def node_prioritizer(self, wrtnode, cluster):
        node = self._get_node(wrtnode)
        return node.node_prioritizers[cluster]

__PW_COL_REGEX = re.compile('\\s+vs\\s+.+\\s+wrt\\s+')

def is_pw_col_name(col):
    return __PW_COL_REGEX.search(col) is not None


__RATING_COL_REGEX = re.compile('\\s+wrt\\s+')

def is_rating_col_name(col):
    if is_pw_col_name(col):
        return False
    else:
        return __RATING_COL_REGEX.search(col) is not None


def anp_manual_scales_from_excel(anp:ANPNetwork, excel_fname):
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


def anp_from_excel(excel_fname):
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
    conn_mat = get_matrix(pd.read_excel(excel_fname, sheet_name=1))
    #print(conn_mat)
    #print(conn_mat.shape)
    anp.node_connection_matrix(conn_mat)
    ## Now pairwise data
    df = pd.read_excel(excel_fname, sheet_name=2)
    df = df.transpose()
    #display(df)
    for col in df:
        # print(col)
        if is_pw_col_name(col):
            anp.import_pw_series(df[col])
        elif is_rating_col_name(col):
            # print("Rating column "+col)
            anp.import_rating_series(df[col])
    # Now let's setup manual rating scales
    anp_manual_scales_from_excel(anp, excel_fname)
    return anp