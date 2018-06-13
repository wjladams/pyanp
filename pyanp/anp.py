
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
        self.subnetwork = None
        self.invert = False

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
            #Make sure parent clusters are connected
            src_cluster = self.cluster
            dest_cluster = self.network._get_node_cluster(dest_node)
            src_cluster.cluster_connect(dest_cluster)

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
        self.prioritizer = Pairwise()
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

    def cluster_connect(self, dest_cluster):
        if isinstance(dest_cluster, ANPCluster):
            dest_cluster_name = dest_cluster.name
        else:
            dest_cluster_name = dest_cluster
        self.prioritizer.add_alt(dest_cluster_name, ignore_existing=True)


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

def sum_subnetwork_formula(priorities:pd.Series, dict_of_series:dict):
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
        self.subnet_formula = sum_subnetwork_formula
        self.default_priority_type = None

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

    def cluster_names(self):
        '''
        :return: List of string names of the clusters
        '''
        return list(self.clusters.keys())

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
        if islist(uname):
            for un in uname:
                self.add_user(un)
            return
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


    def node_names(self, cluster=None)->list:
        '''
        Returns a list of nodes in this network, organized by cluster

        :param cluster: If None, we get all nodes in network, else we get nodes
            in that cluster

        :return: List of strs of node names
        '''
        if cluster is not None:
            cl = self._get_cluster(cluster)
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
        return list(self.clusters.values())

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
        if self.has_subnetwork():
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

    def priority(self, username=None, ptype:PriorityType=None):
        '''
        Synthesize and return the alternative scores

        :param username: The user/users to synthesize for

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
            gp = self.global_priorities(username)
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
        npri: Pairwise
        if (wrtNode is not None) and (rowNode is not None) and (colNode is not None):
            # Node pairwise
            npri = wrtNode.get_node_prioritizer(rowNode, create=True)
            if not isinstance(npri, Pairwise):
                raise ValueError("Node prioritizer was not pairwise")
            npri.vote_series(series, row, col, createUnknownUser=True)
        else:
            # Try cluster pairwise
            wrtcluster = self._get_cluster(wrt)
            rowcluster = self._get_cluster(row)
            colcluster = self._get_cluster(col)
            if wrtcluster is None:
                raise ValueError("wrt="+wrt+" was not a cluster, and the group was not a node comparison")
            if rowcluster is None:
                raise ValueError("row="+row+" was not a cluster, and the group was not a node comparison")
            if colcluster is None:
                raise ValueError("col="+col+" was not a cluster, and the group was not a node comparison")
            npri = self.get_cluster_prioritizer(wrtcluster)
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

    def subnet(self, wrtnode):
        '''
        Makes wrtnode have a subnetwork if it did not already.

        :param wrtnode: The node to give a subnetwork to, or get the subnetwork
            of.
        :return: The ANPNetwork that is the subnet of this node
        '''
        node = self._get_node(wrtnode)
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

        :param node: The node to do this on

        :param value: If None, we return the boolean about if this node is
            inverted.  Otherwise specifies the new value.

        :return: T/F if value=None, telling if the node is inverted.  Otherwise
            returns nothing.
        '''
        node = self._get_node(node)
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

        :return: Nohting
        '''
        # First we need our global priorities
        pris = self.global_priorities(username)
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
        return [node for node in self.node_objs() if node.subnetwork is not None]

    def has_subnetwork(self):
        for node in self.node_objs():
            if node.subnetwork is not None:
                return True
        return False

    def invert_priority(self, p):
        rval = deepcopy(p)
        for i in range(len(p)):
            rval[i] = 1 - rval[i]
        return rval

    def synthesize_combine(self, priorities:pd.Series, alt_scores:dict):
        return self.subnet_formula(priorities, alt_scores)

    def get_cluster_prioritizer(self, wrtcluster):
        cluster = self._get_cluster(wrtcluster)
        return cluster.prioritizer


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