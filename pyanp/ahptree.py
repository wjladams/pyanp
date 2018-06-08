'''
The AHP Tree class and functionality to create one from a spreadsheet
'''

import pandas as pd
import numpy as np
import re

from pyanp.direct import Direct
from pyanp.pairwise import Pairwise
from pyanp.prioritizer import Prioritizer, PriorityType


class AHPTreeNode:
    '''
    Represents a node in an AHPTree class
    '''
    def __init__(self, parent, name:str, alt_names):
        '''
        Initial a new AHPTreeNode

        :param parent: The parent AHPTree this AHPTreeNode is in.

        :param name: The string name of this node.  It should be unique in its parent tree.

        :param alt_names: The alternatives we are comparing in the AHPTree.  As currently implemented
        the parent tree has the list of alternatives and we pass that object to the nodes.  This allows us
        to add new alternatives once in the parent tree and the addition cascades down.
        '''
        self.parent = parent
        self.children = []
        nalts = len(alt_names)
        self.alt_scores = pd.Series(data=[0]*nalts, index=alt_names)
        self.child_prioritizer = Direct()
        self.alt_prioritizer = Direct(alt_names)
        self.alt_names = alt_names
        self.alt_scores_manually_set=False
        self.name = name

    def has_child(self, name:str)->bool:
        '''
        Returns a boolean telling if this node has a child with the given name.

        :param name: The name of the child to check for.

        :return: True/False if the node has the child by the given name or not.
        '''
        return name in [kid.name for kid in self.children]

    def add_child(self, childname:str)->None:
        '''
        Adds a child to this node.
        :param childname: The string name of the child to add.
        :return:
        Nothing
        :raises ValueError:
        If a child by the given name already existed
        '''
        if self.has_child(childname):
            raise ValueError("Cannot duplicate children names")
        kidnode = AHPTreeNode(self.parent, childname, self.alt_names)
        self.children.append(kidnode)
        self.child_prioritizer.add_alt(childname)

    def childnames(self):
        '''
        Get the names of the children of this node
        :return:
        A list of str's of the names of this nodes children.  If it has no children
        we return the empty list.
        '''
        return [child.name for child in self.children]

    def add_alt(self, alt_name:str)->None:
        '''
        Adds an alternative to the alternatives under this node.
        :param alt_name: The new alternative to add
        :return:
        Nothing
        :raises ValueError:
        If the alternative already existed
        '''
        self.alt_prioritizer.add_alt(alt_name)
        self.alt_scores[alt_name]=0.0
        for kid in self.children:
            kid.add_alt(alt_name)

    def nalts(self)->int:
        '''
        Gets the number of alternatives under this node.
        '''
        return len(self.alt_names)


    def has_children(self)->int:
        '''
        :return:
        A boolean telling if this node has children
        '''
        return len(self.children) > 0

    def synthesize(self, username=None)->None:
        '''
        Synthesizes up the alternative scores below this alternative and stores the
        result in the alt_scores.  However if the node has no children and has had
        it's alternative scores manually set via AHPTreeNode.set_alt_scores, then this
        does nothing.  Otherwise it synthesizes upward.

        :param username: The name of the user (or list of names of the users) to synthesize for.

        :return:
        Nothing
        '''
        if self.has_children():
            nalts = self.nalts()
            rval = pd.Series(data=[0]*nalts, index=self.alt_names)
            kidpris = self.child_prioritizer.priority(username, PriorityType.NORMALIZE)
            if np.sum(np.abs(kidpris)) == 0:
                nkids = len(kidpris)
                for key, value in kidpris.iteritems():
                    kidpris[key]=1.0 / nkids
            for child, childpri in zip(self.children, kidpris):
                child.synthesize(username)
                rval += childpri * child.alt_scores
            self.alt_scores = rval
        else:
            if self.alt_scores_manually_set:
                # Do nothing here, alt scores are already setup
                pass
            else:
                self.alt_scores = self.alt_prioritizer.priority(username)

    def set_alt_scores(self, new_scores:dict):
        '''
        Used to manually set (or unset) alternative scores.  If new_scores is None, it unsets
        the manually set values, so that the next call to AHPTreeNode.synthesize() will actually
        synthesize the scores and not use the manually set values.
        :param new_scores: If None, it means undo the manual setting of the scores, otherwise
        it loops over each key, value pair and sets the score in AHPTreeNode.alt_scores
        :return:
        '''
        if new_scores is None:
            self.alt_scores_manually_set = False
        else:
            self.alt_scores_manually_set=True
            #self.alt_scores = pd.Series([0]*self.nalts(), index=self.alt_names, dtype=float)
            if isinstance(new_scores, dict):
                for key, value in new_scores.items():
                    if key not in self.alt_scores.index:
                        raise ValueError("Tried to score alt "+key+" that did not exist.")
                    self.alt_scores[key] = value
            else:
                raise ValueError("Do not know how to set alt scores from type "+type(new_scores))

    def get_nodes_under_hash(self, rval:dict = None)->dict:
        '''
        Returns a dictionary of nodeName:AHPTreeNode of the nodes under this node.  It includes this node as well.
        :param rval: If passed in, we add the dictionary items to this dictionary
        :return: The dictionary of name:AHPTreeNode objects
        '''
        if rval is None:
            rval = {}
        rval[self.name] = self
        for child in self.children:
            child.get_nodes_under_hash(rval)
        return rval

    def nodepw(self, username:str, row:str, col:str, val:float, createUnknownUser=True)->None:
        '''
        Does a pairwise comparison of the children.  If there is not a pairwise comparison
        object being used to prioritize the children, we create one first.
        :param username: The user to perform the comparison on.
        :param row:  The name of the row node of the comparison
        :param col:  The name of the column node of the comparison
        :param val: The comparison value
        :param createUnknownUser: If True, and username did not exist, it will be created and then the vote set.
        Otherwise if the user did not exist, will raise an exception.
        :return:
        Nothing
        :raises ValueError: If the user did not exist and createUnknownUser is False.
        '''
        if not isinstance(self.child_prioritizer, Pairwise):
            self.child_prioritizer = Pairwise(self.childnames())
        self.child_prioritizer.vote(username, row, col, val, createUnknownUser=createUnknownUser)

    def add_user(self, user:str)->None:
        '''
        Adds a user to the prioritizers below this
        :param user: The name of the user to add
        :return:
        Nothing
        :raises ValueError: If the user already existed
        '''
        self.child_prioritizer.add_user(user)
        self.alt_prioritizer.add_user(user)

    def alt_direct(self, node, val):
        '''
        Manually sets the alternative score.  See AHPTreeNode.set_alt_scores() for more info.
        :param node:
        :param val:
        :return:
        '''
        self.set_alt_scores({node:val})

    def _repr_html(self, tab=""):
        '''
        Used by Jupyter to pretty print an instance of AHPTreeNode
        :param tab: How many tabs should we indent?
        :return:
        The html string pretty print version of this
        '''
        rval = tab+"<li><b>Node:</b>"+self.name+"\n"
        if self.has_children():
            # Append child prioritizer info
            rval += self.child_prioritizer._repr_html(tab+"\t")
        if self.has_children():
            rval += tab+"<ul>\n"
            for child in self.children:
                rval += child._repr_html(tab+"\t")
            rval += "</ul>\n"
        else:
            # Should connect to alternatives, let's just report scores
            altscoresstr = tab+"\t\t"+str(self.alt_scores)+"\n"
            altscoresstr = re.sub("\n", "\n"+tab+"\t\t", altscoresstr)
            altscoresstr = altscoresstr.rstrip()
            rval += tab+"\t"+"<ul><li>AltScores=\n"+altscoresstr+"\n"
            rval += tab+"\t"+"</ul>\n"
        return rval

    def usernames(self, rval:list=None)->list:
        '''
        Returns the names of all users involved in this AHPTreeNode
        :param rval: If not None, we add the names to this list
        :return:
        List of str user names.
        '''
        if rval is None:
            rval = []
        if self.child_prioritizer is not None:
            users = self.child_prioritizer.usernames()
            for user in users:
                if user not in rval:
                    rval.append(user)
        if self.alt_prioritizer is not None:
            for user in self.alt_prioritizer.usernames():
                if user not in rval:
                    rval.append(user)
        return rval

class AHPTree(Prioritizer):
    '''
    Represents all of the data of an ahp tree.
    '''
    def __init__(self, root_name="Goal", alt_names = None):
        '''
        Creates a new AHPTree object

        :param root_name: The name of the root node of the tree, defaults to Goal.

        :param alt_names: The alts to start this tree with.
        '''
        if alt_names is None:
            alt_names = []
        self.alt_names = alt_names
        self.root = AHPTreeNode(self, root_name, alt_names)

    def add_alt(self, alt_name:str)->None:
        '''
        Adds an alternative to this tree and all of the nodes in the tree.

        :param alt_name: The name of the new alternative to add.

        :return: Nothing

        :raises ValueError: If an alternative already existed with the given name
        '''
        if alt_name in self.alt_names:
            raise ValueError("Cannot add duplicate alternative name "+alt_name)
        self.alt_names.append(alt_name)
        self.root.add_alt(alt_name)

    def nodepw(self, username:str, wrt:str, row:str, col:str, val, createUnknownUser=True)->None:
        '''
        Pairwise compares a nodes for a given user.

        :param username: The name of the user to do the comparison for.  If the user doesn't exist, this will create
        the user if createUnknownUser is True, otherwise it will raise an exception

        :param wrt: The name of the wrt node.

        :param row: The name of the row node for the comparison, i.e. the dominant node.

        :param col: The name of the column node for the comparison, i.e. the recessive node.

        :param val: The vote value

        :return: Nothing

        :raises ValueError: If wrt, row, or col node did not exist.  Also if username did not exist and
        createUnknownUsers is False.
        '''
        node = self.get_node(wrt)
        node.nodepw(username, row, col, val, createUnknownUser=createUnknownUser)

    def isalt(self, name:str)->bool:
        '''
        Tells if the given alternative name is an alternative in this tree.
        :param name: The name of the alternative to check.
        :return:
        True if the alternative is in the list of alts for this tree, false otherwise.
        '''
        return name in self.alt_names

    def alt_direct(self, wrt:str, alt_name:str, val:float)->None:
        '''
        Directly sets the alternative score under wrt node.  See AHPTreeNode.alt_direct for more information
        as that is the function that does the hard work.
        :param wrt: The name of the wrt node.
        :param alt_name: The name of the alternative to direclty set.
        :param val: The new directly set value.
        :return:
        Nothing
        :raises ValueError:
        * If there is no alternative by that name
        * If the wrt node did not exist
        '''
        nodeObj = self.get_node(wrt)
        if self.isalt(alt_name):
            nodeObj.alt_direct(alt_name, val)
        else:
            raise ValueError("Do not know how to direct non-alts, sorry")

    def add_user(self, user:str)->None:
        '''
        Adds the user to this AHPTree object.
        :param user: The name of the user to add
        :return:
        Nothing
        :raises ValueError:
        If the user already existed.
        '''
        self.root.add_user(user)

    def usernames(self)->list:
        '''
        :return:
        The names of the users in this tree.
        '''
        return self.root.usernames()

    def nalts(self):
        '''
        :return:
        The number of alternatives in this tree.
        '''
        return len(self.alt_names)

    def priority(self, username=None, ptype:PriorityType=None)->pd.Series:
        '''
        Calculates the scores of the alternatives.  Calls AHPTree.synthesize() first to calculate.
        :param username: The name (or list of names) of the user (users) to synthensize.  If username is None,
        we calculate for the group.
        :param ptype: Do we want to rescale the priorities to add to 1 (normalize), or so that the largest value
        is a 1 (idealize), or just leave them unscaled (Raw).
        :return:
        The alternative scores, which is a pd.Series whose index is alternative names, and values are the scores.
        '''
        self.synthesize(username)
        return self.root.alt_scores

    def synthesize(self, username=None)->None:
        '''
        Does ahp tree synthesis to calculate the alternative scores wrt to all nodes in the tree.

        :param username: The name/names of the user/users to synthesize wrt.  If None, that means do the full \
        group average.

        :return:
        Nothing
        '''
        self.root.synthesize(username)

    def add_child(self, childname:str, undername:str=None)->None:
        '''
        Adds a child node of a given name under a node.

        :param childname: The name of the child to add.

        :param undername: The name of the node to add the child under

        :return: Nothing

        :raises ValueError: If undername was not a node, or if childname already existed as a node.
        '''
        if undername is None:
            under = self.root
        else:
            under = self.get_node(undername)
        under.add_child(childname)

    def get_nodes_hash(self)->dict:
        '''
        :return:
        A dictionary of nodeName:nodeObject for all nodes in this tree.
        '''
        return self.root.get_nodes_under_hash()

    def get_node(self, nodename:str)->AHPTreeNode:
        '''
        :param nodename: The string name of the node to get.  If None, we return the root node.
        If nodename is actually an AHPTreeObject, we simply return that object.

        :return: The AHPTreeNode object corresponding to the node with the given name

        :raises KeyError: If no such node existed
        '''
        if nodename is None:
            return self.root
        elif isinstance(nodename, AHPTreeNode):
            return nodename
        else:
            nodes = self.get_nodes_hash()
            return nodes[nodename]

    def nodenames(self, undername:str=None, rval=None)->list:
        '''
        Name of all nodes under the given node, including that node.

        :param undername: The name of the node to get all nodes under, but only if underNode is not set.
        It can also be an AHPTreeNode, but that is really for internal use only

        :param rval: If not

        :return: The node names as a list
        '''
        if rval is None:
            rval = []
        underNode = self.get_node(undername)
        rval.append(underNode.name)
        for kid in underNode.children:
            self.nodenames(undername=kid, rval=rval)
        return rval

    def _repr_html_(self):
        '''
        Used by Jupyter to pretty print an AHPTree instance
        :return:
        '''
        rval = "<ul>\n"
        rval = rval+self.root._repr_html(tab="\t")
        rval += "\n</ul>"
        return rval

    def global_priority(self, username = None, rvalSeries=None, undername:str=None, parentMultiplier=1.0) -> pd.Series:
        '''
        Calculates and returns the global priorities of the nodes.

        :param username: The name/names of the users to calculate for.  None means the group average.

        :param rvalSeries: If not None, add the results to that series

        :param undername: If None, use the root node, otherwise a string for the name of the node to go under.  Internally
            we also allow for AHPTreeNode's to be passed in this way.

        :param parentMultiplier: The value to multiply the child priorities by.

        :return: The global priorities as a Series whose index is the node names, and values are the global priorities.
        '''
        if rvalSeries is not None:
            rval = rvalSeries
        else:
            rval = pd.Series(dtype=float)
        underNode = self.get_node(undername)
        rval[underNode.name] = parentMultiplier
        if not underNode.has_children():
            # We are done
            return rval
        kidpris = underNode.child_prioritizer.priority(username=username)
        for kid, pri in zip(underNode.children, kidpris):
            self.global_priority(username, rval, undername=kid, parentMultiplier=parentMultiplier * pri)
        return rval

    def global_priority_table(self)->pd.DataFrame:
        '''
        Calculates the global priorities for every user, and the group

        :return: A dataframe whose columns are "Group" for the total group average, and then each user name.  The
            rows are the node names, and values are the global priority for the given node and user.
        '''
        average = self.global_priority()
        rval = pd.DataFrame(index=average.index)
        rval['Group']=average
        users = self.usernames()
        for user in users:
            rval[user] = self.global_priority(user)
        return rval

    def priority_table(self)->pd.DataFrame:
        '''
        :return: A dataframe whose columns are "Group" for the total group average, and then each user name.
        The rows are the alternative names, and the values are the alternative scores for each user.
        '''
        average = self.priority()
        rval = pd.DataFrame(index=average.index)
        rval['Group']=average
        users = self.usernames()
        for user in users:
            rval[user] = self.priority(user)
        return rval

    def incon_std(self, username, wrt:str=None)->float:
        '''
        Calcualtes the standard inconsistency score for the pairwise comparison of the children nodes
        for the given user

        :param username: The string name/names of users to do the inconsistency for.  If more than one user
            we average their pairwise comparison matrices and then calculate the incosnsitency of the result.

        :param wrt: The name of the node to get the inconsistency around.  If None, we use the root node.

        :return: The standard Saaty inconsistency score.
        '''
        node = self.get_node(wrt)
        if isinstance(node.child_prioritizer, Pairwise):
            return node.child_prioritizer.incon_std(username)
        else:
            return None

    def nodes(self, undername:str=None, rval=None):
        '''
        Returns the AHPTreeNode objects under the given node, including that node

        :param undername: The string name of the node to get the nodes under.  It can also be an AHPTreeNode object
            as well.  If None it means the root node.

        :param rval: If not None, it should be a list to add the AHPTreeNode's to.

        :return: The list of AHPTreeNode objects under the given node.
        '''
        underNode = self.get_node(undername)
        if rval is None:
            rval = []
        rval.append(underNode)
        for node in underNode.children:
            self.nodes(node, rval)
        return rval

    def incon_std_series(self, username:str)->pd.Series:
        '''
        Calculates the inconsistency for all wrt nodes for a user / user group.  See AHPTree.incon_std()
        for details about the actual calculation.

        :param username: The name/names of the user to calculate the inconsistency for.

        :return: A pandas.Series whose index is wrt node names, and whose values are the inconsistency of the given user(s)
            on that comparison.
        '''
        nodes = self.nodes()
        nodesWithKids = [node for node in nodes if node.has_children()]
        rval = [self.incon_std(username, node.name) for node in nodesWithKids if node.has_children()]
        rval = pd.Series(data=rval, index=[node.name for node in nodesWithKids])
        return rval

    def incond_std_table(self)->pd.DataFrame:
        '''
        Calculates the inconsistency for all users and wrt nodes in this tree.

        :return: A pandas.DataFrame whose columns are users (first column is called "Group" and is for the group average) and
            whose rows are wrt nodes.  The values are the inconsistencies for the given user on the give wrt node's
            pairwise comparison.
        '''
        average = self.incon_std_series(username=None)
        rval = pd.DataFrame(index=average.index)
        rval['Group']=average
        users = self.usernames()
        for user in users:
            rval[user] = self.incon_std_series(user)
        return rval

    def node_pwmatrix(self, username, wrt:str)->np.ndarray:
        '''
        Gets the pairwise comparison matrix for the nodes under wrt.

        :param username: The name/names of the users to get the pairwise comparison of.

        :param wrt: The name of the wrt node, or the AHPTreeNode object.

        :return: A numpy array of the pairwise comparison information.  If more than one user specified in usernames param
            we take the average of the group.
        '''
        node = self.get_node(wrt)
        pri = node.child_prioritizer
        if isinstance(pri, Pairwise):
            return pri.matrix(username)
        else:
            return None

class _ColInfo:
    '''
    Used internally by ahptree.ahptree_fromdf()
    '''
    __wrtre = re.compile("^(.+)\s+vs\s+(.+)\s+wrt\s+(.+)$")
    __avsb = re.compile("^(.+)\s+vs\s+(.+)$")
    __directre = re.compile("^(.+)\s+wrt\s+(.+)$")
    __theGoalNode = "Goal"
    def __init__(self, col):
        # print("For col="+col)
        self.column = col
        minfo = self.__wrtre.match(col)
        dom, rec, wrt = [None] * 3
        if minfo is not None:
            dom, rec, wrt = minfo.groups()
            self.data=("pw", dom, rec, wrt)
        else:
            # We do not have wrt, try a vs b
            minfo = self.__avsb.match(col)
            if minfo is not None:
                wrt = self.__theGoalNode
                dom, rec = minfo.groups()
                self.data=("pw", dom, rec, wrt)
            else:
                # If we made it here, it is not pairwise, try direct
                minfo = self.__directre.match(col)
                if minfo is not None:
                    node, wrt = minfo.groups()
                    self.data=("direct", node, wrt)
                else:
                    # If we made it here, it has to be demographic
                    self.data=("demo", col)


    def ispw(self):
        return self.data[0] == "pw"


    def isdirect(self):
        return self.data[0] == "direct"


    def wrt(self):
        if self.ispw() or self.isdirect():
            return self.data[-1]
        else:
            return None

    def node(self):
        if self.isdirect():
            return self.data[1]
        else:
            return None

    def dom(self):
        if self.ispw():
            return self.data[1]
        else:
            return None

    def rec(self):
        if self.ispw():
            return self.data[2]
        else:
            return None


    def compares(self, node):
        if self.ispw():
            return (self.data[1] == node) or (self.data[2] == node)
        elif self.isdirect():
            return (self.data[1] == node)
        else:
            return False

def _colinfos_fromdf(df:pd.DataFrame):
    rval = [_ColInfo(col) for col in df]
    return rval

def _nodes_from_colinfos(infos):
    nodes = []
    for info in infos:
        if info.ispw():
            nodes.append(info.data[1])
            nodes.append(info.data[2])
            nodes.append(info.data[3])
        elif info.isdirect():
            nodes.append(info.data[1])
            nodes.append(info.data[2])
    rval = list(dict.fromkeys(nodes))
    return rval


def _node_parents(colinfos, node):
    rval = []
    for info in colinfos:
        if info.compares(node):
            rval.append(info.wrt())
    rval = list(dict.fromkeys(rval))
    return rval


def _node_alts(colinfos, nodes):
    rval = [node for node in nodes if len(_node_parents(colinfos, node)) > 1]
    return rval


def _node_root(colinfos, nodes):
    rval = [node for node in nodes if len(_node_parents(colinfos, node)) <= 0]
    return rval


def _node_children(colinfos, node):
    rval = []
    for info in colinfos:
        if info.wrt() == node:
            if info.ispw():
                rval.append(info.data[1])
                rval.append(info.data[2])
            elif info.isdirect():
                rval.append(info.data[1])
    rval = list(dict.fromkeys(rval))
    return rval


def ahptree_fromdf(colinfos, currentAHPTree=None, currentNode=None) -> AHPTree:
    '''
    Create an AHPTree object from a well formated dataframe/spreadsheet of values.
    :param colinfos: Can either be:
    1. A string that is the name of excel file to read the data in from.
    2. A pd.DataFrame of the data to use
    3. The list of _ColInfos from a call to ahptree.__colinfos_fromdf()
    :param currentAHPTree:
    If not None, it is the ahptree we are adding information to.  This is really
    here so that the function can recursively call itself and shouldn't be used.
    :param currentNode:
    If not None, the current node we are parsing at.  This is really
    here so that the function can recursively call itself and shouldn't be used.
    :return:
    The AHPTree that contains the data from the spreadsheet
    '''
    if isinstance(colinfos, str):
        colinfos = pd.read_excel(colinfos)
    df = colinfos
    if isinstance(colinfos, pd.DataFrame):
        colinfos = _colinfos_fromdf(colinfos)
    nodes = _nodes_from_colinfos(colinfos)
    alts = _node_alts(colinfos, nodes)
    #print(alts)
    root = _node_root(colinfos, nodes)
    if len(root) > 1:
        raise ValueError("Too many root nodes, needs exactly1, had " + str(root))
    root = root[0]
    isToplevel = False
    if currentAHPTree is None:
        isToplevel = True
        currentAHPTree = AHPTree(root)
        currentNode = root
    for kid in _node_children(colinfos, currentNode):
        if kid not in alts:
            #print("Adding node=" + kid + " under=" + currentNode)
            currentAHPTree.add_child(kid, currentNode)
            ahptree_fromdf(colinfos, currentAHPTree, kid)
    # Finally add alts, but only if in top-level
    if isToplevel:
        for alt in alts:
            currentAHPTree.add_alt(alt)
        for user in df.index:
            if user != "all":
                currentAHPTree.add_user(user)
        # Now let's do all of the votes
        for info in colinfos:
            colseries = df[info.column]
            if info.ispw():
                wrt = info.wrt()
                dom = info.dom()
                rec = info.rec()
                for user in colseries.index:
                    val = colseries[user]
                    # print(val)
                    if (not np.isnan(val)) and (user != "all"):
                        currentAHPTree.nodepw(user, wrt, dom, rec, val)
            elif info.isdirect():
                wrt = info.wrt()
                node = info.node()
                for user in colseries.index:
                    val = colseries[user]
                    if not np.isnan(val):
                        currentAHPTree.alt_direct(wrt, node, val)
    return currentAHPTree
