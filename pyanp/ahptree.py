'''
The AHP Tree class
'''

import pandas as pd
import numpy as np
import re

from pyanp.direct import Direct
from pyanp.pairwise import Pairwise
from pyanp.prioritizer import Prioritizer, PriorityType


class AHPTree(Prioritizer):
    def __init__(self, root_name="Goal", alt_names = None):
        if alt_names is None:
            alt_names = []
        self.alt_names = alt_names
        self.root = AHPNode(root_name, alt_names)

    def add_alt(self, alt_name):
        if alt_name in self.alt_names:
            raise ValueError("Cannot add duplicate alternative name "+alt_name)
        self.alt_names.append(alt_name)
        self.root.add_alt(alt_name)

    def nodepw(self, user, wrt, dom, rec, val):
        node = self.get_node(wrt)
        node.nodepw(user, dom, rec, val)

    def isalt(self, name):
        return name in self.alt_names

    def direct(self, wrt, node, val):
        nodeObj = self.get_node(wrt)
        if self.isalt(node):
            nodeObj.alt_direct(node, val)
        else:
            raise ValueError("Do not know how to direct non-alts, sorry")

    def add_user(self, user):
        self.root.add_user(user)

    def nalts(self):
        return len(self.alt_names)

    def priority(self, username=None, ptype:PriorityType=None):
        self.synthesize(username)
        return self.root.alt_scores

    def synthesize(self, username=None):
        self.root.synthesize(username)

    def add_child(self, childname, undername=None):
        if undername is None:
            under = self.root
        else:
            under = self.get_node(undername)
        under.add_child(childname)

    def get_nodes_hash(self):
        return self.root.get_nodes_under_hash()

    def get_node(self, nodename):
        nodes = self.get_nodes_hash()
        return nodes[nodename]

    def nodenames(self, underNode=None, rval=None):
        if rval is None:
            rval = []
        if underNode is None:
            underNode = self.root
        rval.append(underNode.name)
        for kid in underNode.children:
            self.nodenames(underNode=kid, rval=rval)
        return rval

    def _repr_html_(self):
        rval = "<ul>\n"
        rval = rval+self.root._repr_html(tab="\t")
        rval += "\n</ul>"
        return rval

class AHPNode:
    def __init__(self, name:str, alt_names):
        self.children = []
        nalts = len(alt_names)
        self.alt_scores = pd.Series(data=[0]*nalts, index=alt_names)
        self.child_prioritizer = Direct()
        self.alt_prioritizer = Direct(alt_names)
        self.alt_names = alt_names
        self.alt_scores_manually_set=False
        self.name = name

    def has_child(self, name):
        return name in [kid.name for kid in self.children]

    def add_child(self, childname):
        if self.has_child(childname):
            raise ValueError("Cannot duplicate children names")
        kidnode = AHPNode(childname, self.alt_names)
        self.children.append(kidnode)
        self.child_prioritizer.add_alt(childname)

    def childnames(self):
        return [child.name for child in self.children]

    def add_alt(self, alt_name):
        self.alt_prioritizer.add_alt(alt_name)
        self.alt_scores[alt_name]=0.0
        for kid in self.children:
            kid.add_alt(alt_name)

    def nalts(self):
        return len(self.alt_names)


    def has_children(self):
        return len(self.children) > 0

    def synthesize(self, username=None):
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

    def set_alt_scores(self, new_scores):
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

    def get_nodes_under_hash(self, rval:dict = None):
        if rval is None:
            rval = {}
        rval[self.name] = self
        for child in self.children:
            child.get_nodes_under_hash(rval)
        return rval

    def nodepw(self, user, dom, rec, val):
        if not isinstance(self.child_prioritizer, Pairwise):
            self.child_prioritizer = Pairwise(self.childnames())
        self.child_prioritizer.vote(user, dom, rec, val)

    def add_user(self, user):
        self.child_prioritizer.add_user(user)
        self.alt_prioritizer.add_user(user)

    def alt_direct(self, node, val):
        self.set_alt_scores({node:val})

    def _repr_html(self, tab=""):
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


class ColInfo:
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

def colinfos_fromdf(df:pd.DataFrame):
    rval = [ColInfo(col) for col in df]
    return rval

def nodes_from_colinfos(infos):
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


def node_parents(colinfos, node):
    rval = []
    for info in colinfos:
        if info.compares(node):
            rval.append(info.wrt())
    rval = list(dict.fromkeys(rval))
    return rval


def node_alts(colinfos, nodes):
    rval = [node for node in nodes if len(node_parents(colinfos, node)) > 1]
    return rval


def node_root(colinfos, nodes):
    rval = [node for node in nodes if len(node_parents(colinfos, node)) <= 0]
    return rval


def node_children(colinfos, node):
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


def create_ahptree(colinfos, currentAHPTree=None, currentNode=None) -> AHPTree:
    if isinstance(colinfos, str):
        colinfos = pd.read_excel(colinfos)
    df = colinfos
    if isinstance(colinfos, pd.DataFrame):
        colinfos = colinfos_fromdf(colinfos)
    nodes = nodes_from_colinfos(colinfos)
    alts = node_alts(colinfos, nodes)
    #print(alts)
    root = node_root(colinfos, nodes)
    if len(root) > 1:
        raise ValueError("Too many root nodes, needs exactly1, had " + str(root))
    root = root[0]
    isToplevel = False
    if currentAHPTree is None:
        isToplevel = True
        currentAHPTree = AHPTree(root)
        currentNode = root
    for kid in node_children(colinfos, currentNode):
        if kid not in alts:
            #print("Adding node=" + kid + " under=" + currentNode)
            currentAHPTree.add_child(kid, currentNode)
            create_ahptree(colinfos, currentAHPTree, kid)
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
                    if (not np.isnan(val)) and (user != "all"):
                        currentAHPTree.nodepw(user, wrt, dom, rec, val)
            elif info.isdirect():
                wrt = info.wrt()
                node = info.node()
                for user in colseries.index:
                    val = colseries[user]
                    if not np.isnan(val):
                        currentAHPTree.direct(wrt, node, val)
    return currentAHPTree
