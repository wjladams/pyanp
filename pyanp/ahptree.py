'''
The AHP Tree class
'''

import pandas as pd
import numpy as np

from pyanp.direct import Direct
from pyanp.prioritizer import Prioritizer, PriorityType


class AHPTree(Prioritizer):
    def __init__(self, alt_names = None):
        if alt_names is None:
            alt_names = []
        self.alt_names = alt_names
        self.root = AHPNode("Root", alt_names)

    def add_alt(self, alt_name):
        if alt_name in self.alt_names:
            raise ValueError("Cannot add duplicate alternative name "+alt_name)
        self.alt_names.append(alt_name)
        self.root.add_alt(alt_name)

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

    def add_alt(self, alt_name):
        self.alt_prioritizer.add_alt(alt_name)

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
            self.alt_scores = pd.Series([0]*self.nalts(), index=self.alt_names, dtype=float)
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