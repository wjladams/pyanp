'''
Code for the direct data object
'''

import numpy as np
import pandas as pd

from pyanp.prioritizer import Prioritizer, PriorityType, priority_type_default


class Direct(Prioritizer):
    def __init__(self, alt_names=None):
        if alt_names is None:
            alt_names = []
        data = []
        if len(alt_names) > 0:
            data = [0]*len(alt_names)
        self.data = pd.Series(data=data, index=alt_names)
        self.loc = self.data.loc
        self.iloc = self.data.iloc


    def __getitem__(self, item):
        return self.data[item]

    def add_alt(self, alt_name):
        if alt_name in self.data.index:
            raise ValueError("Cannot have duplicate alternative names")
        self.data[alt_name] = 0


    def priority(self, username=None, ptype:PriorityType=None):
        if ptype is None:
            ptype = priority_type_default()
        return ptype.apply(self.data)