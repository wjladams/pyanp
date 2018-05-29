'''
A prioritizer is the root class of all
'''

from enum import Enum, auto
from copy import deepcopy
import numpy as np
import pandas

class PriorityType(Enum):
    RAW = auto()
    NORMALIZE = auto()
    IDEALIZE = auto()

    def apply(self, vals):
        rval = copy_array_as_float(vals)
        if self == PriorityType.RAW:
            return rval
        elif self == PriorityType.NORMALIZE:
            s = np.sum(np.abs(vals))
            if s != 0:
                for i in range(len(rval)):
                    rval[i] /= float(s)
            return rval
        elif self == PriorityType.IDEALIZE:
            s = max(np.abs(vals))
            if s != 0:
                for i in range(len(rval)):
                    rval[i] /= s
            return rval
        else:
            raise ValueError("Unknown PriorityType "+str(self))

def priority_type_default():
    return PriorityType.RAW


def copy_array_as_float(src):
    if isinstance(src, (list, tuple)):
        return deepcopy(src)
    elif hasattr(src, "dtype"):
        rval = src.astype(float)
        return rval
    else:
        return deepcopy(src)



class Prioritizer:
    def add_alt(self, alt_name):
        raise ValueError("Should be overriden in subclass")

    def priority(self, username=None, ptype:PriorityType=None) -> pandas.Series:
        raise ValueError("Should be over riden in subclass")

    def nalts(self):
        raise ValueError("Should be overrriden in subclass")

    def add_user(self, uname):
        raise ValueError("Should be overriden in subclass")

    def ussernames(self):
        raise ValueError("Should be overriden in subclass")

    def _repr_html(self, tab="\t"):
        raise ValueError("Should override in subclass")