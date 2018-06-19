'''
A prioritizer is the root class of all things that prioritize objects (e.g. Pairwise and AHPTree).
'''

from enum import Enum
from copy import deepcopy
import numpy as np
import pandas

class PriorityType(Enum):
    '''
    An enumeration telling how to normalize priorities for a calculation
    '''
    RAW = 1
    """Leave the priorities unchanged."""
    NORMALIZE = 2
    """Divide priorities by sum, so that they sum to 1."""
    IDEALIZE = 3
    """Divide priorities by max, so that the largest is 1."""

    def apply(self, vals):
        '''
        Returns a copy of the parameter vals that has been adjusted as this
        PriorityType would.

        :param vals: A list-like object of values.  We return a copy that is adjusted.

        :return:  A list-like of the same type as 'vals' that has been normalized as
        this PriorityType would do.
        '''
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
    '''
    This class is the abstract representation of anything that prioritizes
    a list of items.  Examples include :py:class:`pyanp.pairwise.Pairwise`
    for doing group pairwise comparisons and :py:class:`pyanp.ahptree.AHPTree`
    for doing group AHP tree models.
    '''

    def add_alt(self, alt_name:str, ignore_existing=True)->None:
        '''
        Add an alternative to the prioritizer.  This should be overriden by
        the implementing class.

        :param alt_name: The name of the alternative to add.

        :return: Nothing
        '''
        raise ValueError("Should be overriden in subclass")

    def priority(self, username=None, ptype:PriorityType=None) -> pandas.Series:
        '''
        Calculates the alternative priorities.  Should be overriden by the
        implementing class.

        :param user_name: The name/names of the users to calculate the priority
        of.  If None, we get the priority of the group average.  If
        it is a string, we get the priority of that user.  If it is a list
        of users, we get the priority of the group average for that list of
        users.

        :param ptype: How should we normalize the resulting priorities
            (if at all).

        :return: A pandas.Series whose indices are the alternative names and
            whose values are the priorities of those alternatives.
        '''
        raise ValueError("Should be over riden in subclass")

    def nalts(self):
        '''
        :return: The number of alternatives (things you are pairwise comparing)
            in this group pairwise comparison object.
        '''
        return len(self.alt_names())

    def add_user(self, uname):
        '''
        Adds a user to this prioritizer object.

        :param user_name: The name of the user to add

        :return: Nothing

        :raises ValueError: If the user already existed.
        '''
        raise ValueError("Should be overriden in subclass")

    def user_names(self):
        '''
        :return: A list of the users in this prioritizer object.
        '''
        raise ValueError("Should be overriden in subclass")

    def _repr_html(self, tab="\t"):
        raise ValueError("Should override in subclass")

    def alt_names(self):
        '''
        :return: A list of the alts in this prioritizer object.
        '''
        raise ValueError("Should be overriden in subclass")

    def data_names(self, append_to=None, post_pend="")->str:
        '''
        Return string of newline separated names for the data
        that this prioritizer needs for each user.

        :param append_to: If not none, elements are appended here, otherwise
            a new list is created.

        :param post_pend: A string to post_pend to each name

        :return: List of strings of names.
        '''
        raise ValueError("Should be overriden in subclass")

    def priority_df(self, user_infos=None)->pandas.DataFrame:
        """
        Returns the priority scores dataframe for all users and the group

        :param user_infos: A list of users to do this for, if None is a part
            of this list, it means group average.  If None, it defaults to
            None plus all users.

        :return: pandas.DataFrame rows are alternatives, cols are users.
        """
        if user_infos is None:
            user_infos = list(self.user_names())
            user_infos.insert(0, None)
        rval = pandas.DataFrame()
        for user in user_infos:
            if user is None:
                uname = "Group Average"
            else:
                uname = user
            rval[uname] = self.priority(user)
        return rval