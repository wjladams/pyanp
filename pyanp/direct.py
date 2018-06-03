'''
Code for the direct data object
'''

import numpy as np
import pandas as pd

from pyanp.prioritizer import Prioritizer, PriorityType, priority_type_default


class Direct(Prioritizer):
    '''
    Represents the concept of directly setting data.  It is a single user only
    :class:`pyanp.prioritizer.Prioritizer` instance.
    '''
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

    def add_alt(self, alt_name:str)->None:
        '''
        Adds an alternative.

        :param alt_name: The name of the alt to add.

        :return: Nothing

        :raises ValueError: If the alternative already existed
        '''
        if alt_name in self.data.index:
            raise ValueError("Cannot have duplicate alternative names")
        self.data[alt_name] = 0


    def priority(self, username=None, ptype:PriorityType=None):
        '''
        Gets the priority for the given user.  At the moment it simply ignores
        user since direct data only stores one data set for all users.

        :param username: The name of the user, but it is ignored.

        :param ptype: Should we normalize, idealize, or leave the priority alone.

        :return: A pandas.Series whose index is the alternative names and whose values are the priorities.
        '''
        if ptype is None:
            ptype = priority_type_default()
        return ptype.apply(self.data)

    def add_user(self, uname:str)->None:
        '''
        Does nothing since Direct current does not have users.

        :param uname: The name of the user we should add

        :return: Nothing
        '''
        pass

    def usernames(self):
        '''
        Direct has no notion of users at the moment, so this returns the empty list.

        :return: Empty list
        '''
        return []