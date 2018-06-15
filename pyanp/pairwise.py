'''
Group pairwise object and calculations.  See :py:mod:`pyanp.priority` for
all methods of calculating priorities from a pairwise comparison matrix
in addition to inconsistency calculations.
'''

import numpy as np
import pandas as pd
from pyanp.priority import incon_std
from pyanp.general import islist
from pyanp.prioritizer import Prioritizer, PriorityType
from pyanp.priority import pri_eigen
from copy import deepcopy
import re

class Pairwise(Prioritizer):
    '''
    Creates a new group pairwise comparison object.

    :param alts: The list alternatives (things you are comparing) to start with.
        Should be a list-like object of strings.

    :param users: The users to start the group pairwise comparison object
        with.  It should be a list-like object of strings.

    :param demographic_cols: The names of the demographic columns to start
        the group pairwise comparison object with.  It should be a list-like
        object of strings.
    '''
    def __init__(self, alts=None, users=None, demographic_cols = None):
        if alts is None:
            alts = []
        if users is None:
            users = []
        if demographic_cols is None:
            demographic_cols = ['Name', 'Age']
        self.alts = alts
        all_cols = demographic_cols + ['Matrix']
        #if 'Name' not in demographic_cols:
        #    all_cols = ['Name'] + all_cols
        self.df = pd.DataFrame(data = None, columns=all_cols)
        self.priority_calc = pri_eigen


    def is_user(self, user_name:str)->bool:
        '''
        Checks if a user exists in this group pairwise comparison object.

        :param user_name: The name of the user to look for

        :return: True/False
        '''
        return user_name in self.df.index

    def is_alt(self, alt_name:str)->bool:
        '''
        Checks if an alternative (a thing you are pairwise comparing) exists in
        this group pairwise comparison object.

        :param alt_name: The name of the alternative to check for.

        :return: True/False
        '''
        return alt_name in self.alts

    def nalts(self)->int:
        '''
        :return: The number of alternatives (things you are pairwise comparing)
            in this group pairwise comparison object.
        '''
        return len(self.alts)

    def _blank_pairwise(self):
        '''
        Creates a blank pairwise comparison for the right number of alts
        '''
        nalts = self.nalts()
        return np.identity(nalts)

    def add_user(self, user_name:str)->None:
        '''
        Adds a user to this group pairwise comparison object.

        :param user_name: The name of the user to add

        :return: Nothing

        :raises ValueError: If the user already existed.
        '''
        if self.is_user(user_name):
            raise ValueError("User "+user_name+" already existed")
        ncols = len(self.df.columns)
        data = [None]*(ncols-1)+[self._blank_pairwise()]
        self.df.loc[user_name] = data

    def add_alt(self, alt_name:str, ignore_existing=False)->None:
        '''
        Adds an alternative (thing you are pairwise comparing) to this group
        pairwise comparison object.

        :param alt_name: The name of the alternative to add.

        :return: Nothing

        :raises ValueError: If the alternative already esisted.
        '''
        if self.is_alt(alt_name):
            if ignore_existing:
                return
            else:
                raise ValueError("Alt "+alt_name+" already existed")
        self.alts.append(alt_name)
        for user in self.df.index:
            mat = self.matrix(user)
            self.df.loc[user, "Matrix"] = add_place(mat)


    def matrix(self, user_name=None, createUnknownUser:bool=True)->np.ndarray:
        '''
        Gets the pairwise comparison for a user or group of users.

        :param user_name: The name/names of the user/users to get the
            comparisons of.  If None, that means to get the group average for
            all users.  If it is a string, that means get the pairwise comparison
            matrix of that user.  If it is a list-like of strings, we get the
            group average matrix for all of those users.

        :param createUnknownUser: If True and the user_name did not exist, we
            should create that user.  Otherwise throw an error if we request
            for a non-existant user.

        :return: The numpy array of the pairwise comparisons.

        :raises ValueError: If createUnknownUser=False and we request for a single
            non-existant user.
        '''
        if user_name is None:
            user_name = self.usernames()
        if isinstance(user_name, (str, int, float)):
            # We are just doing a single user
            if not self.is_user(user_name):
                if createUnknownUser:
                    self.add_user(user_name)
                else:
                    raise ValueError("No such user " + user_name)
            return self.df.loc[user_name, "Matrix"]
        else:
            mats = [self.df.loc[user, 'Matrix'] for user in user_name]
            if len(mats) == 0:
                return np.identity(self.nalts(), dtype=float)
            return geom_avg_mats(mats)

    def incon_std(self, user_name)->float:
        '''
        Calculates the standard Saaty pairwise comparison inconsistency for
        a user or group of users.

        :param user_name: The name/names of the users to get the inconsistency
            of.  If None, we get the inconsistency of the group average matrix.  If
            it is a string, we get the inconsistency of that user.  If it is a list
            of users, we get the inconsistency of the group average for that list of
            users.

        :return: The Saaty inconsistency score.
        '''
        matrix = self.matrix(user_name)
        return incon_std(matrix)


    def alt_index(self, alt_name_or_index)->int:
        '''
        Find the index (integer location) of the given alternative in the
        pairwise comparison matrices.

        :param alt_name_or_index: If this is an integer, we simply return that
            integer.  Otherwise we look up the index of the alternative name in
            the list of alternatives in this object.

        :return: The index that alternative has in the pairwise comparison
            matrices.
        '''
        if isinstance(alt_name_or_index, (int)):
            return alt_name_or_index
        if alt_name_or_index not in self.alts:
            raise ValueError("No such alt "+alt_name_or_index)
        return self.alts.index(alt_name_or_index)

    def vote_series(self, votes:pd.Series, row, col, createUnknownUser:bool=True)->None:
        '''
        Changes a single pairwise value for a series of users.

        :param votes: Series whose index is usernames and values are their votes.

        :param row: The integer or string name of the row to compare at.

        :param col: The integer or string name of the column to compare at.


        :param createUnknownUser: If True and a username does not exist in this
            object, we will create it first, then do the comparison.  Otherwise
            it throws an exception for unknown users.

        :return: Nothing

        :raises ValueError: If the user does not exist and createUnknownUsers is False.
        '''
        for uname, val in votes.iteritems():
            self.vote(uname, row, col, val, createUnknownUser=createUnknownUser)

    def vote_matrix(self, user_name:str, val=np.ndarray, createUnknownUser:bool=True):
        '''
        Sets the vote matrix for a user

        :param user_name:
        :param val:
        :return:
        '''
        mat = self.matrix(user_name, createUnknownUser=createUnknownUser)
        mat[:,:] = val

    def vote(self, user_name:str, row, col, val:float=0, createUnknownUser:bool=True)->None:
        '''
        Changes a single pairwise value for a single user.

        :param user_name: The string name of the user whose pairwise comparison
            vote you wish to change.

        :param row: The integer or string name of the row to compare at.

        :param col: The integer or string name of the column to compare at.

        :param val: The new pairwise comparison value

        :param createUnknownUser: If True and user_name does not exist in this
            object, we will create it first, then do the comparison.  Otherwise
            it throws an exception for unknown users.

        :return: Nothing

        :raises ValueError: If the user does not exist and createUnknownUsers is False.
        '''
        mat = self.matrix(user_name, createUnknownUser=createUnknownUser)
        row = self.alt_index(row)
        col = self.alt_index(col)
        if row == col:
            # Cannot vote self at all
            raise ValueError("row cannot equal column when voting")
        mat[row, col] = val
        if val == 0:
            mat[col, row] = 0
        else:
            mat[col, row] = 1.0/val

    def unvote(self, user_name:str, row, col, createUnknownUser:bool=True)->None:
        '''
        Unsets a pairwise comparison

        :param user_name: The string name of the user whose pairwise comparison
            vote you wish to unset.

        :param row: The integer or string name of the row to compare at.

        :param col: The integer or string name of the column to compare at.

        :param createUnknownUser: If True and user_name does not exist in this
            object, we will create it first, then do the unset operation.
            Otherwise it throws an exception for unknown users.

        :return: Nothing

        :raises ValueError: If the user does not exist and createUnknownUsers is False.
        '''
        self.vote(user_name, row, col, val=0)

    def usernames(self):
        '''
        :return: A list of the users in this group pairwise comparison object.
        '''
        return list(self.df.index)

    def priority(self, username=None, ptype:PriorityType=None):
        '''
        Calculates the resulting priority for the given user / users.

        :param user_name: The name/names of the users to calculate the priority
            of.  If None, we get the priority of the group average matrix.  If
            it is a string, we get the priority of that user.  If it is a list
            of users, we get the priority of the group average for that list of
            users.

        :param ptype: How should we normalize the resulting priorities
            (if at all).

        :return: A pandas.Series whose indices are the alternative names and
            whose values are the priorities of those alternatives.
        '''
        mat = self.matrix(username)
        rval = self.priority_calc(mat)
        return pd.Series(data=rval, index=self.alts)

    def _repr_html(self, tab="\t"):
        rval = tab+"<ul>\n"
        for user in self.usernames():
            mat = self.matrix(user)
            matstr = tab+"\t"+str(mat)
            matstr = re.sub("\n", "\n"+tab+"\t", matstr)
            rval += tab+"\t"+"<li>"+str(user)+"\n"+matstr+"\n"
        rval += tab+"</ul>"
        return rval

    def data_names(self, append_to=None, post_pend=""):
        '''
        '''
        alt_names = self.alt_names()
        nalts = len(alt_names)
        if append_to is None:
            append_to = []
        for alt1pos in range(nalts):
            for alt2pos in range(alt1pos+1, nalts):
                append_to.append(alt_names[alt1pos]+" vs "+alt_names[alt2pos]+" "+post_pend)
        return append_to

    def alt_names(self):
        '''
        :return: List of string alt names
        '''
        return deepcopy(self.alts)


def add_place(mat):
    '''
    Adds a row and column to the end of a matrix, and makes the last entry 1, rest of the
    added entries are zeroes

    :param mat: The matrix to add an entry to.

    :return: New matrix
    '''
    if mat is None:
        return np.array([[1]])
    nrows = len(mat)
    if nrows == 0:
        return np.array([[1]])
    ncols = len(mat[0])
    rval = np.hstack([mat, [[0]]*nrows])
    rval = np.vstack([rval, [0]*(ncols+1)])
    rval[nrows,ncols]=1
    return rval

def geom_avg_mats(mats)->np.ndarray:
    '''
    Calculates the geometric average of the given matrices.

    :param mats: A list-like object of numpy arrays

    :return: A numpy array that is the geometric average
    '''
    if len(mats) <= 0:
        raise ValueError('Need more then 0 matrices')
    nrows, ncols = mats[0].shape
    rval = np.zeros([nrows, ncols])
    for r in range(nrows):
        for c in range(ncols):
            rval[r,c]=1
            nonzerocount=0
            for mat in mats:
                val = mat[r,c]
                if val != 0:
                    rval[r,c] *= val
                    nonzerocount += 1
            if nonzerocount > 0:
                rval[r,c] = rval[r,c] ** (1.0/nonzerocount)
            else:
                rval[r,c] = 0
    return rval
