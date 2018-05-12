'''
Group pairwise object and calculations
'''

import numpy as np
import pandas as pd
from pyanp.general import islist
from pyanp.priority import pri_eigen

class Pairwise:
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


    def is_user(self, user_name):
        return user_name in self.df.index

    def nalts(self):
        return len(self.alts)

    def _blank_pairwise(self):
        '''
        Creates a blank pairwise comparison for the right number of alts
        :return:
        '''
        nalts = self.nalts()
        return np.identity(nalts)

    def add_user(self, user_name):
        if self.is_user(user_name):
            raise ValueError("User "+user_name+" already existed")
        ncols = len(self.df.columns)
        data = [None]*(ncols-1)+[self._blank_pairwise()]
        self.df.loc[user_name] = data

    def matrix(self, user_name):
        if user_name is None:
            user_name = self.usernames()
        if isinstance(user_name, (str)):
            # We are just doing a single user
            if not self.is_user(user_name):
                raise ValueError("No such user " + user_name)
            return self.df.loc[user_name, "Matrix"]
        else:
            mats = [self.df.loc[user, 'Matrix'] for user in user_name]
            return geom_avg_mats(mats)

    def alt_index(self, alt_name_or_index):
        if isinstance(alt_name_or_index, (int)):
            return alt_name_or_index
        if alt_name_or_index not in self.alts:
            raise ValueError("No such alt "+alt_name_or_index)
        return self.alts.index(alt_name_or_index)

    def vote(self, user_name, row, col, val=0):
        mat = self.matrix(user_name)
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

    def unvote(self, user_name, row, col):
        self.vote(user_name, row, col, val=0)

    def usernames(self):
        return list(self.df.index)

    def priority(self, user_name):
        mat = self.matrix(user_name)
        return self.priority_calc(mat)


def geom_avg_mats(mats):
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