'''
Class for all rating related things.
'''

from pyanp.prioritizer import Prioritizer, PriorityType

import re
from enum import Enum
import pandas as pd
import numpy as np
from pyanp.general import islist

__CLEANER = re.compile('\\s+')


def clean_word(word):
    word = word.strip().lower()
    word = __CLEANER.sub(string=word, repl=' ')
    return word


class WordEvalType(Enum):
    LINEAR = 1
    EXPONENTIAL = 2
    MANUAL = 3


class WordEval:
    def __init__(self, vals):
        self.names_to_synonyms = vals
        self.keys = list(vals.keys())
        self.lookup_synonym = {}
        self.base = 0.9
        self.type = WordEvalType.LINEAR
        for key, synonyms in vals.items():
            self.lookup_synonym[key] = key
            for synonym in synonyms:
                self.lookup_synonym[synonym] = key

    def get_key(self, word):
        if word in self.lookup_synonym:
            return self.lookup_synonym[word]
        else:
            return None

    def keys_match_score(self, word_list):
        keys_used = set()
        none_count = 0
        for word in word_list:
            if word is not None and isinstance(word, str) and len(word) > 0:
                key = self.get_key(word)
                if key is not None:
                    keys_used.add(key)
                else:
                    none_count += 1
        percent = len(keys_used) / len(self.keys)
        rval = percent - none_count
        return rval

    def eval(self, word):
        if isinstance(word, pd.Series):
            data = [self.eval(w) for w in word]
            rval = pd.Series(data=data, index = word.index, dtype=float)
            return rval
        word = clean_word(word)
        key = self.get_key(word)
        if key is None:
            return None
        nitems = len(self.names_to_synonyms)
        if self.type is WordEvalType.LINEAR:
            index = self.keys.index(key)
            # print(index)
            if nitems <= 1:
                return nitems
            else:
                return (nitems - 1 - index) / (nitems - 1)
        elif self.type is WordEvalType.EXPONENTIAL:
            index = self.keys.index(key)
            if nitems <= 1:
                return nitems
            else:
                return self.base ** index
        else:
            raise ValueError("Have not done manual case yet")

        pass




STD_WORD_EVALUATORS = {
    'hml': WordEval({
        'high': ('h', 'hi'),
        'medium': ('medi', 'med', 'me', 'm'),
        'low': ('lowe', 'lo', 'l')
    }),
    'vhhmlvl':WordEval({
        'very high': (
        'ver high', 'vy high', 'v high', 'vhigh', 'very hi', 'very h', 'v h', 'vh'),
        'high': ('hig', 'hi', 'h'),
        'medium': (
        'mediu', 'med', 'me', 'm', 'okay', 'ok', 'o', 'average', 'aver', 'avg'),
        'low': ('lo', 'l', 'lw', 'bad', 'bd', 'not high', 'not hi', 'not h'),
        'very low': ('ver low', 'vy low', 'v low', 'vlow', 'vlo', 'vl', 'v lo')
    }),
    'abcdf': WordEval({
        'a': (),
        'b': (),
        'c': (),
        'd': (),
        'f': ('e')
    }),
    'egobvb': WordEval({
        'excellent':('excel', 'excl', 'exc','ex', 'e', '++', 'very good', 'vy good'
                     'vy gd', 'vy g', 'v good', 'vgood', 'vg', 'great'),
        'good':('g', 'gd'),
        'okay':('ok', 'equal', '=', 'equals', 'eq'),
        'bad':('b', 'bd', 'not good', 'notgood', 'not g', 'ngood', 'ng'),
        'very bad':('horrible', 'horrid', 'v bad', 'vbad', 'veryb', 'verybad',
                    'vb', 'v b')
    })
}

def best_std_word_evaluator(list_of_words, return_name=True)->WordEval:
    scores = {name:weval.keys_match_score(list_of_words)   for name,weval in STD_WORD_EVALUATORS.items()}
    rval = max(scores, key=scores.get)
    if return_name:
        return rval
    else:
        return STD_WORD_EVALUATORS[rval]


class Rating(Prioritizer):
    '''
    Represents rating a full group of items for a group of users.
    The data is essentially a dataframe, and for each column, a
    WordEval object to evaluate that to scores.
    '''
    def __init__(self):
        self.df = pd.DataFrame()
        self.word_evals = {}

    def is_alt(self, alt:str)->bool:
        return alt in self.df.columns

    def nusers(self):
        return len(self.df.index)

    def nalts(self):
        return len(self.df.columns)

    def add_alt(self, alt_name:str, ignore_existing=True):
        if islist(alt_name):
            for alt in alt_name:
                self.add_alt(alt)
            return
        if self.is_alt(alt_name):
            if ignore_existing:
                # We already have an alternative like this, we were told
                # to ignore this.
                return None
            else:
                raise ValueError("Already have an alt name "+alt_name)
        else:
            self.df[alt_name] = [None]*self.nusers()
            self.word_evals[alt_name] = None

    def add_user(self, uname):
        if islist(uname):
            for un in uname:
                self.add_user(un)
            return
        # Add alt for singular
        if uname in self.df.index:
            # Already existed
            return
        else:
            self.df.loc[uname,:] = [None] * self.nalts()

    def user_names(self):
        return list(self.df.index)

    def alt_names(self):
        return list(self.df.columns)


    def vote_column(self, alt_name, votes):
        if not self.is_alt(alt_name):
            raise ValueError("No such alternative "+alt_name)
        self.df[alt_name] = votes

    def priority(self, username=None, ptype:PriorityType=None):
        values = self.vote_values(username=username)
        rval = values.mean()
        for key, val in rval.iteritems():
            if np.isnan(val):
                rval[key]=0
        return rval

    def vote_values(self, username=None, alt_name=None):
        if username is None:
            df = self.df
        else:
            df = self.df.loc[username,:]
        if alt_name is not None:
            votes = df[alt_name]
            weval = self.word_evals[alt_name]
            if weval is None:
                weval = best_std_word_evaluator(votes, return_name=False)
            if all([isinstance(vote, float) and np.isnan(vote) for vote in votes]):
                return pd.Series(index=self.user_names())
            else:
                return weval.eval(votes)
        else:
            rval = pd.DataFrame(index=self.user_names())
            for alt in self.alt_names():
                rval[alt] = self.vote_values(username=username, alt_name=alt)
            return rval