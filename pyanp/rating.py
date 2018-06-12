'''
Class for all rating related things.
'''

from pyanp.prioritizer import Prioritizer, PriorityType

import re
from enum import Enum
import pandas as pd
import numpy as np
from pyanp.general import islist


__SPACE_REGEXP = re.compile('\\s+')


def clean_word(word:str)->str:
    '''
    Cleans a word before subjecting it to ratings lookup

    :param word: The word to clean.

    :return: The sanitized word
    '''
    word = word.strip().lower()
    word = __SPACE_REGEXP.sub(string=word, repl=' ')
    return word


class WordEvalType(Enum):
    '''
    What kind of WordEval will we use.
    '''
    LINEAR = 1
    EXPONENTIAL = 2
    MANUAL = 3


class WordEval:
    '''
    Information for a Word Evaluator, i.e. a function that inputs a word and
    outputs a numeric value.
    '''
    def __init__(self, vals):
        self.names_to_synonyms = vals
        self.keys = list(vals.keys())
        self.lookup_synonym = {}
        self.base = 0.9
        self.type = WordEvalType.LINEAR
        self.values = {}
        for key, synonyms in vals.items():
            self.lookup_synonym[key] = key
            for synonym in synonyms:
                if isinstance(synonym, (float, int)):
                    # You are telling us the value of this key, not a
                    # a synonym
                    self.values[key] = synonym
                else:
                    # This is actually a synonym
                    self.lookup_synonym[synonym] = key

    def get_key(self, word):
        '''
        Find the key word for this word.  A WordEval has a list of words that
        represent different levels/numerical values.  Those words are called
        keys.  In addition, each key has a list of synonyms.  For instance
        the keyword "high" might have a synonym "hi" or "h".  In that case
        get_key("hi") would return "high".

        :param word: The word to look up a synonym for.

        :return: The key if this word is a key or a synonym.  If it is not a
            synonym or key, we return None.
        '''
        if word in self.lookup_synonym:
            return self.lookup_synonym[word]
        else:
            return None

    def keys_match_score(self, word_list):
        '''
        This function tells us how well this WordEval interprets a list of
        words.  It is used for searhcing through the "standard list" of words
        to find the best match for a data set.

        :param word_list: The list-like of words to see how we can match.

        :return: A score <= 1.  A positive number means no missing words, i.e.
            every word in word_list has a value in this WordEval object.
            The larger number means our word_list uses more of the names in this
            WordEval object.
        '''
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
        '''
        Evaluates a word, or a pandas.Series of words.

        :param word: The string word to evaluate to a number, or a pandas.Series
            of data.

        :return: The float value if we can evaluate, or None if a single value
            is passed in.  If the word was actually a pandas.Series, we return
            a pandas.Series with the same index.
        '''
        if isinstance(word, pd.Series):
            data = [self.eval(w) for w in word]
            rval = pd.Series(data=data, index = word.index, dtype=float)
            return rval
        word = clean_word(word)
        key = self.get_key(word)
        if key is None:
            return None
        if key in self.values:
            # We have it manually set
            return self.values[key]
        # If we make it here, we have to work the other way round
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



## Does this do anything feverish?
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

def best_std_word_evaluator(list_of_words, return_name=True):
    '''
    Finds the WordEval in STD_WORD_EVALUATOR that best matches the list of words

    :param list_of_words: The list of words to look for best matches of

    :param return_name: Should we return the best WordEval or its name in the
        STD_WORD_EVALUATOR.

    :return: The name of the best match, or the best match WordEval
    '''
    scores = {name:weval.keys_match_score(list_of_words)   for name,weval in STD_WORD_EVALUATORS.items()}
    rval = max(scores, key=scores.get)
    if return_name:
        return rval
    else:
        return STD_WORD_EVALUATORS[rval]


class Rating(Prioritizer):
    '''
    Represents rating a full group of alternatives for a group of users.
    The data is essentially a dataframe and a WordEval object to
    evaluate that to scores.
    '''
    def __init__(self):
        self.df = pd.DataFrame()
        self.word_eval = None

    def is_alt(self, alt:str)->bool:
        '''
        Tells if the item is an alternative

        :param alt: The name of the alternative to check for.

        :return: True/False
        '''
        return alt in self.df.columns

    def nusers(self)->int:
        '''
        The number of users in this system.

        :return: The number of users
        '''
        return len(self.df.index)

    def nalts(self)->int:
        '''
        :return: The number of alternatives in this system.
        '''
        return len(self.df.columns)

    def add_alt(self, alt_name, ignore_existing=True):
        '''
        Adds an alternative/s, by name

        :param alt_name: A str name, or a list of names to add.
        :param ignore_existing: If True and we try to add an existing alternative
            we simply skip by, otherwise we throw an error.

        :return: Nothing
        '''
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

    def add_user(self, uname):
        '''
        Adds one or more uses to this system.

        :param uname: The str name of the user to add, or a list of str names
            of users to add.

        :return: Nothing
        '''
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
        '''
        :return: A list of str names of users in this system.  Ordered as the
            data in the ratings votes are ordered (the rows).
        '''
        return list(self.df.index)

    def alt_names(self):
        '''
        :return: A list of str alternative names in this system.  Ordered as the
            data in the ratings votes are ordered (columns).
        '''
        return list(self.df.columns)


    def vote_column(self, alt_name, votes, createUnknownUsers=True):
        '''
        Specifies all votes (across all users) for a specific alternative.

        :param alt_name: The name of the alternative to set the data for

        :param votes: Should either be a list with self.nusers() items, or a
            pandas.Series or dict with usernames as index.

        :param createUnknownUsers: If True and unknown users appear in the index
            of votes, we will create those users before trying to do the
            assignment.

        :return: Nothing
        '''
        if not self.is_alt(alt_name):
            raise ValueError("No such alternative "+alt_name)
        if createUnknownUsers:
            if isinstance(votes, pd.Series):
                for uname in votes.keys():
                    if not self.is_user(uname):
                        self.add_user(uname)
        self.df[alt_name] = votes

    def priority(self, username=None, ptype:PriorityType=None):
        '''
        Calculates the alternative priority for the specified user/users and the
        given normalizer type.

        :param username: The name (this of names) of the user (users) to get
            the overall priority of.  If None, then we return the total group
            average.

        :param ptype: How should we normalize?

        :return: A pandas.Series whose index is self.alt_names() and whose values
            are the priorities.
        '''
        values = self.vote_values(username=username)
        rval = values.mean()
        for key, val in rval.iteritems():
            if np.isnan(val):
                rval[key]=0
        if ptype is None:
            return rval
        else:
            return ptype.apply(rval)

    def vote_values(self, username=None, alt_name=None):
        '''
        Gets the numeric vote values for the given user/alternative (or whole
        column or dataframe).

        :param username: If None, we get the values for all users.  If a list
            get the values for each user in the list, or it could just be a single
            username.

        :param alt_name: Either None, meaing get it for all alternatives, or
            a single alternative name (to get one column).

        :return: If username=None and alt_name=None, returns a pandas.DataFrame
            of the numeric values.  Otherwise returns a pandas.Series of values
            as the result.
        '''
        if username is None:
            df = self.df
        else:
            df = self.df.loc[username,:]
        if alt_name is not None:
            votes = df[alt_name]
            weval = self.word_eval
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

    def is_user(self, uname:str):
        '''
        :param uname: The name of the user to check for
        :return: True/False if the given user exists in the system.
        '''
        return uname in self.df.index

    def set_word_eval(self, param):
        '''
        Sets the WordEval object

        :param param: This could either be a WordEval object, or a something
            that WordEval(param) would work with

        :return: None
        '''
        if param is None or isinstance(param, WordEval):
            self.word_eval=param
        else:
            self.word_eval=WordEval(param)
