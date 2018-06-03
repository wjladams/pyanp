'''
This is where we store standard examples of pairwise and supermatrices.

'''
import pandas as pd
from pyanp.general import islist

# The dataframe storing the pairwise example matrices
PAIRWISE_EXS = pd.DataFrame(columns=["description", "keywords", "matrix", "size", "author"])
# The dataframe storing the supermatrix examples matrices
SUPERMATRIX_EXS = pd.DataFrame(columns=["description", "keywords", "matrix", "size", "author"])


def _clean_keywords(keywords):
    '''
    Cleans keywords for example matrices, to make sure they are easily searched

    :param keywords: The keywords sent to '_add_pairwise' or '_add_supermatrix'

    :return: A cleansed version, stripped and to lowercase, among other things.
    '''
    if keywords is None:
        return []
    if isinstance(keywords, str):
        #Only one keyword
        return [keywords.strip().lower()]
    elif islist(keywords):
        return [_clean_keywords(keyword) for keyword in keywords]
    else:
        return keywords


def _add_pairwise(name, description, keywords, mat, author=''):
    '''
    Adds a new pariwise example

    :param name: The unique id of this pairwise matrix example.

    :param description: A stringy description of the example

    :param keywords: A list of stringy keywords

    :param mat: The numpy matrix

    :param author: Optional author email

    :return:
    '''
    global PAIRWISE_EXS
    if name in PAIRWISE_EXS.index:
        raise ValueError("Already had a pairwise example with that name")
    size = len(mat)
    keywords = _clean_keywords(keywords)
    PAIRWISE_EXS.loc[name]=[description, keywords, mat, size, author]

def _add_supermatrix(name, description, keywords, mat, author=''):
    '''
    Adds a new supermatrix example

    :param name: Unique string id of the example

    :param description: A string description

    :param keywords: A list of string keywords

    :param mat: The numpy matrix

    :param author: Optional string email of the contributor/author

    :return:
    '''
    global SUPERMATRIX_EXS
    if name in SUPERMATRIX_EXS.index:
        raise ValueError("Already had a supermatrix example with that name")
    size = len(mat)
    keywords = _clean_keywords(keywords)
    SUPERMATRIX_EXS.loc[name]=[description, keywords,  mat, size, author]


def matrix_matching(df=None, description=None, keywords=None, size=None, author=None):
    '''
    Finds matrices that match search criteria

    :param df: The dataframe to search through, either SUPERMATRIX_EXS or PAIRWISE_EXS
        if None we use SUPERMATRIX_EXS

    :param description: A substring to search through description

    :param keywords: A list of keywords to find

    :param size: The size

    :param author: The contributing author, a substring search

    :return: List of indices of the matches in the given dataframe
    '''
    if df is None:
        global SUPERMATRIX_EXS
        df = SUPERMATRIX_EXS
    indices = list(df.index)
    if description is not None:
        description = description.lower()
        indices = [index for index in indices if description in
                   df.loc[index, 'description'].lower()]
    if keywords is not None:
        if islist(keywords):
            for keyword in keywords:
                keyword = str(keyword).lower().strip()
                indices = [i for i in indices if keyword in df.loc[i, 'keywords']]
        elif isinstance(keywords, str):
            keywords = str(keywords).lower().strip()
            indices = [i for i in indices if keywords in df.loc[i, 'keywords']]
    if size is not None:
        indices = [i for i in indices if size == df.loc[i, 'size']]
    if author is not None:
        author = author.strip().lower()
        indices = [i for i in indices if author in df.loc[i, 'keywords']]
    return indices


def supermatrix_ex(name=None, description=None, keywords=None, size=None, author=None):
    '''
    Find the supermatrix example that matches the conditions

    :param name: If not None, we find the single matrix with this name/id

    :param description: Substring search in description

    :param keywords: exact match keywords

    :param size: exact match size

    :param author: substring search author

    :return: A single numpy.array item if only one matches the constraint, or a panda.Series
        indexed by name or the resulting numpy.array s.
    '''
    global SUPERMATRIX_EXS
    if name is not None:
        return SUPERMATRIX_EXS.loc[name,'matrix']
    else:
        indices = matrix_matching(SUPERMATRIX_EXS, description, keywords, size, author)
        return SUPERMATRIX_EXS.loc[indices, 'matrix']

def pairwisematrix_ex(name=None, description=None, keywords=None, size=None, author=None):
    '''
    Find the pairwise matrix example that matches the conditions

    :param name: If not None, we find the single matrix with this name/id

    :param description: Substring search in description

    :param keywords: exact match keywords

    :param size: exact match size

    :param author: substring search author

    :return: A single numpy.array item if only one matches the constraint, or a panda.Series
        indexed by name or the resulting numpy.array s.
    '''
    global PAIRWISE_EXS
    if name is not None:
        return PAIRWISE_EXS.loc[name,'matrix']
    else:
        indices = matrix_matching(PAIRWISE_EXS, description, keywords, size, author)
        return PAIRWISE_EXS.loc[indices, 'matrix']




from pyanp.exmatcontrib import *