'''
This is where we store standard examples of pairwise and supermatrices.

'''
import numpy as np
import pandas as pd

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
    SUPERMATRIX_EXS.loc[name]=[description, keywords,  mat, size]


def islist(val):
    '''
    Simple function to check if a value is list like object
    :param val:
    :return:
    '''
    if val is None:
        return False
    elif isinstance(val, str):
        return False
    else:
        return hasattr(val, "__len__")


def matrix_matching(df=SUPERMATRIX_EXS, description=None, keywords=None, size=None, author=None):
    '''
    Finds matrices that match search criteria
    :param df: The dataframe to search through, either SUPERMATRIX_EXS or PAIRWISE_EXS
    :param description: A substring to search through description
    :param keywords: A list of keywords to find
    :param size: The size
    :param author: The contributing author, a substring search
    :return: List of indices of the matches in the given dataframe
    '''
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
    :return:
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
    :return:
    '''
    global PAIRWISE_EXS
    if name is not None:
        return PAIRWISE_EXS.loc[name,'matrix']
    else:
        indices = matrix_matching(PAIRWISE_EXS, description, keywords, size, author)
        return PAIRWISE_EXS.loc[indices, 'matrix']


_add_pairwise("2x2ex1", "2x2 pairwise matrix with double value", ['2'],
              np.array([
                 [1, 2],
                 [1/2, 1]
             ]))

_add_pairwise("3x3_236", "2 x, 3x, 6x consistent matrix", ["236", "consistent"],
              np.array([
                 [1, 2, 6],
                 [1/2, 1, 3],
                 [1/6, 1/3, 1]
             ]))

_add_pairwise("3x3_235", "2 x, 3x, 5x inconsistent matrix", ["235", "consistent"],
              np.array([
                 [1, 2, 6],
                 [1/2, 1, 3],
                 [1/6, 1/3, 1]
             ]))


_add_supermatrix('4x4ex1', 'A standard 4x4 example', ['4', ' FuLl'],
                 np.array([
                    [0.2, 0.25, 0.05, 0.18],
                    [0.3, 0.3, 0.25, 0.07],
                    [0.4, 0.3, 0.5, 0.3],
                    [0.1, 0.15, 0.2, 0.45]
                ]))

_add_supermatrix('3x3ex1', 'A simple 3x3 example', ['3'],
                 np.array([
                    [0.3, 0.2, 0.0],
                    [0.1, 0.5, 0.0],
                    [0.6, 0.3, 1.0]
                ])
                 )