'''
This is where we store standard examples of pairwise and supermatrices.

'''
import numpy as np
import pandas as pd

PAIRWISE_EXS = pd.DataFrame(columns=["description", "keywords", "matrix", "size"])
SUPERMATRIX_EXS = pd.DataFrame(columns=["description", "keywords", "matrix", "size"])


def clean_keywords(keywords):
    if keywords is None:
        return None
    if isinstance(keywords, str):
        #Only one keyword
        return keywords.strip().lower()
    elif islist(keywords):
        return [clean_keywords(keyword) for keyword in keywords]
    else:
        return keywords


def add_pairwise(name, description, keywords, mat):
    global PAIRWISE_EXS
    if name in PAIRWISE_EXS.index:
        raise ValueError("Already had a pairwise example with that name")
    size = len(mat)
    keywords = clean_keywords(keywords)
    PAIRWISE_EXS.loc[name]=[description, keywords, mat, size]

def add_supermatrix(name, description, keywords, mat):
    global SUPERMATRIX_EXS
    if name in SUPERMATRIX_EXS.index:
        raise ValueError("Already had a supermatrix example with that name")
    size = len(mat)
    keywords = clean_keywords(keywords)
    SUPERMATRIX_EXS.loc[name]=[description, keywords,  mat, size]


def islist(val):
    if val is None:
        return False
    elif isinstance(val, str):
        return False
    else:
        return hasattr(val, "__len__")


def matrix_matching(df=SUPERMATRIX_EXS, description=None, keywords=None, size=None):
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
    return indices


def supermatrix_ex(name=None, description=None, keywords=None, size=None):
    global SUPERMATRIX_EXS
    if name is not None:
        return SUPERMATRIX_EXS.loc[name,'matrix']
    else:
        indices = matrix_matching(SUPERMATRIX_EXS, description, keywords, size)
        return SUPERMATRIX_EXS.loc[indices, 'matrix']

def pairwisematrix_ex(name=None, description=None, keywords=None, size=None):
    global PAIRWISE_EXS
    if name is not None:
        return PAIRWISE_EXS.loc[name,'matrix']
    else:
        indices = matrix_matching(PAIRWISE_EXS, description, keywords, size)
        return PAIRWISE_EXS.loc[indices, 'matrix']


add_pairwise("2 2x2", "2x2 pairwise matrix with double value", '2',
             np.array([
                 [1, 2],
                 [1/2, 1]
             ]))

add_pairwise("236", "2 x, 3x, 6x consistent matrix", "236, consistent",
             np.array([
                 [1, 2, 6],
                 [1/2, 1, 3],
                 [1/6, 1/3, 1]
             ]))


add_supermatrix('4x4ex1', 'A standard 4x4 example', ['4', ' FuLl'],
                np.array([
                    [0.2, 0.25, 0.05, 0.18],
                    [0.3, 0.3, 0.25, 0.07],
                    [0.4, 0.3, 0.5, 0.3],
                    [0.1, 0.15, 0.2, 0.45]
                ]))

add_supermatrix('3x3ex1', 'A simple 3x3 example', [3],
                np.array([
                    [0.3, 0.2, 0.0],
                    [0.1, 0.5, 0.0],
                    [0.6, 0.3, 1.0]
                ])
                )