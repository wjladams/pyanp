'''
Example matrices
'''

import numpy as np

PAIRWISE_EXS = [
    dict(name='Something', keywords=['a', 'b','c'], description="Something more",
         matrix = np.array([
             [1, 2, 3],
             [1/2, 1, 6],
             [1/3, 1/6, 1]
         ])
         ),
    dict(name='2x2', keywords=['toy'], description="Simple 2x2 example",
         matrix=np.array([
             [1, 2],
             [1 / 2, 1]
         ])
         ),

]

SUPERMATRIX_EXS = [
    ## Follow same format as PAIRWISE_EXS
]



def names_equal(n1, n2):
    return n1.lower() == n2.lower()


def get_ex_byname(name, type = "pw"):
    list_of_matrices = PAIRWISE_EXS
    type = type.lower()
    if type.find("super"):
        list_of_matrices = SUPERMATRIX_EXS
    rval = [info['matrix'] for info in list_of_matrices if names_equal(info['name'], name)]
    if len(rval) == 1:
        return rval[0]
    else:
        return rval
