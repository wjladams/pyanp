from pyanp.exmats import _add_pairwise, _add_supermatrix
import numpy as np
author = "Dr. Bill Adams <wjladams@gmail.com>"



_add_pairwise("2x2ex1", "2x2 pairwise matrix with double value", ['2'],
              np.array([
                 [1, 2],
                 [1/2, 1]
             ]), author)

_add_pairwise("3x3_236", "2 x, 3x, 6x consistent matrix", ["236", "consistent"],
              np.array([
                 [1, 2, 6],
                 [1/2, 1, 3],
                 [1/6, 1/3, 1]
             ]), author)

_add_pairwise("3x3_235", "2 x, 3x, 5x inconsistent matrix", ["235", "consistent"],
              np.array([
                 [1, 2, 6],
                 [1/2, 1, 3],
                 [1/6, 1/3, 1]
             ]), author)







_add_supermatrix('4x4ex1', 'A standard 4x4 example', ['4', ' FuLl'],
                 np.array([
                    [0.2, 0.25, 0.05, 0.18],
                    [0.3, 0.3, 0.25, 0.07],
                    [0.4, 0.3, 0.5, 0.3],
                    [0.1, 0.15, 0.2, 0.45]
                ]), author)

_add_supermatrix('3x3ex1', 'A simple 3x3 example', ['3'],
                 np.array([
                    [0.3, 0.2, 0.0],
                    [0.1, 0.5, 0.0],
                    [0.6, 0.3, 1.0]
                ]), author)

#print("Loading for "+author+"\n")