from pyanp.exmats import _add_pairwise, _add_supermatrix
import numpy as np
author = "Dr. Elena Rokou <erokou@gmail.com>"

#print("Loading for "+author+"\n")
_add_pairwise("ex2x2-1.1", "2x2 pairwise matrix with double value", ['2'],
              np.array([
                 [1, 5],
                 [1/5, 1]
             ]), author)
