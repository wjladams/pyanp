'''
Generally useful math and other functions.
'''

import numpy as np
import pandas as pd

def linear_interpolate(xs, ys, x):
    '''
    Piecewise linear interpolation between a bunch of x and y coordinates.

    :param xs: Increasing x values (no dupes)

    :param ys: The y values

    :param x: The x value to linearly interpolate at

    :return: if x < xs[0], returns ys[0], if x > xs[-1], returns ys[-1]
        else linearly interpolates.
    '''
    if x <= xs[0]:
        return ys[0]
    elif x >= xs[-1]:
        return ys[-1]
    #Okay we are in between, find first xs we are in between
    for i in range(1, len(xs)):
        if x <= xs[i]:
            slope = (ys[i]-ys[i-1])/(xs[i]-xs[i-1])
            return ys[i-1]+(x-xs[i-1])*slope
    #Should never make it here
    raise ValueError("linear interpolation failure")


def islist(val):
    '''
    Simple function to check if a value is list like object

    :param val: The object to check its listiness.

    :return: Boolean True/False
    '''
    if val is None:
        return False
    elif isinstance(val, str):
        return False
    else:
        return hasattr(val, "__len__")


def get_matrix(fname_or_df)->np.ndarray:
    '''
    Returns a dataframe from a csv/excel filename (or simply returns the
    dataframe if it is passed as input

    :param fname_or_df: The file name to get as a dataframe, or a dataframe
    (in which case that param is returned)

    :return: The dataframe
    '''
    if isinstance(fname_or_df, str):
        fname = fname_or_df.lower()
        rval = None
        if fname.endswith(".csv"):
            rval = pd.read_csv(fname_or_df, index_col=0)
        elif fname.endswith(".xls") or fname.endswith(".xlsx"):
            rval = pd.read_excel(fname_or_df)
        # I need to know if the columns were without headers
        try:
            fvals = [float(v) for v in rval.columns]
            # column names were all numbers, that indicates there was no header
            if fname.endswith(".csv"):
                rval = pd.read_csv(fname_or_df, header=None)
            elif fname.endswith(".xls") or fname.endswith(".xlsx"):
                rval = pd.read_excel(fname_or_df, header=None)
        except:
            pass
    elif isinstance(fname_or_df, pd.DataFrame):
        rval = fname_or_df
    elif isinstance(fname_or_df, np.ndarray):
        return fname_or_df
    else:
        raise ValueError("Cannot handle your stuff")
    return rval.values.copy()