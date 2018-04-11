'''
Generally useful math and other functions.
'''

def linear_interpolate(xs, ys, x):
    '''

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
    :param val:
    :return:
    '''
    if val is None:
        return False
    elif isinstance(val, str):
        return False
    else:
        return hasattr(val, "__len__")

