'''
Monte Carlo ANP calculations.  The main classes are MCAnp, which can perform simulations on all ANP/AHP
structures, and PrioritySim, which is the resulting structure that a simulation returns
'''
import numpy as np
from pyanp.priority import pri_eigen
from pyanp.prioritizer import PriorityType
from pyanp.direct import Direct
from pyanp.ahptree import AHPTree, AHPTreeNode
from pyanp.pairwise import Pairwise
from copy import deepcopy
import pandas as pd
from scipy.stats import triang


def ascale_mscale(val: (float, int)) -> float:
    """
    Converts an additive scale vote to multiplicative
    :param val: The additive scale vote
    :return: The corresponding multiplicative scale vote
    """
    if val is None:
        return 0
    elif val < 0:
        val = -val
        val += 1
        val = 1.0 / val
        return val
    else:
        return val + 1


def mscale_ascale(val: (float, int)) -> float:
    """
    Converts a multiplicative scale vote to additive
    :param val: The multiplicative scale vote
    :return: The corresponding additive scale vote
    """
    if val == 0:
        return None
    elif val >= 1:
        return val - 1
    else:
        val = 1 / val
        val = val - 1
        return -val


DEFAULT_DISTRIB = triang(c=0.5, loc=-1.5, scale=3.0)


def avote_random(avote):
    """
    Returns a random additive vote in the neighborhood of the additive vote avote
    according to the default disribution DEFAULT_DISTRIB
    """
    if avote is None:
        return None
    raw_val = DEFAULT_DISTRIB.rvs(size=1)[0]
    return avote + raw_val


def mvote_random(mvote):
    """
    Returns a random multiplicative vote in the neighborhhod of the multiplicative vote mvote
    according to the default distribution DEFAULT_DISTRIB.  This is handled by converting
    the multiplicative vote to an additive vote, calling avote_random() and converting the
    result back to an additive vote
    """
    avote = mscale_ascale(mvote)
    rval_a = avote_random(avote)
    rval = ascale_mscale(rval_a)
    return rval


def direct_random(direct, max_percent_chg=0.2) -> float:
    """
    Returns a random direct data value near the value `direct'.  This function
    creates a random percent change, between -max_percent_chg and +max_percent_chg, and
    then changes the direct value by that factor, and returns it.
    """
    pchg = np.random.uniform(low=-max_percent_chg, high=max_percent_chg)
    return direct * (1 + pchg)


class MCAnp:
    def __init__(self):
        # Setup the random pairwise vote generator
        self.pwvote_random = mvote_random
        # Setup the random direct vote generator
        self.directvote_random = direct_random
        # Set the default user to use across the simulation
        # follows the standard from Pairwise class, i.e. it can be a list
        # of usernames, a single username, or None (which means total group average)
        self.username = None
        # What is the pairwise priority calculation?
        self.pwprioritycalc = pri_eigen

    def sim(self, src, dest=None, ptype:PriorityType=None, count:int=None):
        """
        Performs simulation.
        This function calls sim_NAME depending on the class of the src object.
        If the dest object is None, we create a dest object by calling deepcopy().
        In either case, we always return the allocated dest object
        """
        if dest is None:
            dest = deepcopy(src)
        # Which kind of src do we have
        if (count is not None) and (count > 0):
            # Doing multiple counts
            rval = [self.sim(src, dest, ptype, count=None) for i in range(count)]
            return PrioritySim(rval)
        # If we make it here, we are doing a single one
        if isinstance(src, np.ndarray):
            # We are simulating on a pairwise comparison matrix
            return self.sim_pwmat(src, dest, ptype)
        elif isinstance(src, Pairwise):
            # We are simulating on a multi-user pairwise comparison object
            return self.sim_pw(src, dest, ptype)
        elif isinstance(src, AHPTree):
            # We are simulating on an ahp tree object
            return self.sim_ahptree_fill(src, dest, ptype)
        elif isinstance(src, Direct):
            # We are simulating on an ahp direct data
            return self.sim_direct_fill(src, dest, ptype)
        else:
            raise ValueError("Src class is not handled, it is " + type(src).__name__)

    def sim_fill(self, src, dest=None):
        """
        Fills in data on a structure prior to doing the simulation calculations.
        This function calls sim_NAME_fill depending on the class of the src object.
        If the dest object is None, we create a dest object by calling deepcopy().
        In either case, we always return the allocated dest object
        """
        if dest is None:
            dest = deepcopy(src)
        # Which kind of src do we have
        if isinstance(src, np.ndarray):
            # We are simulating on a pairwise comparison matrix
            return self.sim_pwmat_fill(src, dest)
        elif isinstance(src, Pairwise):
            # We are simulating on a multi-user pairwise comparison object
            return self.sim_pw_fill(src, dest)
        elif isinstance(src, AHPTree):
            # We are simulating on an ahp tree object
            return self.sim_ahptree_fill(src, dest)
        elif isinstance(src, Direct):
            # We are simulating on an ahp direct data
            return self.sim_direct_fill(src, dest)
        else:
            raise ValueError("Src class is not handled, it is " + type(src).__name__)

    def sim_pwmat_fill(self, pwsrc: np.ndarray, pwdest: np.ndarray = None) -> np.ndarray:
        """
        Fills in a pairwise comparison matrix with noisy votes based on pwsrc
        If pwsrc is None, we create a new matrix, otherwise we fill in pwdest
        with noisy values based on pwsrc and the self.pwvote_random parameter.
        In either case, we return the resulting noisy matrix
        """
        if pwdest is None:
            pwdest = deepcopy(pwsrc)
        size = len(pwsrc)
        for row in range(size):
            pwdest[row, row] = 1.0
            for col in range(row + 1, size):
                val = pwsrc[row, col]
                if val >= 1:
                    nvote = self.pwvote_random(val)
                    pwdest[row, col] = nvote
                    pwdest[col, row] = 1 / nvote
                elif val != 0:
                    nvote = self.pwvote_random(1 / val)
                    pwdest[col, row] = nvote
                    pwdest[row, col] = 1 / nvote
                else:
                    pwdest[row, col] = nvote
                    pwdest[col, row] = nvote
        return pwdest

    def sim_pwmat(self, pwsrc: np.ndarray, pwdest: np.ndarray = None, pritype:PriorityType=None) -> np.ndarray:
        """
        creates a noisy pw comparison matrix from pwsrc, stores the matrix in pwdest (which
        is created if pwdest is None) calculates the resulting priority and returns that
        """
        pwdest = self.sim_pwmat_fill(pwsrc, pwdest)
        rval = self.pwprioritycalc(pwdest)
        return pd.Series(data=rval, index=['alt_'+str(i) for i in range(len(rval))])

    def sim_pw(self, pwsrc: Pairwise, pwdest: Pairwise, pritype:PriorityType=None) -> np.ndarray:
        """
        Performs a simulation on a pairwise comparison matrix object and returns the
        resulting priorities
        """
        pwdest = self.sim_pw_fill(pwsrc, pwdest)
        return pwdest.priority(self.username, pritype)
        # mat = pwdest.matrix(self.username)
        # rval = self.pwprioritycalc(mat)
        # return rval

    def sim_pw_fill(self, pwsrc: Pairwise, pwdest: Pairwise = None) -> Pairwise:
        """
        Fills in the pairwise comparison structure of pwdest with noisy pairwise data from pwsrc.
        If pwdest is None, we create one first, then fill in.  In either case, we return the pwdest
        object with new noisy data in it.
        """
        if pwdest is None:
            pwdest = deepcopy(pwsrc)
        for user in pwsrc.usernames():
            srcmat = pwsrc.matrix(user)
            destmat = pwdest.matrix(user)
            self.sim_pwmat_fill(srcmat, destmat)
        return pwdest

    def sim_direct_fill(self, directsrc: Direct, directdest: Direct = None) -> Direct:
        """
        Fills in the direct data structure of directdest with noisy data from directsrc.
        If directdest is None, we create on as a deep copy of directsrc, then fill in.
        In either case, we return the directdest object with new noisy data in it.
        """
        if directdest is None:
            directdest = deepcopy(directsrc)
        for altpos in range(len(directdest)):
            orig = directsrc[altpos]
            newvote = self.directvote_random(orig)
            directdest.data[altpos] = newvote
        return directdest

    def sim_direct(self, directsrc: Direct, directdest: Direct = None, pritype:PriorityType=None) -> np.ndarray:
        """
        Simulates for direct data
        """
        directdest = self.sim_direct_fill(directsrc, directdest)
        return directdest.priority(ptype=pritype)

    def sim_ahptree_fill(self, ahpsrc: AHPTree, ahpdest: AHPTree) -> AHPTree:
        """
        Fills in the ahp tree structure of ahpdest with noisy data from ahpsrc.
        If ahpdest is None, we create one as a deepcopy of ahpsrc, then fill in.
        In either case, we return the ahpdest object with new noisy data in it.
        """
        if ahpdest is None:
            ahpdest = deepcopy(ahpsrc)
        self.sim_ahptreenode_fill(ahpsrc.root, ahpdest.root)
        return ahpdest

    def sim_ahptreenode_fill(self, nodesrc: AHPTreeNode, nodedest: AHPTreeNode) -> AHPTreeNode:
        """
        Fills in data in an AHPTree
        """
        # Okay, first we fill in for the alt_prioritizer
        if nodesrc.alt_prioritizer is not None:
            self.sim_fill(nodesrc.alt_prioritizer, nodedest.alt_prioritizer)
        # Now wefill in the child prioritizer
        if nodesrc.child_prioritizer is not None:
            self.sim_fill(nodesrc.child_prioritizer, nodedest.child_prioritizer)
        # Now for each child, fill in
        for childsrc, childdest in zip(nodesrc.children, nodedest.children):
            self.sim_ahptreenode_fill(childsrc, childdest)
        # We are done, return the dest
        return nodedest

    def sim_ahptree(self, ahpsrc:AHPTree, ahpdest:AHPTree=None, pritype:PriorityType=None) -> np.ndarray:
        """
        Perform the actual simulation
        """
        ahpdest = self.sim_ahptree_fill(ahpsrc, ahpdest)
        return ahpdest.priority(ptype=pritype)


class PrioritySim:
    """
    Represents a priority simulation event
    """
    def __init__(self, sims):
        """
        Create a new PrioritySim class
        :param sims: Should be something that pd.DataFrame(data=sims) will work with
        """
        self.df = pd.DataFrame(data=sims)


    def __str__(self):
        return str(self.df)