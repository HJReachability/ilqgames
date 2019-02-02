import torch 
import numpy as np

def quadratize(c, x0, u0):
    """
    Compute the quadratic approximation of the cost objective
    for a given state `x0` and `u0.` Outputs `Q` and `f` of 
    the following equation

        c(x,u) = c(x0,u0) + f^Tz + 0.5z^T Qz

    where 
        z = [x-x0; u-u0]
    """
    pass
