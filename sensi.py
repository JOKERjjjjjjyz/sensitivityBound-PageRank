import numpy as np
import random
import math


def jyz_brute(n,D,alpha):
    # jyz的暴力放缩版本
    z = []
    z.append(0)
    z.append(0)
    a = 0
    b = 2/D * (1-alpha)
    # b = 1 - alpha
    aa = a
    bb = b
    for i in range(2, n):
        aa = a
        bb = b
        a = (1 - alpha) * ((D-2) * aa / D  + 2 * bb / D) #非x y的user,item
        b = min(2 * (1 - alpha), (1 - alpha) * (2 / D + aa))#x,y
        # b = min(2*(1-alpha),(1-alpha) * (1+aa))
        z.append(a)
    return a


