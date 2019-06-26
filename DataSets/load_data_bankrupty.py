import pandas as pd
import numpy as np


def load():
    X,Y = [],[]
    file = open(r"C:\Users\Geekzone\IdeaProjects\untitled10\Fichiers\Qualitative_Bankruptcy.data.txt")
    L = file.readlines()
    for i in L:
        S = i.split(",")
        M = []
        for k in S[:-1]:
            M.append(k)
        X.append(M)
        Y.append(S[-1].strip())
    print(X)
    print(Y)
    return np.array(X),np.array(Y)


load()