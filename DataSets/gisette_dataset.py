import requests
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

from Destiny.Evaluateur_Precision import Evaluateur_Precision


def load_gisette_dataset():
    r = open (r"C:\Users\Geekzone\Desktop\Hyper Heuristique PFE Crédit Scoring\Hyper Heuristique for Credit Scoring\Destiny\DataSets\gisette_train.data.txt")
    L = r.readlines()
    X = []
    for i in L:
        K = np.array (i.strip().split (' ')).astype ("int")
        X.append (K)
    X = np.array(X)
    r = open (
        r"C:\Users\Geekzone\Desktop\Hyper Heuristique PFE Crédit Scoring\Hyper Heuristique for Credit Scoring\Destiny\DataSets\gisette_train.labels.txt")
    L = r.readlines ()
    Y = []
    for i in L:
        K = np.array (i.strip ().split (' ')).astype ("int")
        Y.append (K[0])
    Y = np.array (Y)
    return X,Y



def load_test_dataset():
    r = open (r"C:\Users\Geekzone\Desktop\Hyper Heuristique PFE Crédit Scoring\Hyper Heuristique for Credit Scoring\Destiny\DataSets\gisette_test.data.txt")
    L = r.readlines()
    X = []
    for i in L:
        K = np.array (i.strip().split (' ')).astype ("int")
        X.append (K)
    X = np.array(X)
    return X



def load_valid_dataset():
    r = open (r"C:\Users\Geekzone\Desktop\Hyper Heuristique PFE Crédit Scoring\Hyper Heuristique for Credit Scoring\Destiny\DataSets\gisette_valid.data.txt")
    L = r.readlines()
    X = []
    for i in L:
        K = np.array (i.strip().split (' ')).astype ("int")
        X.append (K)
    X = np.array(X)
    r = open (
        r"C:\Users\Geekzone\Desktop\Hyper Heuristique PFE Crédit Scoring\Hyper Heuristique for Credit Scoring\Destiny\DataSets\gisette_valid.labels.txt")
    L = r.readlines ()
    Y = []
    for i in L:
        K = np.array (i.strip ().split (' ')).astype ("int")
        Y.append (K[0])
    Y = np.array (Y)
    return X,Y

#train,target = load_gisette_dataset()
#print(target)
#E = Evaluateur_Precision(train,target.ravel())
#E.train(KNeighborsClassifier())
#print(E.vecteur_precision())