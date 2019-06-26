import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.ensemble import RandomForestClassifier , AdaBoostClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier

from Utilitaire.Evaluateur_Precision import Evaluateur_Precision


def load_musk_dataset():
    r = open (r"C:\Users\Geekzone\Desktop\Hyper Heuristique PFE Crédit Scoring\Hyper Heuristique for Credit Scoring\Destiny\DataSets\musk2.data")
    L = r.readlines()
    X = []
    for i in L:
        K = np.array (i.strip().split (','))
        X.append (K)
    r = open (
        r"C:\Users\Geekzone\Desktop\Hyper Heuristique PFE Crédit Scoring\Hyper Heuristique for Credit Scoring\Destiny\DataSets\musk1.data")
    L = r.readlines ()
    for i in L:
        K = np.array (i.strip().split (','))
        X.append (K)
    X = np.array(X)
    X = X.transpose()
    X = X[3:].astype('float')
    Y = X[-1]
    X = X[:-1]
    X = X.transpose()
    return X,Y

