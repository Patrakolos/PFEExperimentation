
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


from Destiny.Evaluateur_Precision import Evaluateur_Precision


def load__train_dataset():
    r = open(r"C:\Users\Geekzone\Desktop\Hyper Heuristique PFE Crédit Scoring\Hyper Heuristique for Credit Scoring\Destiny\DataSets\madelon_train.data.txt")
    L = r.readlines()
    X = []
    for i in L:
        ch = i.strip()
        K = np.array(ch.split(' ')).astype("int")
        X.append (K)
    X = np.array(X)
    r = open (r"C:\Users\Geekzone\Desktop\Hyper Heuristique PFE Crédit Scoring\Hyper Heuristique for Credit Scoring\Destiny\DataSets\madelon_train.labels.txt")
    L = r.readlines ()
    Y = []
    for i in L:
        ch = i.strip ()
        K = np.array (ch.split (' ')).astype ("int")
        Y.append (K[0])
    Y = np.array (Y)
    print("Train : ", X.shape)
    print("Test : ",Y.shape)
    return X,Y

def load_test_dataset():
    r = open (r"madelon_test.data.txt")
    L = r.readlines ()
    X = []
    for i in L:
        ch = i.strip ()
        K = np.array (ch.split (' ')).astype ("int")
        X.append (K)
    X = np.array (X)
    return X

def load_validation_dataset():
    r = open (r"madelon_valid.data.txt")
    L = r.readlines ()
    X = []
    for i in L:
        ch = i.strip ()
        K = np.array (ch.split (' ')).astype ("int")
        X.append (K)
    X = np.array (X)
    return X


#train,target = load__train_dataset()
#print(target)
#E = Evaluateur_Precision(train,target.ravel())
#E.train(KNeighborsClassifier())
#print(E.vecteur_precision())