import requests
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

from Utilitaire.Evaluateur_Precision import Evaluateur_Precision



def load_promoter_dataset():
    r = open (r"C:\Users\aa\Desktop\CodeRefonte\DataSets\promoters.data")
    L = r.readlines()
    X = []
    for i in L:
        K = np.array (i.strip().replace(r"\t\t","").split(","))
        X.append (K)
    X = np.array(X)
    X = X.transpose()
    Y = []
    for i in X[0]:
        if(i=="+"):
            Y.append(1)
        else:
            Y.append(0)
    R = []
    for i in X[2]:
        gen = []
        for g in i:
            if(g == 'a'):
                gen.append(1)
            elif(g == 'c'):
                gen.append(2)
            elif(g == 't'):
                gen.append(3)
            elif(g=='g'):
                gen.append(4)
        R.append(gen)
    return np.array(R),np.array(Y)

#train,target = load_promoter_dataset()
#print(target)
#E = Evaluateur_Precision(train,target.ravel())
#E.train(KNeighborsClassifier())
#print(E.vecteur_precision())