from random import randint

from pandas import DataFrame
from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import KBinsDiscretizer

from DataSets.Dataset import Dataset


#parametrage simulation
from Calculs.Destin import Destiny
from Utilitaire.Evaluateur_Precision import Evaluateur_Precision
from Utilitaire.SubsetGenerator import SubsetGenerator


def sommeCarre(tuple):
    s = 0
    for i in tuple:
        s = s + i * i
    return s

def somme(tuple):
    s = 0
    for i in tuple:
        s = s + i
    return s

nom_dataset = "German"

D = Dataset(nom_dataset)
D.loadDescription()
data,target = D.dataMung(transform=True,discretize=True)


def simuler_somme():
    EP = Evaluateur_Precision(data,target)
    EP.train(AdaBoostClassifier())

    Dest = Destiny()
    Dest.fit(data,target)

    SG = SubsetGenerator()
    SG.fit(data,target)
    liste_tuples = []
    for i in range(0,200):
        taille_random = randint(1,data.shape[1])
        L = SG.GenererListeRandom(taille_random)
        #liste_tuples.append([EP.Evaluer(L),somme(Dest.Projection(L))])
        liste_tuples.append ([EP.Evaluer (L) , sommeCarre (Dest.Projection (L)) ])



    import numpy as np
    import pandas as pd
    liste_tuples = sorted(liste_tuples,key=lambda x:x[1])
    t = np.array(liste_tuples)
    K = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='uniform')
    t = K.fit_transform(t)
    df = DataFrame(t)
    df.to_csv("rez.tmp")
    df = pd.read_csv('rez.tmp')
    print(df)
    import seaborn as sns
    sns.set ()
    import matplotlib.pyplot as plt
    ax = sns.lineplot (x="1" , y="0" ,data=df)
    plt.show ()

