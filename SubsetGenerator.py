import math
import random
from functools import reduce
from itertools import combinations

from sklearn.ensemble import AdaBoostClassifier
import numpy as np

from DataSets.australian_dataset import load_australian_dataset
from DataSets.german_dataset import load_german_dataset
from DataSets.load_spambase_dataset import load_spambase_dataset


class SubsetGenerator():

    pourcentage_combinaison = 20
    max_iter_possible = 10000


    def __init__(self):
        self.__modele = AdaBoostClassifier()
        self.__data = None
        self.__data_masque = None
        self._rfecv = None
        self.__target = None
        self.__threshold = 0
        self.__subset_selectionned = {}
        self.__nbfeatures = 0
        self.__stats={}
        self.__rappport_execution = []
        self.__matrices_redondaces, self.__matrices_importances = [], []


    def fit(self,X,Y):
        self.__nbfeatures = len(X[0])
        self.__data = X
        self.__target = Y
        self.__data_masque = X
        self.__nbfeatures = len(self.__data.transpose())
        self.__threshold = 100
        D = []
        L = list(X.T)
        for i in L:
            D.append(list(i))
        D.append(list(Y))
        total = np.array(D)
        corr = np.corrcoef (total)
        corr = np.abs(corr)
        corr = corr.T
        self.__matrices_importances = corr[-1][:-1]
        self.__matrices_redondaces = corr[:-1]
        self.__matrices_redondaces = self.__matrices_redondaces.T[:-1]
        for i in range(0,len(list(self.__matrices_redondaces))):
            self.__matrices_redondaces[i][i] = 0
        self.__matrices_importances = np.ones(self.__matrices_importances.shape) - self.__matrices_importances / np.sum(self.__matrices_importances)

    def Energie(self,L,mrmR = False):
        masque = [0] * self.__nbfeatures
        for i in L:
            masque[i] = 1
        masque = np.array(masque)
        try:
            #print("Redondance : ",masque.transpose().dot(self.__matrices_redondaces).dot(masque))
            #print("Significativité : ",masque.dot(
            #    self.__matrices_importances))
            e =  masque.transpose().dot(self.__matrices_redondaces).dot(masque) - masque.dot(
                self.__matrices_importances)
            if(mrmR == 'True'):
                return e / len(L)
        except(ZeroDivisionError):
            e = 1000
        return e

    def GenererListeRandom(self,taille):
        r = taille
        L = r * [-1]
        for i in range(0,r):

            try:
                k = random.randint(0,self.__nbfeatures-1)
            except(ValueError):
                print("Erreur du Randint : ",self.__nbfeatures-1)
                print("Data : ",self.__data)
            while (k in L and len (L) != self.__nbfeatures):
                k = random.randint (0 , self.__nbfeatures-1)
            L[i] = k
        return L

    def Alteration_Insensification(self,L):
        min_r = 0
        subset = list(L)
        for i in subset:
            if(self.__matrices_importances[i] < self.__matrices_importances[min_r]):
                min_r = i
        subset.remove(subset[min_r])
        m = np.zeros(self.__nbfeatures)
        D = self.__matrices_redondaces.T
        for i in subset:
            m = m + D[i]
        m = list(m)
        min_red = m.index(min(m))
        while(min_red in subset):
            m.remove(min(m))
            min_red = m.index (min (m))
        subset.append(min_red)
        return subset

    def Energie_Moyenne(self,E):
        LE  =[]
        for i in E:
            LE.append(self.Energie(i))
        return np.mean(LE)

    def Energie_Moyenne_Meilleurs(self,E,nb):
        LE = {}
        for i in E:
            LE[tuple(i)] = self.Energie(i)
        I = sorted(LE.items(),key=lambda x:x[1],reverse=False)
        I = I[:nb]
        L = []
        for i in I:
            L.append(list(i[0]))
        print(L)
        return self.Energie_Moyenne(L)


    def Benchmark(self,t):
        self.__nbfeatures = len(self.__data[0])
        self.__stats[t] = {}
        c = combinations (list(range (0 , self.__nbfeatures)) , t)
        c = list(c)
        LL = []
        for i in c:
            LL.append(list(i))
        c = LL
        nb_com = len(list(c))
        nb_att = int(nb_com * SubsetGenerator.pourcentage_combinaison / 100)

        EC = self.Energie_Moyenne((c))
        self.__stats[t][("Mean",0)] =  EC
        E1 = []
        for i in range(0,nb_att):
            E1.append(self.GenererListeRandom(t))
        self.__stats[t][("Random",i)] = self.Energie_Moyenne(E1)

        for i in range(0,5):
            E2 = self.generer_subset (t , nb_att)
            print("E2 = ",E2)
            self.__stats[t][("Heuristic",i)] = np.array (self.Energie_Moyenne(E2))
        print(self.__stats)
        E3 = []
        self.__stats[t][("Exhaustive",0)] = self.Energie_Moyenne_Meilleurs (c , nb_att)
        return self.__stats[t]


    def getStats(self):
        return self.__stats

    def getRapport(self):
        return self.__rappport_execution

    @staticmethod
    def egaliteSubsets(L1,L2):
        for i in L1:
            if(i not in L2):
                return False
        return True

    @staticmethod
    def verifierExistence(subset,liste_subsets):
        existant = False
        if(len(liste_subsets) == 0):
            return False
        for i in liste_subsets:
            if (SubsetGenerator.egaliteSubsets (i , subset)):
                existant = True
                break
        return existant


    @staticmethod
    def nCk(n , k):
        from operator import mul  # or mul=lambda x,y:x*y
        from fractions import Fraction
        return int (reduce (mul , (Fraction (n - i , i + 1) for i in range (k)) , 1))


    def generer_subset(self,taille,borne_in = None,rapport = False,verbose = False):
        self.__subset_selectionned.clear()
        self.__subset_selectionned[taille] = []
        L = self.GenererListeRandom(taille)
        cpt_iter = 0
        selec = []
        selectivite = True
        if(borne_in > SubsetGenerator.nCk(self.__nbfeatures,taille)):
            print("Scénario 1")
            born_to_select = SubsetGenerator.nCk(self.__nbfeatures,taille)
            selectivite = False
            print("Calcul d'une nouvelle borne : ",born_to_select)
        elif((borne_in / SubsetGenerator.nCk(self.__nbfeatures,taille)) * 100 <= 20):
            print ("Scénario 3")
            born_to_select = int(2*borne_in)
            selectivite = False
        else:
            print ("Scénario 2")
            born_to_select = borne_in
            selectivite = False

        while (cpt_iter < SubsetGenerator.max_iter_possible):

            L = self.Alteration_Insensification(L)

            if (SubsetGenerator.verifierExistence(L,selec)):
                if(verbose):
                    print("Regenération car : ",L," existe dans ",selec)
                L = self.GenererListeRandom(taille)
                if(selectivite):
                    continue


            if (not SubsetGenerator.verifierExistence (L , selec)):
                if(verbose):
                    print ("existant : " , SubsetGenerator.verifierExistence (L , selec))
                    print ("Ajout de : " , L)
                selec.append (L)
                if(verbose):
                    print("Ajouté ! selec = ",selec)
                    print("====")
                cpt_iter = cpt_iter + 1

            # Selectionner l'ensemble courant

            if(len(selec)==born_to_select):
                break

            if (rapport):
                self.__rappport_execution.append ((cpt_iter , self.Energie(L) , taille))


        #Classement et Selecton
        selec_score = []
        for i in selec:
            selec_score.append((i,self.Energie(i)))
        selec_score = sorted(selec_score,key=lambda x:x[1])
        print ("Fin de selection : " , selec_score)

        #Optimisation possible
        W = []
        for i in selec_score:
            W.append(i[0])

        self.__subset_selectionned[len(L)] = W[:borne_in]

        print("Scores selectionnés : ",self.__subset_selectionned[len(L)])
        print("Fin de l'algorithme pour une taille de ",taille, " qui a sélectionnée : ",len(self.__subset_selectionned[len(L)]), "parmi : ",SubsetGenerator.nCk(self.__nbfeatures,taille))
        return self.__subset_selectionned[len(L)]

    def generer_subset2(self,taille,borne = None,rapport = False):
        self.__subset_selectionned.clear()
        self.__subset_selectionned[taille] = []
        L = self.GenererListeRandom(taille)
        e = self.Energie(L)
        T = 100000
        pas = 0.01
        coef_decrementation_temperature = 0.99
        cpt_iter = 0
        nb_iterations_pas = 100
        cptcpt = nb_iterations_pas
        while (cpt_iter < 1000):
            # Mecanisme d'intensification
            NL = self.Alteration_Insensification(L)
            while(NL in self.__subset_selectionned[taille]):
                NL = self.GenererListeRandom(taille)

            # Mise a jour de la température
            T = T * coef_decrementation_temperature
            # Mise a jour de l'energie
            ep = e
            enew = self.Energie(NL)
            if (enew <= ep ):
                cpt_iter = cpt_iter + 1
                L = NL
                e = enew
                if (rapport):
                    self.__rappport_execution.append ((cpt_iter , e , taille))
            else:
                try:
                    #print ("Enew = " , enew)
                    #print ("Ep = " , ep)
                    delta = (enew - ep)
                    p = math.exp(-(delta / T))
                    #print("delta = ",delta)
                    print("p = ",p*100)
                except(OverflowError):
                    print("Overflow")
                    continue
                p = int(p * 100)
                rr = random.randint(0, 100)
                if (p > rr):
                    cpt_iter = cpt_iter + 1
                    L = NL
                    e = enew
                    if (rapport):
                        self.__rappport_execution.append ((cpt_iter , e , taille))
            # Selectionner l'ensemble courant
            if not L in self.__subset_selectionned[len(L)]:
                self.__subset_selectionned[len(L)].append(L)

            if(len(self.__subset_selectionned[len(L)])==borne):
                return self.__subset_selectionned[len(L)]
            print("Iteration = ",cpt_iter," L = ",L)
        print("Fin de l'algorithme pour une taille de ",taille, " qui a sélectionnée : ",len(self.__subset_selectionned[len]))
        return self.__subset_selectionned[len(L)]

def BenchmarkInFile():
    data , target = load_german_dataset ()
    S = SubsetGenerator ()
    S.fit (data , target)
    for i in range (2 , 5):
        S.Benchmark (i)
    St = S.getStats ()
    L = []
    for i in St:
        for j in St[i]:
            L.append ([i , St[i][j] , j[0]])
    L = np.array (L)
    import pandas as pd
    d = pd.DataFrame(L)
    print ("Liste ecrie " , L)
    print(d)
    d.to_csv("SubsetGeneration_Benchmarking.csv")


def StatistiqueExecutionInFile():
    data , target = load_german_dataset ()
    S = SubsetGenerator ()
    S.fit (data , target)

    for t in range (5, 7):
        c = combinations (list (range (0 , len (data[0]))) , t)
        c = list (c)
        LL = []
        for i in c:
            LL.append (list (i))
        c = LL
        nb_com = len (list (c))
        nb_att = int (nb_com * SubsetGenerator.pourcentage_combinaison / 100)
        S.generer_subset (t,nb_att,rapport=True)
    St = S.getRapport ()
    L = []
    for i in St:
        L.append (list(i))
    L = np.array (L)
    import pandas as pd
    d = pd.DataFrame (L)
    print ("Liste ecrie " , L)
    print (d)
    d.to_csv ("SubsetGeneration_Exection.csv")

def DisplayStats():
    import pandas as pd
    import seaborn as sns;
    sns.set ()
    import matplotlib.pyplot as plt
    df = pd.read_csv("SubsetGeneration_Benchmarking.csv")
    ax = sns.lineplot (x="0" , y="1" , hue="2" , data=df)
    plt.show ()
    return df


def DisplayRapportExecution():
    import pandas as pd
    import seaborn as sns;
    sns.set ()
    import matplotlib.pyplot as plt
    df = pd.read_csv ("SubsetGeneration_Exection.csv")
    ax = sns.lineplot (x="0" , y="1" , hue="2" , data=df)
    plt.show ()
    return df


#StatistiqueExecutionInFile()
#DisplayRapportExecution()
BenchmarkInFile()
DisplayStats()
#data,target = load_german_dataset()
#S = SubsetGenerator()
#S.fit(data,target)
#borne = SubsetGenerator.nCk (20 , 3)
#print(borne)
#
#print(S.generer_subset(3,150,verbose=True))
