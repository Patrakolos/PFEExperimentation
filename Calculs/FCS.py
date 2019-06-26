from itertools import combinations
import math
from Calculs.EntropyMeasures import  EntropyMeasures
from Utilitaire.Mesure import Mesure
import numpy as np


class FCS(Mesure):
    seuil_max= 4
    noms_mesures = ["FCS"]
    def __init__(self):
        super().__init__()
        self._liste_mesures = FCS.noms_mesures
        self.__data = None
        self.__target = None
        self.__classe_feature_r = {}
        self.__feature_feature = {}
        self.__mesures = {}
        self.__entropies = {}

    def getEntropy(self,listenoms):

        if not (tuple(listenoms) in self.__entropies):
            D = self.__data.transpose ()
            L = []
            for i in listenoms:
                if (i < len (D) and i != -1):
                    A = D[i]
                else:
                    A = self.__target.transpose ()
                L.append(A)
            L = np.array(L)
            self.__entropies[tuple(listenoms)] = EntropyMeasures.h(L)
        return self.__entropies[tuple(listenoms)]


    def r(self,x,y):
        #D = self.__data.transpose()
        #if(x < len(D) and y!=-1):
        #    A = D[x]
        #else:
        #    A = self.__target.transpose()
        #if(y < len(D) and y!=-1):
        #    B = D[y]
        #else:
        #    B = self.__target.transpose()
        HA = self.getEntropy([x])
        HB = self.getEntropy([y])
        HAB = self.getEntropy([x,y])
        rez = (HA + HB - HAB) / HA + HB
        return rez

    def fit(self,data,target):
        super().fit(data,target)
        self.__data = data
        self.__target = target

    def ranking_function_constructor(self,motclef):
        if(motclef == "FCS"):
            return self.score


    def getFeatureClassCorreclation(self,f):
        if not (f in self.__classe_feature_r.keys()):
            self.__classe_feature_r[f] = self.r(f,-1)
        return self.__classe_feature_r[f]

    def getFeatureFeatureCorreclation(self,f1,f2):
        if not ((f1,f2) in self.__feature_feature.keys()):
            self.__feature_feature[(f1,f2)] = self.r(f1,f2)
        return self.__feature_feature[(f1,f2)]

    def rankingOneByOne(self):
        for i in range(0,len(self.__data[0])):
            self.score([i])


    def rankingBy(self,n):
        if(n>FCS.seuil_max):
            return None
        if not (n in self.__mesures):
            L = list(range(0,len(self.__data[0])-1))
            for i in combinations(L,n):
                self.score(i)
            self.__mesures[n] = sorted(self.__mesures[n].items(),key=lambda x:x[1],reverse=True)
        return self.__mesures[n]

    def score(self , x):
        if not (len(x) in self.__mesures.keys()):
            self.__mesures[len(x)] = {}

        if not (tuple(x) in self.__mesures[len(x)].keys()):
            k = len(x)
            x = sorted(x)
            SFF = 0
            SFC = 0
            cptFF = 0
            cptFC = 0
            for i in x:
                SFC = SFC + self.getFeatureClassCorreclation(i)
                cptFC = cptFC + 1
                for j in x:
                    if(j>i):
                        SFF = SFF  + self.getFeatureFeatureCorreclation(i,j)
                        cptFF = cptFF + 1
            if (SFF == 0):
                SFF = 1
                cptFF = 1
            SFF = SFF/cptFF
            SFC = SFC/cptFC
            denom = math.sqrt(k+k*(k-1)*SFF)
            numer = k*SFC
            self.__mesures[len(x)][tuple(x)] = numer/denom
        return self.__mesures[len(x)][tuple(x)]

noms_mesures = ["FCS"]

classe_mesure = FCS