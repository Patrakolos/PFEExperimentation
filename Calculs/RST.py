import numpy as np

from Utilitaire.Mesure import Mesure

noms_mesures = ["RST"]
class RST(Mesure):

    def __init__(self):
        super().__init__()
        self._liste_mesures = ["RST"]
        self.__data = None
        self.__target = None
        self.feature_score= {}


    def fit(self,data,target):
        super().fit(data,target)
        self.__data = data
        self.__target = target

    def ranking_function_constructor(self,motclef):
        if(motclef == "RST"):
            return self.dependence


    def dependence(self,x):
        x = list(x)
        a = []
        dat = np.transpose(self.__data)
        for i in range(len(x)):
            a.append(dat[x][i])
        patterns = []
        b = np.transpose(a)
        for j in b:
            patterns.append(tuple(j))
        g1 = []
        g2 = []
        liste=[]
        patternsset=list(set(patterns))
        for j in range(len(patternsset)):
            sousliste=[]
            for z in range(len(patterns)):
                if(patternsset[j]==patterns[z]):
                    sousliste.append(z)
                if (self.__target[z] == 1):
                    g1.append(z)
                else:
                    g2.append(z)
            sousliste=set(sousliste)
            liste.append(sousliste)
        n=0
        g1=set(g1)
        g2=set(g2)
        for l in liste:
            if(l<g1 or l<g2):
                n=n+len(l)
        try:
            rez = n / len(patterns)
        except(ZeroDivisionError):
            print("pattern = ",patterns)
            rez = 0
        return rez


noms_mesures = ["RST"]
classe_mesure = RST