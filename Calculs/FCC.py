import numpy as np

from Utilitaire.Mesure import Mesure
from DataSets.german_dataset import load_german_dataset

class FCC(Mesure):

    seuil_max = 4
    noms_mesures = ['FCC']
    def __init__(self):
        super().__init__()
        self.__data = None
        self._liste_mesures = FCC.noms_mesures
        self.__target = None
        self.ranks = {}
        self.feature_score= {}

    def getscore(self):
        return self.feature_score

    def fit(self,data,target):
        super().fit(data,target)
        self.__data = data
        self.__target = target


    def ranking_function_constructor(self,motclef):
        if(motclef == 'FCC'):
            return self.fcc


    def fcc(self,x):
        x = list(x)
        if not (len (x) in self.feature_score.keys()):
            self.feature_score[len(x)] = {}
        if not tuple(x) in self.feature_score[len(x)].keys():
            a=[]
            dat=np.transpose(self.__data)
            for i in range(len(x)):
                a.append(dat[x][i])
            patterns=[]
            b=np.transpose(a)
            for j in b:
                patterns.append(tuple(j))
            setpatterns=set(patterns)
            s=0
            for pat in setpatterns:
                npp=0
                c1=0
                c2=0
                for val in range(len(patterns)):
                    if(pat==patterns[val]):
                        npp=npp+1
                        if(self.__target[val]==1):
                            c1=c1+1
                        else: c2=c2+1
                c1=max(c1,c2)
                s=s+npp-c1
            if(1-float(s)/len(patterns) >= self._liste_thresholds[0]):
                self.feature_score[len(x)][tuple(x)] = 1-float(s)/len(patterns)
            else:
                self.feature_score[len (x)][tuple (x)] = -1
        return self.feature_score[len(x)][tuple(x)]


noms_mesures = ["FCC"]
classe_mesure = FCC

#data, target = load_german_dataset()
#CM = FCC()
#CM.fit(data,target)
#print(CM.rank_with(["FCC"],n=2))