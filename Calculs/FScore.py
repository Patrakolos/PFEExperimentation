from sklearn.feature_selection import f_classif, chi2

import numpy as np

from Utilitaire.Mesure import Mesure
from Utilitaire import Dimension_Reductor


class FScore(Mesure):

    noms_mesures = ["FScore"]


    def __init__(self):
        super().__init__()
        self.__data = None
        self.__target = None
        self._liste_mesures = FScore.noms_mesures
        self.__scores = {}


    def fit(self,data,target):
        super().fit(data,target)
        self.__data = data
        self.__target = target
        sc = f_classif (data , target)
        cpt = 0
        for i in sc[0]:
            tup = cpt , i
            self.__scores[cpt] = tup
            cpt = cpt + 1



    def ranking_function_constructor(self,motclef):
        if(motclef=="FScore"):
            return self.score


    def score(self, x):
        if (len(x) > 1):
            DR = Dimension_Reductor.Dimension_Reductor()
            DR.fit (self.__data , self.__target)
            L = DR.getPCA (x)
            LL = []
            LL.append (L)
            LL = np.array (LL)
            LL = LL.transpose ()
            R = FScore ()
            R.fit(LL,self.__target)
            score = (8,R.score([0]))
        else:
            score = self.__scores[x[0]]
        return score[1]




noms_mesures = ["FScore"]
classe_mesure = FScore
