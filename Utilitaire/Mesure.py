from itertools import combinations

from Utilitaire.Embedded_Thresholding import Embedded_Thresholding
from Utilitaire.SubsetGenerator import SubsetGenerator


class Mesure:

    #Taille de Méga Attribut :
    MegaAttributTailleMax = 4

    def __init__(self):
        self._liste_mesures = []
        self._liste_thresholds = []
        self._calculated_measures = {}
        self._attributs = {}
        self._subsets = None
        self._target = {}
        self.__data = None
        self.__target = None
        self.__Subset_Calculator = SubsetGenerator()


    def setSubsets(self,subset):
        self._subsets = subset

    def getCalculator(self):
        return self.__Subset_Calculator

    def CreateSubsets(self,pourcentage=10):
        self._subsets = {}
        for i in range(2,Mesure.MegaAttributTailleMax+1):
            borne = int(SubsetGenerator.nCk(len(self.__data[0]),i) * (pourcentage / 100))
            self._subsets[i] = self.__Subset_Calculator.generer_subset(i , borne)
        return self._subsets

    def setThresholdsAutomatiquement(self,s=None):
        self.rank_with(n=1)
        for j in self._calculated_measures[1]:
            if(self._liste_thresholds[self._liste_mesures.index (j)]==0):
                self._liste_thresholds[self._liste_mesures.index(j)] = self._calculated_measures[1][j][int(s*(len(self._attributs.keys())-1))][1]
        self._calculated_measures.clear()


    def fit(self,data,target):
        self.__data = data
        self.__target = target
        self.__Subset_Calculator.fit(self.__data, self.__target)
        d = data.transpose()
        cpt = 0
        for i in d:
            self._attributs[str(cpt)] = i
            cpt = cpt +1
        self._attributs["-1"] = target
        self._liste_thresholds = [0] * (len (self._attributs.keys ()) - 1)


    def ranking_function_constructor(self,motclef):
        pass


    def ranked_attributs(self,motclef,nb=1):
        if not motclef in self._liste_mesures:
            return None
        ranker = self.ranking_function_constructor(motclef)
        scores = []
        L = range(0,len(self._attributs.keys())-1)
        if(self._subsets == None or not nb in self._subsets.keys()):
            C = combinations(L,nb)
        else:
            C = self._subsets[nb]

        for i in C:
            K = []
            for j in i:
                K.append(j)
            if(self._liste_thresholds[self._liste_mesures.index(motclef)] == 0 or ranker(tuple(K))>=self._liste_thresholds[self._liste_mesures.index(motclef)]):
                t = i,ranker(tuple(K))
            else:
                t = i,(-1)
            scores.append(t)
        scores.sort(key=lambda x:x[1],reverse=True)
        return scores

    def getCalculatedMeasures(self):
        return self._calculated_measures

    def rank_with(self,lmotclef = None,n = 1):
        if not n in self._calculated_measures.keys():
            self._calculated_measures[n] = {}
        if(lmotclef == None):
            lmotclef = self._liste_mesures
        for mc in lmotclef:
            if not (mc in self._calculated_measures[n].keys()):
                self._calculated_measures[n][mc] = self.ranked_attributs(mc,nb=n)
        return self._calculated_measures


