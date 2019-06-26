import math

from Utilitaire.Mesure import Mesure

class EntropyMeasures(Mesure):

    noms_mesures = ["Entropie" , 'GainInformation' , "GainRatio" , "SymetricalIncertitude" ,
                          "MutualInformation" , "UH" , "US" , "DML"]
    fitted = False

    @staticmethod
    def h(x):
        liste_vecteurs = []
        for i in x:
            liste_vecteurs.append (i)
        dict_valeurs = {}
        for i in range (0 , len (liste_vecteurs[0])):
            t = []
            for v in liste_vecteurs:
                t.append (v[i])
            tu = tuple (t)
            dict_valeurs[tu] = dict_valeurs.get (tu , 0) + 1
        nb_samples = len (liste_vecteurs[0])
        entropie = 0
        for i in dict_valeurs.keys ():
            dict_valeurs[i] = dict_valeurs[i] / nb_samples
            entropie = entropie + dict_valeurs[i] * math.log (dict_valeurs[i] , 2)
        return -entropie

    def __init__(self):
        super().__init__()
        self._liste_mesures = EntropyMeasures.noms_mesures
        self.__entropy = {}


    def fit(self,data,target):
        super().fit(data,target)
        d = data.transpose()
        cpt = 0
        self.__entropy[1] = {}
        for i in d:
            L = []
            L.append(i)
            self._attributs[str(cpt)] = i
            self.__entropy[1][str(cpt)] = EntropyMeasures.h(L)
            cpt = cpt +1
        L = []
        L.append (target)
        self._attributs["-1"] = target
        self.__entropy[1]["-1"] = EntropyMeasures.h(L)
        EntropyMeasures.fitted = True

    def getEntropy(self,liste_nums):
        if not len(liste_nums) in self.__entropy.keys():
            self.__entropy[len(liste_nums)] = {}

        if not tuple(liste_nums) in self.__entropy[len(liste_nums)].keys():
            L = []
            for i in liste_nums:
                L.append(self._attributs[str(i)])
            self.__entropy[len(liste_nums)][tuple(liste_nums)] = EntropyMeasures.h(L)

        return self.__entropy[len(liste_nums)][tuple(liste_nums)]

    def getEntropySachant(self,liste1,liste2):
        A = self.getEntropy(liste1+liste2)
        B = self.getEntropy(liste2)
        return A - B

    def getEntropyMeasures(self):
        return self.__entropy

    def ranking_function_constructor(self,motclef):
        ranker = None
        if(motclef=="Entropie"):
            ranker = self.getEntropy
        elif(motclef=="GainInformation"):
            ranker = self.gain_information
        elif(motclef=="GainRatio"):
            ranker = self.gain_ratio
        elif(motclef=="SymetricalIncertitude"):
            ranker = self.incertitude_symetrique
        elif(motclef=="MutualInformation"):
            ranker = self.mutual_information
        elif(motclef == "UH"):
            ranker = self.Uh_index
        elif(motclef == "US"):
            ranker = self.Us_index
        elif(motclef == "DML"):
            ranker = self.DML_index
        return ranker

    def gain_information(self,listenums):
        listenums = list(listenums)
        return self.getEntropy([-1]) - self.getEntropySachant([-1],listenums)

    def gain_ratio(self,listenums):
        listenums = list (listenums)
        return self.gain_information(listenums) / self.getEntropy(listenums)


    def mutual_information(self , listenumero):
        listenumero = list (listenumero)
        C = self.getEntropy([-1])
        X = self.getEntropy(listenumero)
        CX = self.getEntropySachant(listenumero,[-1])
        return C+X-CX


    def incertitude_symetrique(self,listenums):
        listenums = list (listenums)
        I = self.gain_information(listenums)
        C = self.getEntropy([-1])
        X = self.getEntropy(listenums)
        incert_sym = 2*(I / (C + X))
        return incert_sym

    def Us_index(self,listenumeros):
        X = self.getEntropy(listenumeros)
        listenumeros = list (listenumeros)
        return self.mutual_information(listenumeros)  / X

    def Uh_index(self , listenumeros):
        listenumeros = list (listenumeros)
        X = self.getEntropy(listenumeros+[-1])
        return self.mutual_information (listenumeros) / X

    def DML_index(self , listenumeros):
        listenumeros = list (listenumeros)
        A = self.getEntropySachant(listenumeros,[-1])
        B = self.getEntropySachant([-1],listenumeros)
        return B/A




noms_mesures = ["Entropie" , 'GainInformation' , "GainRatio" , "SymetricalIncertitude" ,
                          "MutualInformation" , "UH" , "US" , "DML"]
classe_mesure = EntropyMeasures


