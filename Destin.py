from sklearn.ensemble import AdaBoostClassifier



from sklearn.neural_network import MLPClassifier

from sklearn.svm import SVC
import numpy as np
import os
from Calculs import *
from DataSets.german_dataset import load_german_dataset
from Utilitaire import SubsetGenerator
from Utilitaire.Evaluateur_Precision import Evaluateur_Precision


class Destiny:
    #Pourcentage Feature par subset :
    percentage_feature = 20
    #C'est simple pour demander un ranking donné tu précise la lettre suivi de l'indice donc D0 pour Chi, I1 pour le gain d'information etc...
    #Sinon on peut indexer en utilisant H et un chiffre qui commence a 0

    #mesures_distance = ["FScore","ReliefF","FCS"]
    #mesures_information = [ 'GainInformation' , "GainRatio" , "SymetricalIncertitude" , "MutualInformation" , "UH" , "US" , "DML"]
    #mesures_classification = ["RF",  "AdaBoost"]
    #mesures_consistance = ['FCC']
    #mesures_dependance = ["RST"]

    #Heuristiques considérés pour des raisons de test


    mesures_consideres = ['FScore', 'RST', 'ReliefF', 'FCS', 'GainInformation', 'GainRatio', 'SymetricalIncertitude',"FCC"]
    mapping_mesures = {}
    nb_heuristiques = 0
    maxH = 0
    alpha=0.02

    def __init__(self):
        self.GestionPathMesures()
        print ("Création d'un objet Destiny manipulant : " , Destiny.mapping_mesures)
        self.__data,self.__target = None,None
        self.__mesures = {}
        self.__max_iterations=5
        self.__Threshold = 0
        self.__nom_mesures = {}
        self.subsetgenerated = None
        self.__subset_calculator = None
        self.__mesures_anterieure = {}
        L = []
        for i in Destiny.mesures_consideres:
            if not (i in L):
                for j in Destiny.mapping_mesures:
                    if(i in j):
                        self.__mesures[j] = Destiny.mapping_mesures[j]()
                        self.__nom_mesures[j] = j
                        for k in j:
                            L = L + list(k)
        print("Toutes les mesures")
        print(self.toutes_les_mesures())
        self.inter=set()
        self.union=set()
        print("Mapping des mesures :")
        print(Destiny.mapping_mesures)
        Destiny.nb_heuristiques =  len(Destiny.mesures_consideres) - 1
        Destiny.maxH = Destiny.nb_heuristiques - 1
        self.liste_mesures = []
        for i in self.__nom_mesures:
            self.liste_mesures = self.liste_mesures + list(self.__nom_mesures[i])

    #L est une liste de types de mesures par exemple ['C' ,'Ce', 'De']
    def GestionSubsets(self,L,Borne = None):
        for i in L:
            self.__mesures[i].CreateSubsets(Borne)

    def GestionPathMesures(self):
        from Calculs import FCC,FCS,ChiMesure,FScore,ReliefF,RST,SklearnClassifiersPrecisionOtO,EntropyMeasures
        L = [FCC,FCS,ChiMesure,FScore,ReliefF,RST,SklearnClassifiersPrecisionOtO,EntropyMeasures]
        for a in (L):
            Destiny.mapping_mesures[tuple (a.noms_mesures)] = a.classe_mesure

    def getMesure(self,nom):
        return self.__mesures[nom]

    def getDataset(self):
        return self.__data,self.__target

    def Projection(self,subset):
        #Ancienne projection :
        #D = self.__mesures["D"].ranking_function_constructor("FCS")(subset)
        #I = self.__mesures["I"].ranking_function_constructor("US")(subset)

        #Nouvelle Projection :
        D = self.__subset_calculator.Energie(subset,mrmR = True)
        I = self.__mesures[tuple(["Entropie" , 'GainInformation' , "GainRatio" , "SymetricalIncertitude" ,
                          "MutualInformation" , "UH" , "US" , "DML"])].ranking_function_constructor("GainRatio") (subset )
        DE = self.__mesures[tuple(["RST"])].dependence(subset)
        Co = self.__mesures[tuple(['FCC'])].fcc(subset)
        return (D,I,DE,Co)


    def getMegaHeuristique(self , ids , nb):
        LmotClef = []
        D = {}
        for i in ids:
            v = int(i[1:])
            if v > len(Destiny.mesures_consideres)-1:
                print("Erreur index trop grand")
            else:
                nm = Destiny.mesures_consideres[v]
                LmotClef.append(nm)
                for j in self.__mesures:
                    if(nm in j):
                        D.update(self.__mesures[j].rank_with([nm],n=nb))
        DDD = {}
        for i in D[nb]:
            if(i in LmotClef):
                DDD[i] = D[nb][i]
        return DDD


    def ThresholdMeasures(self,seuil):
        cpt = 0
        for j in range(0,len(Destiny.mesures_consideres)):
            self.__mesures_anterieure.update(self.getMegaHeuristique(["H"+str(cpt)],1))
            cpt = cpt + 1
        self.__Threshold = seuil
        for i in self.__mesures.keys():
            self.__mesures[i].setThresholdsAutomatiquement(self.__Threshold)


    def attributs_qualitatifs(self,seuil):
        nb = int(seuil * len(self.__data[0]))
        dict_listes = {}
        cpt = 0
        for i in self.__mesures_anterieure:
            L = []
            for j in range(0,nb):
                try:
                    L.append(self.__mesures_anterieure[i][j][0][0])
                except(TypeError):
                    print("ERREUUUUR : ",self.__mesures_anterieure[i])
                    print("i = ",i)
            dict_listes[cpt] = L
            cpt = cpt + 1
        return dict_listes



    def fit(self,X,Y):
        self.__data ,self.__target = X, Y
        for i in self.__mesures.keys():
            self.__mesures[i].fit (X , Y)
            print(i," fini")
            if(self.subsetgenerated == None):
                self.subsetgenerated = self.__mesures[i].CreateSubsets(pourcentage = Destiny.percentage_feature)
                self.__subset_calculator = self.__mesures[i].getCalculator()
            else:
                self.__mesures[i].setSubsets(self.subsetgenerated)
        cpt = 0
        for j in range (0 , len (self.mesures_consideres)):
            self.__mesures_anterieure.update (self.getMegaHeuristique (["H" + str (cpt)] , 1))
            cpt = cpt + 1
        self.activer_treshold()

    def test(self):
        D = {}
        for i in self.__mesures.keys():
            print("Utilisation de ", i)
            D1 = self.__mesures[i].rank_with(n=1)
            D2 = self.__mesures[i].rank_with(n=2)
            D3 = self.__mesures[i].rank_with(n=3)
            D4 = self.__mesures[i].rank_with (n=4)
            for i in D4:
                print(" i = ",i)
                for j in D4[i]:
                    print(j , " len= ",len(D4[i][j])," : " , D4[i][j])

    def tresholder(self,t):
        self.nouvmesures = self.__mesures
        for i in self.__mesures.keys():
            self.__mesures[i].setThresholdsAutomatiquement(t)

    def union_intersection2(self,t):
        self.inter = set()
        self.union = set()
        L=self.attributs_qualitatifs(t)
        for i in L.values():
            self.union=self.union.union(set(i))
            if(len(self.inter)==0):
                self.inter=set(i)
            else:
                self.inter=self.inter.intersection(set(i))
        print("union", self.union)
        print("inter", self.inter)

    def union_intersection(self):
        self.inter=set()
        self.union=set()
        for i in range(self.maxH):
            gj = self.getMegaHeuristique(["H" + str(i )], 1)
            hierlist2 = gj[list(gj.keys())[0]]
            elus = set()
            for h in hierlist2:
                if (h[1] >= 0):
                    elus = elus.union(set(h[0]))
           # print("les elus",elus)
            if (len(self.inter) > 0):
                self.inter = self.inter.intersection(elus)
            else:
                self.inter = elus
            self.union = self.union.union(elus)
        print("union",self.union)
        print("inter",self.inter)

    def evaluer(self):
        E = Evaluateur_Precision(self.__data, self.__target)
        E.train(SVC(gamma="auto"))
        if(len(self.inter)>0):
            a=(self.reguler_par_complexote(E.Evaluer(list(self.inter)),len(self.inter))+self.reguler_par_complexote(E.Evaluer(list(self.union)),len(self.union)))/2
            print(a)
            return a
        else:
            return 0


    def criteron(self,t):
        self.union_intersection2(t)
        return self.evaluer()

    def activer_treshold(self):
        t=0.5
        alpha=0.4
        for i in range(self.__max_iterations):
            p1=t+alpha
            p2=t-alpha
            if(self.criteron((t+p1)/2)>=self.criteron((t+p2)/2)):
                t=(p1+t)/2
            else:t=(p2+t)/2
            alpha=alpha/2
            print("----le treshold est:",t)
        self.ThresholdMeasures(t)


    def getTreshold(self):
        return self.__Threshold
    def getNbAttribute(self):
        return self.__data.shape[1]

    def reguler_par_complexote(self,val,taille):
        #return (val *(1-self.alpha)/(taille)*self.alpha)
        return val

    def criteron_heursitique_unique(self,h,t):
        ep = Evaluateur_Precision(self.__data,self.__target)
        ep.train(AdaBoostClassifier())
        D = self.attributs_qualitatifs(t)
        D = D[h]
        precision = ep.Evaluer(D)
        print("Une précision de : ",precision, " pour une longueur " , len(D), " correpondant au subset : ",D,)
        return ep.Evaluer(D)




    def generer_un_seul_threshold(self,h):
        t = 0.5
        alpha = 0.4
        self.__Threshold=h
        mprecision = 0
        for i in range (self.__max_iterations):
            p1 = t + alpha
            p2 = t - alpha
            if (self.criteron_heursitique_unique (h,(t + p1) / 2) >= self.criteron_heursitique_unique (h,(t + p2) / 2)):
                t = (p1 + t) / 2
                mprecision = self.criteron_heursitique_unique (h , (t + p1) / 2)
            else:
                t = (p2 + t) / 2
                mprecision = self.criteron_heursitique_unique (h,(t + p2) / 2)
            alpha = alpha / 2
            print ("----le treshold est:" , t)
        return t,mprecision

    def rapport_heuristique(self,h,modele = SVC(gamma="auto")):
        t = self.generer_un_seul_threshold(h)[0]
        att_qualitatifs = self.attributs_qualitatifs(t)[h]
        EP = Evaluateur_Precision(self.__data,self.__target)
        EP.train(modele)
        v = EP.Evaluer(att_qualitatifs)
        return EP.Evaluer_Metriques(att_qualitatifs),att_qualitatifs,t,v

    def toutes_les_mesures(self):
        t = []
        for i in Destiny.mapping_mesures:
            t = t + list(i)
        return t

D = Destiny()
data , target = load_german_dataset()
D.fit(data,target)
D.test()
print(D.Projection([4,5,6]))