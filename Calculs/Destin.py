from nltk import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.neural_network import MLPClassifier

from sklearn.svm import SVC
import numpy as np
import os
from Calculs import *
from DataSets.german_dataset import load_german_dataset
from Utilitaire.Evaluateur_Precision import Evaluateur_Precision
from Utilitaire.Tresholding import Tresholding
from sklearn.naive_bayes import GaussianNB
import time


class Destiny:
    #C'est simple pour demander un ranking donné tu précise la lettre suivi de l'indice donc D0 pour Chi, I1 pour le gain d'information etc...
    #Sinon on peut indexer en utilisant H et un chiffre qui commence a 0

    #mesures_distance = ["FScore","ReliefF","FCS"]
    #mesures_information = [ 'GainInformation' , "GainRatio" , "SymetricalIncertitude" , "MutualInformation" , "UH" , "US" , "DML"]
    #mesures_classification = ["RF",  "AdaBoost"]
    #mesures_consistance = ['FCC']
    #mesures_dependance = ["RST"]

    #Heuristiques considérés pour des raisons de test

    percentage_feature = 20

    mesures_consideres = ["FScore","GainRatio" ,"FCC","DML"]
    mesures_obligatoires=['FScore','RST','FCC','GainRatio']
    mapping_mesures = {}
    nb_heuristiques = 0
    maxH = 0
    alpha=0.0
    coef1 = 1
    coef2 = 1
    coef3 = 1
    stop = False
    model_dict={"SVM":SVC,"DTC":DecisionTreeClassifier,"MLP":MLPClassifier,"AdaBoost":AdaBoostClassifier,"KNN":KNeighborsClassifier,"RF":RandomForestClassifier, "BN":GaussianNB}


    def __init__(self,modele,seuilllage="union_intersection"):
        self.GestionPathMesures()
        #print ("Création d'un objet Destiny manipulant : " , Destiny.mapping_mesures)
        self.__data,self.__target = None,None
        self.model_name=modele
        self.model=Destiny.model_dict[modele]()
        self.__mesures = {}
        self.__max_iterations=5
        self.__Threshold = 0
        self.__nom_mesures = {}
        self.seuillage = seuilllage
        self.subsetgenerated = None
        self.__subset_calculator = None
        self.__mesures_anterieure = {}
        #self.__matrices_redondaces, self.__matrices_importances = {} , {}
        L = []
        for i in list(set(Destiny.mesures_consideres+Destiny.mesures_obligatoires)):
            if not (i in L):
                for j in Destiny.mapping_mesures:
                    if(i in j):

                        self.__mesures[j] = Destiny.mapping_mesures[j]()
                        self.__nom_mesures[j] = j
                        for k in j:
                            L = L + list(k)
        #print("Toutes les mesures")
        #print(self.toutes_les_mesures())
        self.inter=set()
        self.union=set()
       # print("Mapping des mesures :")
        ##print(Destiny.mapping_mesures)
        Destiny.nb_heuristiques =  len(Destiny.mesures_consideres)
        Destiny.maxH = Destiny.nb_heuristiques
        self.liste_mesures = []
        for i in self.__nom_mesures:
            self.liste_mesures = self.liste_mesures + list(self.__nom_mesures[i])


    def get_model(self):
        return self.model
    def get_model_name(self):
        return self.model_name

    def set_model_name(self,modele):
        self.model_name=modele

    def set_model(self,modele):
        self.model=modele


    def fit(self, X, Y):
        self.__data, self.__target = X, Y
        if (not Destiny.stop):
            #print("hh")
            for i in self.__mesures.keys():
                self.__mesures[i].fit(X, Y)
                #print(i, " fini")
                if (self.subsetgenerated == None):
                    self.subsetgenerated = self.__mesures[i].CreateSubsets(pourcentage=Destiny.percentage_feature)
                    self.__subset_calculator = self.__mesures[i].getCalculator()
                else:
                    self.__mesures[i].setSubsets(self.subsetgenerated)
        cpt = 0
        if (not Destiny.stop):
            for j in range(0, len(self.mesures_consideres)):
                self.__mesures_anterieure.update(self.getMegaHeuristique(["H" + str(cpt)], 1))
                cpt = cpt + 1
        if (self.seuillage == "union_intersection" and not Destiny.stop):
            self.activer_treshold()
        if (self.seuillage == "complexity" and not Destiny.stop):
            T = Tresholding()
            T.fit(self.__data, self.__target)
            T.getTreshold(self.__data, self.__target, Destiny.coef1, Destiny.coef2, Destiny.coef3)
        if (self.seuillage == "manual" and not Destiny.stop):
            self.ThresholdMeasures(self.__Threshold)
        if (self.seuillage == "unique" and not Destiny.stop):
            self.treshold_unique()
        self.union_intersection2(self.__Threshold)




    #L est une liste de types de mesures par exemple ['C' ,'Ce', 'De']
    def GestionSubsets(self,L,Borne = None):
        for i in L:
            self.__mesures[i].CreateSubsets(Borne)

    def get_mesures_anterieures(self):
        return self.__mesures_anterieure

    def GestionPathMesures(self):
        from Calculs import FCC,FCS,ChiMesure,FScore,ReliefF,RST,SklearnClassifiersPrecisionOtO,EntropyMeasures,Chisquare
        L = [FCC,FCS,ChiMesure,FScore,ReliefF,RST,SklearnClassifiersPrecisionOtO,EntropyMeasures,Chisquare]
        for a in (L):
            Destiny.mapping_mesures[tuple (a.noms_mesures)] = a.classe_mesure

    def clear(self):
        Destiny.mesures_consideres.clear()
        Destiny.mapping_mesures.clear()
        self.__mesures_anterieure.clear()
        self.liste_mesures.clear()
        self.__data, self.__target = None, None
        self.__max_iterations = 10
        self.__Threshold = 0
        self.subsetgenerated = None
        self.__subset_calculator = None
        self.__mesures_anterieure = {}
        self.inter = set()
        self.union = set()
        L = []
        for i in Destiny.mesures_consideres:
            if not (i in L):
                for j in Destiny.mapping_mesures:
                    if (i in j):
                        self.__mesures[j] = Destiny.mapping_mesures[j]()
                        self.__nom_mesures[j] = j
                        for k in j:
                            L = L + list(k)
        for i in self.__nom_mesures:
            self.liste_mesures = self.liste_mesures + list(self.__nom_mesures[i])




    def stop(self,v):
        self.stop=v

    def getMesure(self,nom):
        return self.__mesures[nom]

    def set_treshold(self,t):
        self.__Threshold=t

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
        #DE = self.__mesures[tuple(["RST"])].dependence(subset)
        Co = self.__mesures[tuple(['FCC'])].fcc(subset)
        return (D,I,Co)

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


    def refresh_dict(self,config):
        self.seuillage=config.get_destiny()["seuillage"]
        self.__max_iterations=config.get_destiny()["max_iterations"]
        self.__Threshold= config.get_destiny()["treshold"]
        Destiny.mesures_consideres=config.get_destiny()["mesures"]
        Destiny.nb_heuristiques = len(Destiny.mesures_consideres) - 1
        Destiny.maxH = Destiny.nb_heuristiques - 1
        self.coef1=config.get_destiny()["coef1"]
        self.coef2=config.get_destiny()["coef2"]
        self.coef3=config.get_destiny()["coef3"]
        self.modele = Destiny.model_dict[config.get_destiny()["modele"]]()
        self.__init__(self.model_name,self.seuillage)




    def MinimumRMaxS(self,subset,mc):
        masque = self.__data.shape[1]*[0]
        for i in subset:
            masque[i] = 1
        masque = np.array(masque)
        return masque.transpose().dot(self.__matrices_redondaces[mc]).dot(masque)-masque.dot(self.__matrices_importances[mc])


    def setMatricesImportanceRedondance(self,data,target):
        self.__mesures[("Entropie" , 'GainInformation' , "GainRatio" , "SymetricalIncertitude" ,
                          "MutualInformation" , "UH" , "US" , "DML")].fit (data=data,target=target)
        data = data.transpose ()
        d = list (data)
        d.append (target)
        d = np.array (d)
        c = np.corrcoef (d)
        c = c.transpose()
        self.__matrices_importances["Distance"] = c[-1]
        self.__matrices_redondaces["Distance"] = c[0:-1]
        self.__matrices_redondaces["Distance"] = self.__matrices_redondaces["Distance"].transpose()[:-1]
        self.__matrices_importances["Distance"] = self.__matrices_importances["Distance"].transpose ()[:-1]
        f = self.__mesures[("Entropie" , 'GainInformation' , "GainRatio" , "SymetricalIncertitude" ,
                          "MutualInformation" , "UH" , "US" , "DML")]
        m = np.ones((len(data),len(data)))
        imp = np.ones((len(data)))
        for i in range(0,len(data)):
            C1 = f.getEntropy([i])
            for j in range(0,len(data)):
                C2 = f.getEntropy([j])
                C3 = f.getEntropySachant([j],[i])
                m[i][j] = C1 + C2 - C3
            C2 = f.getEntropy([-1])
            C3 = f.getEntropySachant([i],[-1])
            imp[i] = C1 + C2 - C3
        self.__matrices_redondaces["Information"] = m
        self.__matrices_importances["Information"] = imp
        unit = np.ones(self.__matrices_importances["Distance"].shape[0])
        for i in self.__matrices_importances:
            self.__matrices_importances[i] = unit - self.__matrices_importances[i]/self.__matrices_importances[i].sum()
        return self.__matrices_redondaces,self.__matrices_importances


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




    def getMatriceImportanceRedondance(self):
        return self.__matrices_redondaces,self.__matrices_importances

    def test(self):
        D = {}
        for i in self.__mesures.keys():
            print("Utilisation de ", i)
            D1 = self.__mesures[i].rank_with(n=1)
            D2 = self.__mesures[i].rank_with(n=2)
            D3 = self.__mesures[i].rank_with(n=3)
            for i in D3:
                print(" i = ",i)
                for j in D3[i]:
                    print(j , " len= ",len(D3[i][j])," : " , D3[i][j])

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
            if (len(self.inter) > 0):
                self.inter = self.inter.intersection(elus)
            else:
                self.inter = elus
            self.union = self.union.union(elus)

    def evaluer(self):
        E = Evaluateur_Precision(self.__data, self.__target)
        E.train(SVC(gamma="auto"))
        if(len(self.inter)>0):
            a=(self.reguler_par_complexote(E.Evaluer(list(self.inter)),len(self.inter))+self.reguler_par_complexote(E.Evaluer(list(self.union)),len(self.union)))/2
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
        ep.train(self.model)
        D = self.attributs_qualitatifs(t)
        D = D[h]
        a=ep.Evaluer(D)
        precision = a
        return precision

    def treshold_unique(self):
        gt=0.5
        max=0
        for j in range(Destiny.maxH):
            t = 0.5
            alpha = 0.4
            for i in range(self.__max_iterations):
                p1 = t + alpha
                p2 = t - alpha
                arg1=self.criteron_heursitique_unique(j,((t + p1) / 2))
                arg2=self.criteron_heursitique_unique(j,((t + p2) / 2))
              #  print("les valeurs",arg2,(t + p2) / 2," I ",arg1,(t + p1) / 2)
                if (arg1 >= arg2):
                    t = (p1 + t) / 2
                    if (arg1 > max):
                        max = arg1
                        gt = t
                else:
                    t = (p2 + t) / 2
                    if (arg2 > max):
                        max = arg2
                        gt = t
                alpha = alpha / 2

        self.ThresholdMeasures(gt)

    def return_criteron(self,index,m):
        t = 0.5
        max=0
        alpha = 0.4
        for i in range(m):
            p1 = t + alpha
            p2 = t - alpha
            arg1 = self.criteron_heursitique_unique(index, ((t + p1) / 2))
            arg2 = self.criteron_heursitique_unique(index, ((t + p2) / 2))
            if (arg1 >= arg2):

                t = (p1 + t) / 2
                if (arg1 > max):
                    max = arg1
                    gt = t
            else:
                t = (p2 + t) / 2
                if (arg2 > max):
                    max = arg2
                    gt = t
            alpha = alpha / 2
        return max


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


