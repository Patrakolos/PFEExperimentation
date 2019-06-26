#import random
import math
from itertools import combinations

import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.ensemble import RandomForestClassifier , AdaBoostClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import RFECV
from sklearn.datasets import make_classification
import numpy as np
import random



class Embedded_Thresholding:


    borne_complexite = 3000

    def __init__(self):
        self.__modele = AdaBoostClassifier()
        self.__data = None
        self.__data_masque = None
        self._rfecv = None
        self.__target = None
        self.__threshold = 0
        self.__subset_selectionned = {}
        self.__nbfeatures = 0
        self.__matrices_redondaces, self.__matrices_importances = {}, {}


    def setMatrices(self,m1,m2):
        self.__matrices_importances = m2
        self.__matrices_redondaces = m1

    def fit(self,X,Y):
        self.__data = X
        self.__target = Y
        self.__data_masque = X
        #rfecv = RFECV (estimator=self.__modele , step=1 , cv=StratifiedKFold (2) ,
        #               scoring='accuracy')
        #rfecv.fit(self.__data,self.__target)
        self.__nbfeatures = len(self.__data.transpose())
        self.__threshold = 100


    def getThresholdEmbedded(self,modele):
        rfecv = RFECV (estimator=modele , step=1 , cv=StratifiedKFold (2) ,
                       scoring='accuracy')
        rfecv.fit(self.__data,self.__target)
        return rfecv.n_features_


    def compute_subset(self,liste_chiffre):
        if not (tuple(liste_chiffre) in self.__subset_selectionned.keys()):
            self.__data_masque = self.__data.transpose ()
            masque = np.array (len (self.__data_masque) * [False])
            for i in liste_chiffre:
                masque[i] = True
            self.__data_masque = self.__data_masque[masque]
            self.__data_masque = self.__data_masque.transpose ()
            rfecv = RFECV (estimator=self.__modele , step=1 , cv=StratifiedKFold (2) ,
                           scoring='accuracy')
            rfecv.fit (self.__data_masque , self.__target)
            self.__subset_selectionned[tuple(liste_chiffre)] = rfecv.ranking_,rfecv.grid_scores_
        return self.__subset_selectionned[tuple(liste_chiffre)]





    def Energie(self,L,mc="Distance"):
        masque = [0] * self.__nbfeatures
        for i in L:
            masque[i] = 1
        masque = np.array(masque)
        try:
           e =  masque.transpose().dot(self.__matrices_redondaces[mc]).dot(masque) - masque.dot(
                self.__matrices_importances[mc])
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


    def Alteration_Insensification(self,L,mc="Distance"):
        max = -1
        min_i = -1
        for i in L:
            if(self.__matrices_importances[mc][i] > max):
                max = self.__matrices_importances[mc][i]
                min_i = i
        max_r = -1
        rempl = None
        for j in L:
            if(self.__matrices_redondaces[mc][min_i][j] > max_r):
                rempl = j
                max_r = self.__matrices_redondaces[mc][min_i][j]
        NL = L
        new_att = random.randint(0,self.__nbfeatures-1)
        while(new_att in L):
            new_att = random.randint(0, self.__nbfeatures - 1)
        NL[NL.index(rempl)] = new_att
        return NL


    def generer_subset(self,taille,borne = None):
        self.__subset_selectionned[taille] = []
        L = self.GenererListeRandom(taille)
        e = self.Energie(L)
        T = 1
        booll = False
        mc_courant = "Distance"
        emin_d,emin_mutinf = 1000, 1000
        nb_iterations_pas = 1000000
        cptcpt = nb_iterations_pas
        pas = 0.01
        coef_decrementation_temperature = 0.99
        cpt_iter = 0
        while (cpt_iter < 100000):
            cpt_iter = cpt_iter + 1
            # Mecanisme d'intensification
            NL = self.Alteration_Insensification(L, mc_courant)
            while(NL in self.__subset_selectionned[taille]):
                booll = not booll
                if(booll == True):
                    if(mc_courant == "Distance"):
                        mc_courant = "Information"
                    else:
                        mc_courant = "Distance"
                    NL = self.Alteration_Insensification(L,mc_courant)
                else:
                    NL = self.GenererListeRandom(taille)
            # Mise a jour de la température
            T = T * coef_decrementation_temperature
            if (cptcpt == 0):
                cptcpt = nb_iterations_pas
                T = T - pas
            # Mise a jour de l'energie
            ep = e
            enew = self.Energie(NL,mc_courant)
            if (enew <= emin_d and mc_courant=="Distance"):
                L = NL
                e = enew
            elif(enew <= emin_mutinf and mc_courant=="Information"):
                L = NL
                e = enew
            else:
                try:
                    p = 0
                    if(mc_courant == "Distance"):
                        p = math.exp(-((emin_d - e) / T))
                    if (mc_courant == "Information"):
                        p = math.exp(-((emin_mutinf - e) / T))
                except(OverflowError):
                    continue
                p = int(p * 100)
                rr = random.randint(0, 100)
                if (p > rr):
                    if ( mc_courant == "Distance"):
                        L = NL
                        emin_mutinf = enew
                    elif ( mc_courant == "Information"):
                        L = NL
                        emin_d = enew
            # Selectionner l'ensemble courant
            self.__subset_selectionned[len(L)].append(L)
            if(len(self.__subset_selectionned[len(L)])==borne):
                return self.__subset_selectionned[len(L)]
        return self.__subset_selectionned[len(L)]

