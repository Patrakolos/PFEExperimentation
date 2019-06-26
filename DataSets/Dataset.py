import os
import urllib
from os import F_OK
import requests
import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from Utilitaire.Evaluateur_Precision import Evaluateur_Precision

def emptyFunction(data,target):
    return data,target

class Dataset():


    def __init__(self,nom):
        self._nom = nom
        self._lien_web = ""
        self._data = None
        self._target = None
        self._categorical_attributes = []
        self._categorical_ordered_attributes = []
        self._precisions_brutes = {}
        self._nb_samples = 0
        self._nb_features  = 0
        self.__dataMungingFunction = emptyFunction
        self.__explication_attributs = {}
        self.__nom_attributs = {}

    def download(self,_lien_web):
        if(not os.access(r"../Local_Datasets/_"+self._nom+".csv",F_OK)):
            print("Création de la ressource : ", self._nom)
            f = open(r"../Local_Datasets/_"+self._nom+".csv","w")
            print("Téléchargement de la ressource : ",self._nom," de l'url : ",_lien_web)
            user_agent = {'User-agent': 'Mozilla/5.0'}
            r = requests.get(_lien_web,headers = user_agent)
            f.write(str(r.text.replace(" ",",")))
        else:
            print("Dataset déja Téléchargé, il est accessible a l'URL : ")


    def describeAttributs(self,dict):
        self.__explication_attributs = {}
        self.__nom_attributs ={}
        self._categorical_attributes = dict["Categorique"]
        self._categorical_ordered_attributes = dict["Categorique_Ordre"]
        if("Explication" in dict.keys()):
            self.__explication_attributs = dict["Explication"]
        if ("Nom" in dict.keys()):
            self.__nom_attributs = dict["Nom"]

    def getDesc(self):
        return self._categorical_attributes ,self._categorical_ordered_attributes,self.__nom_attributs ,self.__explication_attributs

    def storeDescription(self):
        import json
        D = {}
        if(self._categorical_attributes != None):
            D["Categorique"] = self._categorical_attributes
        if (self._categorical_attributes != None):
            D["Categorique_Ordre"] = self._categorical_ordered_attributes
        if (self.__explication_attributs != None):
            D["Explication"] = self.__explication_attributs
        if (self.__nom_attributs != None):
            D["Nom"] = self.__nom_attributs
        if (not os.access(r"../Dataset_Metadatas/_" + self._nom + ".json", F_OK)):
            print("Création du fichier de description : ", self._nom)
            f = open(r"../Dataset_Metadatas/_" + self._nom + ".json", "w")
            f.write(json.dumps(D))
        else:
            print("Description déja existante")


    def loadDescription(self):
        import json
        if ( os.access(r"../Dataset_Metadatas/_" + self._nom + ".json", F_OK)):
            f = open(r"../Dataset_Metadatas/_" + self._nom + ".json", "r")
            D = json.load(f)
            self.describeAttributs(D)
        else:
            print("Description non enregistrée")




    def getAttributDescription(self,attribut):
        D = {}
        D["type"] = "Continue"
        if(attribut in self._categorical_attributes):
            D["type"] = "Categorique"
        if (attribut in self._categorical_ordered_attributes):
            D["type"] = "Categorique_Ordre"
        if(str(attribut) in self.__explication_attributs):
            D["Explication"] = self.__explication_attributs[str(attribut)]
        if (str(attribut) in self.__nom_attributs):
            D["Nom"] = self.__nom_attributs[str(attribut)]
        return D

    def setDataMungingFunction(self,fun):
        self.__dataMungingFunction = fun


    def getNom(self):
        return self._nom

    def getShape(self):
        return self._nb_samples,self._nb_features



    def addToLocal(self,path):
        if (not os.access(r"../Local_Datasets/_" + self._nom + ".csv", F_OK)):
            if(os.access(path,F_OK)):
                print("Création de la ressource : ", self._nom)
                f = open(r"../Local_Datasets/_" + self._nom + ".csv", "w")
                f2 = open(path,"r")
                f.write(f2.read().replace(" ",","))
            else:
                print("Fichier a ajouter introuvable")
        else:
            c = input("Dataset déja Téléchargé, voulez vous ajouter les données du fichiers aux données existantes ? (O/N)")
            if(c == "O"):
                f = open(r"../Local_Datasets/_" + self._nom + ".csv", "a")
                f2 = open(path, "r")
                f.write(f2.read().replace(" ",","))

    def dataMung(self,transform = False):
        if (os.access(r"../Local_Datasets/_" + self._nom + ".csv", F_OK)):
            df = pd.read_csv(r"../Local_Datasets/_" + self._nom + ".csv")
            arr = np.array(df)
            arr = arr.transpose()
            self._data = arr[:-1].transpose()
            self._target = arr[-1].transpose()
            self._target = self._target.astype('int')
            self._nb_features = self._data.shape[1]
            self._nb_samples = self._data.shape[0]
        if(transform):
            self._data,self._target = self.transform(self._data,self._target)
        return self._data,self._target

    def transform(self,data,target):
        self.LAencodeAttributs()
        self.OHencodeAttributs()
        return self.__dataMungingFunction(data,target)

    def benchMarkDataset(self,modele):
        EP = Evaluateur_Precision(self._data, self._target)
        EP.train(modele)
        print(EP.score())

    def LAencodeAttributs(self):
        LABC = LabelEncoder()
        for i in self._categorical_attributes:
            self._data[:, i] = LABC.fit_transform(self._data[:, i])

    def getDataTarget(self):
        return self._data,self._target

    def OHencodeAttributs(self):
        if(len(self._categorical_ordered_attributes) > 0):
            LABC = OneHotEncoder(categorical_features=self._categorical_ordered_attributes)
            self._data = LABC.fit_transform(self._data).toarray()


