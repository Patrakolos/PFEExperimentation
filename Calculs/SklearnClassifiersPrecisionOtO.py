from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.ensemble import RandomForestClassifier , AdaBoostClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier

from Utilitaire.Mesure import Mesure
from Utilitaire.Evaluateur_Precision import Evaluateur_Precision



class SklearnClassifiersPrecisionOtO (Mesure):
    # liste_modeles = ["BN","RF","LSVM","RBFSVM","GaussianProcess","AdaBoost","QDA","KNN","DTC","MLP"]
    noms_mesures = ["BN" , "RF" , "KNN" , "AdaBoost"]  # Minimum viable
    fitted = len(noms_mesures)*[False]

    def __init__(self):
        super ().__init__ ()
        self.__data = None
        self.__target = None
        self.__evaluateurs = {}
        self._liste_mesures = SklearnClassifiersPrecisionOtO.noms_mesures


    def ranking_function_constructor(self , motclef):
        if (SklearnClassifiersPrecisionOtO.fitted[SklearnClassifiersPrecisionOtO.noms_mesures.index (motclef)] == False):
            self.setup_modeles (self.__data , self.__target , [motclef])
        return self.__evaluateurs[motclef].Evaluer

    def fit(self , data , target):
        super ().fit (data , target)
        self.__data = data
        self.__target = target
        self.setup_modeles (data , target,SklearnClassifiersPrecisionOtO.noms_mesures)




    def print_scores(self):
        for i in self.__evaluateurs:
            print (i + " : " + str (self.__evaluateurs[i].score ()))

    def print_multiples_scores(self):
        for i in self.__evaluateurs:
            print (i + " : " + str (self.__evaluateurs[i].vecteur_precision ()))

    def setup_modeles(self , data , target,L):
        self.__data = data
        self.__target = target
        for m in L:
            if(m in SklearnClassifiersPrecisionOtO.noms_mesures):
                self.__evaluateurs[m] = Evaluateur_Precision (data , target)
                self.__evaluateurs[m].train (SklearnClassifiersPrecisionOtO.modele_generator (m))
                SklearnClassifiersPrecisionOtO.fitted[SklearnClassifiersPrecisionOtO.noms_mesures.index(m)] = True

    @staticmethod
    def modele_generator(motclef):
        if (motclef == "BN"):
            return GaussianNB ()
        elif (motclef == "DTC"):
            return DecisionTreeClassifier ()
        elif (motclef == "LSVM"):
            return SVC (kernel='linear')
        elif (motclef == "RBFSVM"):
            return SVC (kernel='rbf')
        elif (motclef == "GaussianProcess"):
            return GaussianProcessClassifier ()
        elif (motclef == "AdaBoost"):
            return AdaBoostClassifier ()
        elif (motclef == "QDA"):
            return QuadraticDiscriminantAnalysis ()
        elif (motclef == "KNN"):
            return KNeighborsClassifier ()
        elif (motclef == "RF"):
            return RandomForestClassifier (n_estimators=10)
        elif (motclef == 'MLP'):
            return MLPClassifier ()

    def masquer(self , masque):
        for k in self.__evaluateurs.keys ():
            self.__evaluateurs[k].masquer (masque)

#Mention de fin de fichier
noms_mesures = ["BN" , "RF" , "KNN" , "AdaBoost"]
classe_mesure = SklearnClassifiersPrecisionOtO

