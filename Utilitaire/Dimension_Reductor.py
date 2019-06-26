import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

class Dimension_Reductor(object):
    def __init__(self):
        self.__feature_scores = None
        self.__vectors = None
        self.__data = None
        self.__target = None


    def fit(self, X,Y):
        P = PCA()
        self.__data = X
        self.__target = Y
        P.fit(X,Y)
        self.__feature_scores , self.__vectors = P.explained_variance_ratio_,P.components_


    def getPCA(self,liste_nums):
        D = self.__data.transpose()
        masque = np.array(len(D)*[False])
        for i in liste_nums:
            masque[i] = True
        D = D[masque]
        D = D.transpose()
        P = PCA()
        P.fit(D,self.__target)
        c = P.components_[0]
        A = D.dot(c)
        return A


    def getLDA(self,liste_nums):
        D = self.__data.transpose()
        masque = np.array(len(D)*[False])
        for i in liste_nums:
            masque[i] = True
        D = D[masque]
        D = D.transpose()
        P = LinearDiscriminantAnalysis()
        P.fit(D,self.__target)
        c = P.coef_[0]
        A = D.dot(c)



    def Score(self):
        L = len(self.__vectors[0])*[0]
        A = np.array(L)
        for i in range(0,len(self.__vectors)-1):
            I = np.array(self.__vectors[i])
            I = self.__feature_scores[i] * I
            A = A + I
        return A



