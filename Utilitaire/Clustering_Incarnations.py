from itertools import combinations

import time
from NaturePackage import Nature as nat
from Calculs.Destin import Destiny
from sklearn.cluster import k_means
from Utilitaire.Evaluateur_Precision import Evaluateur_Precision
from sklearn.linear_model import LinearRegression


class Clustering_Incarnations:
    model=LinearRegression()
    data=[]
    target=[]
    nb_samples=0
    tolerance=10
    def __init__(self):
        self.__population = None
        self.__projections = []
        self.__destiny = None
        self.clusters = {}
        self.alphas_locaux = []
        self.actul_score=0

    def fit(self,X,Y):
        self.__destiny.fit(X,Y)

    def ajouter_population(self,X):
        self.__population = X


    def setDestiny(self,D):
        self.__destiny = D

    def projeter(self):
        self.__projections = []
        print("population :",self.__population)
        for i in self.__population:
            k=time.time()
            self.__projections.append(self.__destiny.Projection(i))
            k=time.time()-k
        print("temps d'une projection :",k)
        return self.__projections


    @classmethod
    def rafraichir_poids(cls,donnee,precision,D):
        cls.data.append(D.Projection(donnee))
        cls.target.append(precision)
        cls.nb_samples=cls.nb_samples+1
        if(cls.nb_samples>cls.tolerance):
            cls.model.fit(cls.data,cls.target)
        pass
    @classmethod
    def carreProjection(cls,projection,D):
        p=D.Projection(projection)
        s=0
        if(cls.nb_samples<cls.tolerance):
            for i in p:
                s = s + i*i
        else:
            s=cls.model.predict([p])
        #print("___Prediction",s)
        D=Destiny(D.get_model_name())
        return D.reguler_par_complexote(s,len(projection))


    def maxCarreProjection(self,liste_projections,D):
        m = 0
        im = liste_projections[0]
        for i in set(liste_projections):
            self.actul_score = self.actul_score + Clustering_Incarnations.carreProjection(i,D)
            if((i not in nat.Nature.alphas_locaux) and (i not in nat.Nature.condid)):
                if (Clustering_Incarnations.carreProjection(i,D) >= m):
                    m = Clustering_Incarnations.carreProjection(i,D)
                    im = i
                nat.Nature.condid.append(i)
                print("Evaluation: l: ", i, " Eval: ", m)
        return im

    def maxprecision(self,liste_projection):
        max=0
        im=liste_projection[0]
        E = Evaluateur_Precision(nat.Nature.DM.getDataset()[0], nat.Nature.DM.getDataset()[1])
        E.train(nat.Nature.modele)
        for i in set(liste_projection):
            self.actul_score=self.actul_score+E.Evaluer(i)
            if ((i not in nat.Nature.alphas_locaux) and (i not in nat.Nature.condid)):
                kk=E.Evaluer(i)
                if ( kk>= max):
                    max = kk
                    im = i
                nat.Nature.condid.append(i)
                print("Evaluation: l: ", i," Eval: ",kk)


        return im


    def clusteriser(self,train,evolve,D):
        print("train",train)
        print("set",set(self.__projections))
        if(evolve):
            KM = k_means(self.__projections,n_clusters=nat.Nature.nb_cluster)
        else:
            KM = k_means(self.__projections, n_clusters=4)
        cpt = 0
        Rez = {}
        for i in KM[1]:
            W = []
            t = tuple(self.__population[cpt])
            W.append(t)
            Rez[i] = Rez.get(i,[]) + W
            cpt = cpt + 1
        #for i in Rez:
            #print(i , " : " , Rez[i])
        self.clusters = Rez
        print("Rez",Rez)
        self.alphas_locaux = (len(self.clusters.keys()))*[0]
        self.actul_score=0
        print("Alpha locaux",self.alphas_locaux)
        for i in self.clusters:
            if not train:
                C = self.maxCarreProjection(self.clusters[i],D)
            else:
                C=self.maxprecision(self.clusters[i])
            try:
                self.alphas_locaux[i] = C
            except(IndexError):
                print("Erreur car i=", i , " et self.__alphasLocaux=",self.alphas_locaux)


