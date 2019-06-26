import re
import random
import itertools

from Utilitaire.Clustering_Incarnations import Clustering_Incarnations
from Calculs import Destin as dest
from Utilitaire.Evaluateur_Precision import Evaluateur_Precision
from NaturePackage import Genome
from NaturePackage import Fabriquant as fb
import math
import time

from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

class Nature:
    #Hyper parametres:
    Psupp=0.3
    Pstop=0.4
    maxA = 3
    maxH= 6
    maxP = 200
    maxS = 3
    nb_promo=4
    alpha=0
    nb_cluster=5
    Tol = 3
    tol_evolutivite = 0.25
    scrutin="Condorcet"
    metric="accuracy"
    modele = SVC(gamma='auto')
    evolve_strategies=True
    only_global_crossing=False
    random_initialisation=False
    train_all=False
    paralel_crossing=False
    strat = [[0.1, 0.0, 0.5, 0.02]]#, [0.5, 0.3, 0.5, 0.7], [0.3, 0.6, 0.5, 0.7]#]


    #Parametres de stockage
    condid=[]
    stop = False
    population = []
    actualalpha = None
    DM=None
    modjahidin=[]
    population_clusterised = {}
    alphas_locaux = []
    alpha_global = []
    actual_precision=0
    qualite=0
    PM=1
    evolutivite_inter=0
    actuel_score=0
    taille=0
    iteratore=0

    @classmethod
    def csm(cls,G0I,G0A,Strat):
        st=""
        exp="\d[H\d]+/"
        gi=re.findall(exp,G0I)
        ga=re.findall(exp,G0A)
        for s,s2 in itertools.zip_longest(gi,ga):
            if(s!=None):
                p = random.random()
                if(p<Strat[0]):
                    st=st+"S"
                else:
                    if(p<Strat[1]*cls.PM):
                        pp=random.random()
                        if(pp<Strat[2]):
                            st=st+"MI"
                        else:
                            st=st+"MO"
                    else:
                        if(s2 != None):
                            if(s[0]==s2[0]):
                                pp = random.random()
                                if (pp < Strat[3]):
                                    st = st + "CI"
                                else:
                                    st = st + "CO"
                            else:
                                st=st+"CO"
                        else:
                            st = st + "MO"
        return st

    @classmethod
    def Grand(cls, G=None):
        st = ""
        if (G is None):
            n = random.randint(1,cls.maxA)
            st = st + str(n)
        else:
            st = st + G[0]
        k = random.randint(0, cls.maxH-1)
        st = st + "H" + str(k)
        while(random.random()<cls.Pstop):
            k = random.randint(0, cls.maxH-1)
            st = st + "H" + str(k)
        return st

    @classmethod
    def MergeH(cls,gi, ga):
        return gi + ga[1:]

    @classmethod
    def PseudoTransoducteur(cls, GOI, GOA, CSM):
        st = ""
        exp = "(\d[H\d]+)/"
        exp2 = "S|CI|CO|MI|MO"

        gi = re.findall(exp, GOI)
        ga = re.findall(exp, GOA)
        csm = re.findall(exp2, CSM)
        modif = itertools.zip_longest(gi, ga, csm)
        boola=True
        for i, a, c in modif:
            if(random.random()<cls.Psupp or boola):
                boola=False
                if (gi != None):
                    if (c == "S"):
                        st = st + i + "/"
                    if (c == "MI"):
                        st = st + cls.Grand(i) + "/"
                    if (c == "MO"):
                        st = st + cls.Grand() + "/"
                    if (c == "CI"):
                        st = st + cls.MergeH(i, a) + "/"
                    if (c == "CO"):
                        st = st + a + "/"
        return st

    @classmethod
    def validate(cls, G,bourrer):
        if (len(G.identity) == 0):
            if(not cls.random_initialisation):
                if(cls.iteratore<cls.maxH-1):
                    pp=cls.iteratore
                    cls.iteratore=cls.iteratore+1
                else:
                    pp = random.randint(0,cls.maxH-1)
                stre="1H"+str(pp)+"/"
                for i in range(int(cls.DM.getNbAttribute()*cls.DM.getTreshold())):
                    stre=stre+"1H"+str(pp)+"/"
                G.identity=stre
            else:
                setr="1H1/"
                mute="MO"
                for i in range(int(cls.DM.getNbAttribute() * cls.DM.getTreshold())):
                    setr = setr + "1H1/"
                    mute=mute+"MO"
                G.identity = cls.PseudoTransoducteur(setr, "", mute)
        fab = fb.Fabriquant(G, cls.DM,bourrer,cls.scrutin)
        VG = fab.genome
        return VG

    @classmethod
    def monoevolv(cls, VGOI, VGOA, strat,bourrer):
        st = cls.csm(VGOI.identity, VGOA.identity, strat)
        GN = Genome.Genome()
        GN.identity = cls.PseudoTransoducteur(VGOI.identity, VGOA.identity, st)
        VGN = cls.validate(GN,bourrer)
        return VGN

    @classmethod
    def eludeAlpha(cls,evolve):
        a=time.time()
        P = []
        for i in cls.population:
            P.append(i.resultat)
        CI = Clustering_Incarnations()
        CI.setDestiny(Nature.DM)
        CI.ajouter_population(P)
        print("Time0 :", time.time() - a)
        CI.projeter()
        print("Time1 :", time.time() - a)
        CI.clusteriser(cls.train_all,evolve,cls.DM)
        print("Time2 :", time.time() - a)
        Nature.population_clusterised = CI.clusters
        Nature.alphas_locaux = CI.alphas_locaux
        print("Time3 :", time.time() - a)
        if(len(cls.modjahidin)==cls.nb_promo):
            cls.modjahidin.remove(cls.modjahidin[0])
            cls.modjahidin.append(cls.alphas_locaux)
        else: cls.modjahidin.append(cls.alphas_locaux)
        print("Time4 :",time.time()-a)
        E = Evaluateur_Precision(Nature.DM.getDataset()[0],Nature.DM.getDataset()[1])
        print("nature model",Nature.modele)
        E.train(Nature.modele)
        max=cls.qualite
        for i in Nature.alphas_locaux:
            print("alpha :",i)
            precision=E.Evaluer(i)
            Clustering_Incarnations.rafraichir_poids(i,precision,cls.DM)
            c=cls.DM.reguler_par_complexote(precision,len(i))
            if c > max:
                max = c
                cls.alpha_global = i
                cls.taille=len(i)
                cls.actual_precision=precision
            else:
                if(c==max):
                    if(len(i)<cls.taille):
                        cls.alpha_global=i
                        cls.taille=len(i)
                        cls.actual_precision=precision

        cls.qualite = max
        lesalpha=cls.alphas_locaux
        cls.alphas_locaux=[]
        for k in lesalpha:
            kk=0
            while(kk<len(cls.population)):
                if(cls.population[kk].resultat == list(k)):
                    cls.alphas_locaux.append(cls.population[kk])
                    kk=len(cls.population)+1
                kk=kk+1
        ll=0
        print("----------ALPHA GLOBAL",cls.alpha_global)
        while(ll<len(cls.population)):
            if (cls.population[ll].resultat == list(cls.alpha_global)):
                print("al'lpha",cls.population[ll].incarnation)
                cls.actualalpha=cls.population[ll]
                ll=len(cls.population)+1
            ll=ll+1
        cls.alter_strategies(CI,evolve)


    @classmethod
    def alter_strategies(cls,C,evolve):
        cls.evolutivite_inter=(C.actul_score - cls.actuel_score)/(cls.maxP*100)
        cls.actuel_score=C.actul_score
        if(evolve):
            if(cls.evolutivite_inter>4):
                cls.PM=max(cls.PM*(1-cls.tol_evolutivite),0.2)
            else:
                cls.PM=min(1.8,cls.PM*(1+cls.tol_evolutivite))



    @classmethod
    def init(cls,D):
        cls.stop=False
        cls.DM=D
        cls.modele=cls.DM.get_model()
        cls.maxH=len(cls.DM.mesures_consideres)
        cls.alpha=D.alpha
        cls.population=[]
        cls.iteratore=0
        VNG=Genome.Genome()
        for i in range(cls.maxP):
            if(not cls.stop):
                cls.population.append(cls.monoevolv(VNG,VNG,cls.strat[0],False))
        print("l'evolution check")
        if(not cls.stop):
            cls.eludeAlpha(False)


    @classmethod
    def evolve(cls):
        print("evolution")
        aa=0
        bb=0
        if(cls.paralel_crossing):
            cls.population = sorted(cls.population, key=lambda k: random.random())
            for i in range(int(cls.maxP/2)):
                cls.population[i] = cls.monoevolv(cls.population[i], cls.population[cls.maxP-i-1],
                                              cls.strat[random.randint(0, cls.maxS - 1)], True)
                cls.population[cls.maxP-i-1] = cls.monoevolv(cls.population[i], cls.population[i],
                                                  cls.strat[random.randint(0, cls.maxS - 1)], True)
        else:
            for i in range(cls.maxP):
                if(not cls.stop):
                    lp=time.time()
                    if(not cls.only_global_crossing):
                        cls.population[i]=cls.monoevolv(cls.population[i],cls.alphas_locaux[cls.getcluster(cls.population[i])],cls.strat[0],True)
                    aa=aa+(time.time()-lp)
                    lp=time.time()
                    cls.population[i] = cls.monoevolv(cls.population[i], cls.actualalpha, cls.strat[0],True)
                    bb=bb+(time.time()-lp)
            po=time.time()
        if(not cls.stop):
            cls.eludeAlpha(cls.evolve_strategies)


    @classmethod
    def getcluster(cls,G):
        cpt=0
        trouve=False
        while(cpt<len(cls.population_clusterised)):
            cpt2=0
            while(cpt2<len(cls.population_clusterised[cpt])):
                if(G.resultat == list(cls.population_clusterised[cpt][cpt2])):
                    return cpt
                    cpt2=len(cls.population_clusterised[cpt])+1
                    trouve=True
                cpt2=cpt2+1
            if(trouve):
                cpt = len(cls.population_clusterised) + 1
            cpt=cpt+1


        return -1

    @classmethod
    def clear(cls):
        cls.population.clear()
        cls.modjahidin.clear()
        cls.population_clusterised.clear()
        cls.alpha_global=None
        cls.alphas_locaux.clear()
        cls.qualite=0
        cls.actual_precision=0
        cls.actuel_score=0
        cls.PM=1
        cls.taille=0

    @classmethod
    def stop(cls,v):
        cls.stop=v

    @classmethod
    def refresh_dict(cls,config):
        cls.Psupp=config.get_nature()["Psupp"]
        cls.Pstop=config.get_nature()["Pstop"]
        cls.maxA=config.get_nature()["MaxA"]
        cls.maxP=config.get_nature()["MaxP"]
        cls.tol_evolutivite=config.get_nature()["tol_evolutivite"]
        cls.nb_promo=config.get_nature()["nb_promotions"]
        cls.nb_cluster=config.get_nature()["nb_cluster"]
        cls.Tol=config.get_nature()["Tol"]
        cls.only_global_crossing=config.get_nature()["only_global_crossing"]
        cls.train_all=config.get_nature()["train_all"]
        cls.scrutin=config.get_nature()["scrutin"]
        cls.evolve_strategies=config.get_nature()["evolve_strategies"]
        cls.random_initialisation=config.get_nature()["random_initialisation"]
        cls.modele=dest.Destiny.model_dict[config.get_destiny()["modele"]]()
        cls.metric=config.get_nature()["metric"]

    @classmethod
    def run(cls,D,treshold):
        cls.init(D)
        while(cls.condid/(cls.it*cls.maxP)>treshold):
            cls.evolve()
            print("Alpha :",cls.actualalpha)
        return cls.actualalpha



def sigmoid(x):
  return 1 / (1 + math.exp(-x))



