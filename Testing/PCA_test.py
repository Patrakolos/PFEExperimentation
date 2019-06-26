import re

from sklearn.svm import SVC

from Utilitaire.Evaluateur_Precision import Evaluateur_Precision
from Calculs import Destin
from DataSets import load_promoters_dataset, german_dataset, Interface_Datasets
from sklearn.ensemble import AdaBoostClassifier
import matplotlib.pyplot as plt
from NaturePackage import Fabriquant as fb
from NaturePackage import Genome
from NaturePackage import Nature
from Interface import Configuration
import numpy as np

#dat=Interface_Datasets.data_map["German"]
#dat.loadDescription()
#dat.dataMung(True)
#data,target=dat.getDataTarget()
#E=Evaluateur_Precision(data,target)
#E.train(AdaBoostClassifier())
#
#
#color=["red","blue","yellow","green"]
#LL=[]
#Destin.Destiny.mesures_consideres = ["FScore"]
#D = Destin.Destiny("Adaboost", "manual")
#D.set_treshold(0.95)
#Destin.Destiny.stop = False
#
#D.fit(data,target)
#gene="1H0/"
#g=Genome.Genome()
#L=[]
#for i in range(25):
#    gene=gene+"1H0/"
#    g.identity=gene
#    f=fb.Fabriquant(g,D,False)
#    L.append(E.Evaluer(f.genome.resultat))
#print(L)
##plt.plot([0.7296666666666668, 0.7247272727272728, 0.7337474747474747, 0.7517373737373737, 0.7637878787878789, 0.7637878787878789, 0.7637878787878789],color="green")
##plt.plot([0.7296666666666668, 0.7297272727272728, 0.7327575757575758, 0.7457474747474747, 0.7687575757575758, 0.7687575757575758, 0.7687575757575758],color="blue")
##plt.plot([0.7227272727272728, 0.7497272727272728, 0.7407474747474747, 0.7667575757575757, 0.7647979797979799, 0.7647979797979799, 0.7647979797979799],color="red")
##plt.ylabel('Accuracy')
##plt.xlabel('Iterati,on')
##plt.title("GainRatio - Généralisation de l'entropie")
##plt.savefig('GainRatio.png')
#plt.plot(L)
#plt.show()

#modele chaine de caractere
#le gene se gait avec des H commencant par 0; qui sont l'indice de la liste mesure
#nom du dataset
#bourrer si bourage
#bourrage union_intersection unique manual
#eviter de mettre 1 en seuil
#un gene se finit par /
def gene_eval(gene,list_mesures,modele,dataset,bourrer=False,seuillage="manual",seuil=0.99):

    Destin.Destiny.stop=False
    Destin.Destiny.mesures_consideres=list_mesures
    D = Destin.Destiny(modele,seuillage)
    if(seuillage=="manual"):
        D.set_treshold(seuil)
    dat = Interface_Datasets.data_map[dataset]
    dat.loadDescription()
    dat.dataMung(True)
    data, target = dat.getDataTarget()
    D.fit(data, target)
    E = Evaluateur_Precision(data, target)
    E.train(Destin.Destiny.model_dict[modele]())
    g = Genome.Genome()
    g.identity = gene
    f=fb.Fabriquant(g,D,bourrer)
    return E.Evaluer(f.genome.resultat)


def gene_show_variations(gene,list_mesures,modele,dataset):
    Destin.Destiny.stop = False
    Destin.Destiny.mesures_consideres = list_mesures
    D = Destin.Destiny(modele, "manual")
    dat = Interface_Datasets.data_map[dataset]
    dat.loadDescription()
    dat.dataMung(True)
    data, target = dat.getDataTarget()
    D.fit(data, target)
    E = Evaluateur_Precision(data, target)
    E.train(Destin.Destiny.model_dict[modele]())
    exp = "\dH.*?/"
    a = re.findall(exp, gene)
    genne = ""
    gg = Genome.Genome()
    L = []
    buf=0
    for i in a:
        genne = genne + i
        gg.identity = genne
        f = fb.Fabriquant(gg, D, False)
        e=E.Evaluer(f.genome.resultat)
        L.append(e-buf)
        buf=e
    plt.plot(L[1:], color='red')
    plt.title("Custom heuristic")
    plt.ylabel("Accuracy")
    plt.xlabel("Iteration")
    plt.show()
    return L
def gene_show_variations(gene,list_mesures,modele,dataset):
    Destin.Destiny.stop = False
    Destin.Destiny.mesures_consideres = list_mesures
    D = Destin.Destiny(modele, "manual")
    dat = Interface_Datasets.data_map[dataset]
    dat.loadDescription()
    dat.dataMung(True)
    data, target = dat.getDataTarget()
    D.fit(data, target)
    E = Evaluateur_Precision(data, target)
    E.train(Destin.Destiny.model_dict[modele]())
    exp = "\dH.*?/"
    a = re.findall(exp, gene)
    genne = ""
    gg = Genome.Genome()
    L = []
    buf=0
    for i in a:
        genne = genne + i
        gg.identity = genne
        f = fb.Fabriquant(gg, D, False)
        e=E.Evaluer(f.genome.resultat)
        L.append(e-buf)
        buf=e
    plt.plot([0]*(len(L)-1), alpha=0.2)
    plt.plot(L[1:], color='red')
    plt.title("Custom heuristic")
    plt.ylabel("Accuracy")
    plt.xlabel("Iteration")
    plt.show()
    return L

def gene_show_evolution(gene,list_mesures,modele,dataset):
    Destin.Destiny.stop = False
    Destin.Destiny.mesures_consideres = list_mesures
    D = Destin.Destiny(modele, "manual")
    dat = Interface_Datasets.data_map[dataset]
    dat.loadDescription()
    dat.dataMung(True)
    data, target = dat.getDataTarget()
    D.fit(data, target)
    E = Evaluateur_Precision(data, target)
    E.train(Destin.Destiny.model_dict[modele]())
    exp = "\dH.*?/"
    a = re.findall(exp, gene)
    genne = ""
    gg = Genome.Genome()
    L = []
    buf=0
    for i in a:
        genne = genne + i
        gg.identity = genne
        f = fb.Fabriquant(gg, D, False)
        e=E.Evaluer(f.genome.resultat)
        L.append(e)
    plt.plot([max(L)]*len(L), alpha=0.2)
    plt.plot(L, color='purple')
    plt.title("Custom heuristic")
    plt.ylabel("Accuracy")
    plt.xlabel("Iteration")
    plt.show()
    return L


def test_unique(heuristic,modele,dataset,max=1):
    Destin.Destiny.stop = False
    Destin.Destiny.mesures_consideres = [heuristic]
    D = Destin.Destiny(modele, "manual")
    dat = Interface_Datasets.data_map[dataset]
    dat.loadDescription()
    dat.dataMung(True)
    data, target = dat.getDataTarget()
    D.fit(data,target)
    gene = str(max)+"H0/"
    g=Genome.Genome()
    L=[]
    E = Evaluateur_Precision(data, target)
    E.train(Destin.Destiny.model_dict[modele]())
    for i in range(int(data.shape[1]/max+1)):
       gene=gene+str(max)+"H0/"
       g.identity=gene
       f=fb.Fabriquant(g,D,False)
       L.append(E.Evaluer(f.genome.resultat))
    plt.plot(L,color='green')
    plt.title(heuristic)
    plt.ylabel("Accuracy")
    plt.xlabel("Iteration")
    plt.show()
    return L

def exploration_ratio(max_iterations,config,dataset):
    Destin.Destiny.stop=False
    Nature.Nature.stop=False
    D=Destin.Destiny("BN")
    D.refresh_dict(config)
    dat = Interface_Datasets.data_map[dataset]
    dat.loadDescription()
    dat.dataMung(True)
    data, target = dat.getDataTarget()
    D.fit(data, target)
    Nature.Nature.refresh_dict(config)
    Nature.Nature.init(D)
    Y=[]
    for i in range(max_iterations):
        Nature.Nature.evolve()
        Y.append(len(Nature.Nature.condid)/(4+(Nature.Nature.maxP*(i+1))))
    plt.plot(Y, color='green')
    plt.title("Exploration Ratio")
    plt.ylabel("Accuracy")
    plt.xlabel("Iteration")
    plt.show()
    return Y







gnene="1H1H1/1H2/1H3/1H3/1H3/1H3/1H3/1H3/1H0/1H3/1H3/1H0/1H1/1H2/1H0/1H2/1H3/"
lise=["FScore","GainRatio" ,"FCC","DML"]
modele="AdaBoost"
dataset="German"
c=Configuration.Configuration()
#print(gene_show_evolution(gnene,lise,modele,dataset))
print(exploration_ratio(25,c,"German"))
#test_unique("UH","AdaBoost","German",1)
max([0.89,1.34,3.24,6.45,7.24])





