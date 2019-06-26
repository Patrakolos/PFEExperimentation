from sklearn.svm import SVC

from Utilitaire.Evaluateur_Precision import Evaluateur_Precision
from Calculs import Destin
from DataSets import load_promoters_dataset, german_dataset, Interface_Datasets
from sklearn.ensemble import AdaBoostClassifier
import matplotlib.pyplot as plt
from NaturePackage import Fabriquant as fb
from NaturePackage import Genome

dat=Interface_Datasets.data_map["German"]
dat.loadDescription()
dat.dataMung(True)
data,target=dat.getDataTarget()
E=Evaluateur_Precision(data,target)
E.train(AdaBoostClassifier())
#print(E.Evaluer([0, 1, 2, 4, 5, 11, 6]))
#print(E.Evaluer([4, 0, 19, 1, 5, 2, 14, 3, 11]))
#print(E.Evaluer([0, 1, 2, 4, 5, 11, 6]))
#print(E.Evaluer([19, 9, 17, 13, 18, 14, 15, 16, 0, 8, 5, 2, 7, 10, 11, 6]))
#print(E.Evaluer([4, 1, 2, 12, 3, 0, 5, 6, 7, 8, 9]))
#print(E.Evaluer((0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 19)))

D=Destin.Destiny("Adaboost","manual")
D.set_treshold(0.95)
Destin.Destiny.stop=False
#data,target=german_dataset.load_german_dataset()
D.fit(data,target)
D.mesures_consideres=["FScore"]
#L=D.getMegaHeuristique(["H0"],3)
#print("L",L)
#LP=[i[0] for i in L["FScore"]]
#print("LP",LP)
#VLP=[]

#moy=0
#for j in LP:
#    k=E.Evaluer(set(j))
#    print("j",j)
#    VLP.append(k)
#    moy=moy+k
#moy=moy/len(LP)
#
##for j in range(len(VLP)):
##    VLP[j]=VLP[j]*140#/moy
#
##VLP=sorted(VLP,key=lambda t: t[1],reverse=True)
#print(VLP)
#print([i for i in range(len(VLP))])
##plt.plot([i for i in range(len(LP))],[k[1] for k in L["FScore"]],color="red")
#plt.plot([i for i in range(len(VLP))],VLP,color="green")
#plt.show()
#
##print("L",L["FScore"])
##print("VLP",VLP)

gene="3H0/"
g=Genome.Genome()
L=[]
for i in range(7):
    gene=gene+"3H0/"
    g.identity=gene
    f=fb.Fabriquant(g,D,False)
    L.append(E.Evaluer(f.genome.resultat))
plt.plot(L)
plt.show()
print(L)


