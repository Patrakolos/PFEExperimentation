
import os

from sklearn.svm import SVC

from Calculs.Destin import Destiny
from DataSets.load_promoters_dataset import load_promoter_dataset
from NaturePackage.Nature import Nature
from Utilitaire import Evaluateur_Precision
from NaturePackage import Fabriquant as fab
from NaturePackage import Genome
#data,target = load_promoter_dataset()
##
#D = Destiny()
#D.fit(data,target)
##N = Nature()
##N.init(D)
##for i in range(10):
##    N.evolve()
##    print("-----------------",N.actual_precision)
##
#G=Genome.Genome()
#G.identity="#/"
#f=fab.Fabriquant(G,D,False)
#
#E = Evaluateur_Precision.Evaluateur_Precision(D.getDataset()[0],D.getDataset()[1])
#E.train(SVC())
#print(f.genome.resultat)
#print(E.Evaluer([1, 4, 5, 6, 9, 14, 15, 16, 17, 18, 19, 20, 21, 22, 26, 27, 30, 33, 36, 37, 38, 39, 40, 41, 42, 44, 47, 50, 51]))

import matplotlib.pyplot as plt

x=[1,2,3,4,5,6,7,8,9,10]
y=[1,20,3,14,9,6,7,9,9,8]
plt.bar(x,y,label="bar chart")

plt.xlabel('x')
plt.ylabel('y')
plt.title('Heuristic by threshold')
plt.legend()
plt.show()

