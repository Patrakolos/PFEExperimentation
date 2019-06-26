import time


from Calculs import Destin as dest
from NaturePackage import Nature as nat
from DataSets import german_dataset
from DataSets import Interface_Datasets
from sklearn.model_selection import train_test_split
from DataSets import Dataset
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import KFold
from DataSets import load_promoters_dataset

from Utilitaire.Evaluateur_Precision import Evaluateur_Precision


def test_nature(nb_it,dataa,targett):
    D = dest.Destiny("AdaBoost", "unique")
    dest.Destiny.stop=False
    D.fit(dataa,targett)
    nat.Nature.train_all=False
    nat.Nature.stop=False
    nat.Nature.paralel_crossing=False
    nat.Nature.only_global_crossing=True
    #for i in range(len(dest.Destiny.mesures_consideres)):
    #    dict[dest.Destiny.mesures_consideres[i]]= dict[dest.Destiny.mesures_consideres[i]]+D.return_criteron(i,10)
    a=time.time()
    nat.Nature.init(D)
    print("Alpha",nat.Nature.actualalpha.identity)
    print("quality",nat.Nature.actual_precision)
    print("Temps: ",time.time()-a)
    print("Union",D.union)
    print("intersection",D.inter)
    for i in range(nb_it):
        nat.Nature.evolve()
        print("Alpha", nat.Nature.actualalpha.identity)
        print("quality", nat.Nature.actual_precision)
        print("Temps: ", time.time() - a)
        print("Condid",len(nat.Nature.condid))

    print("Nature: ",nat.Nature.actual_precision)
    print("Dict : ",dict)

dest.Destiny.mesures_consideres = ["FScore","GainRatio" ,"FCC","DML"]
Interface_Datasets.init()
dat=Interface_Datasets.data_map["German"]
dat.loadDescription()
dat.dataMung(True)
data,target=dat.getDataTarget()
#data,target=load_promoters_dataset.load_promoter_dataset()
#data_train,data_test,target_train,target_test=train_test_split(data,target, test_size=0.33, random_state=2)
kf = KFold(n_splits=10)
moy=0
E=Evaluateur_Precision(data,target)
E.train(AdaBoostClassifier())
dict={}
for i in dest.Destiny.mesures_consideres:
    dict[i]=0
cpt=0
test_nature(30,data,target)

#for train_index, test_index in kf.split(data):
#   data_train, data_test = data[train_index], data[test_index]
#   target_train, target_test = target[train_index], target[test_index]
#   test_nature(10,data_train,target_train)
#   k=E.Evaluer(nat.Nature.actualalpha.resultat)
#   print("K! :",k)
#   cpt=cpt+1
#   moy=moy+k
#   nat.Nature.clear()
#   print("MOY! ", moy)

print("MOY! ",moy)
print("Dict",dict)




