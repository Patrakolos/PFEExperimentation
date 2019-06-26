from sklearn.naive_bayes import GaussianNB
from DataSets import german_dataset, Interface_Datasets
from Utilitaire.Evaluateur_Precision import Evaluateur_Precision
from sklearn.ensemble import AdaBoostClassifier
from sklearn import naive_bayes

Interface_Datasets.init()
dat=Interface_Datasets.data_map["German"]
dat.loadDescription()
dat.dataMung(True)
data,target=dat.getDataTarget()
E = Evaluateur_Precision(data, target)
E.train(naive_bayes.GaussianNB())
k=E.Evaluer((0, 1, 2, 3, 4, 5, 7, 8, 9, 11, 12,13,14,15,16,17,18,19))
print(k)