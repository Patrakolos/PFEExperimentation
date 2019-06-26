import os

from DataSets.Dataset import Dataset

data_map  = {}
noms_dataset = ["Australian", "German", "Heart" ]


def download():
    D = Dataset("Australian")
    D.download("http://archive.ics.uci.edu/ml/machine-learning-databases/statlog/australian/australian.dat")
    D = Dataset("German")
    D.download("https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data")
    D = Dataset("Heart")
    D.download("http://archive.ics.uci.edu/ml/machine-learning-databases/statlog/heart/heart.dat")
    for i in noms_dataset:
        D = Dataset(i)
        data_map[i] = D
    descriptionAttributs()
    for i in noms_dataset:
        D = Dataset(i)
        data_map[i] = D
        D.storeDescription ()





def descriptionAttributs():
    data_map["German"].describeAttributs(
        {"Categorique" : [0,1,4,6],
        "Categorique_Ordre" : [1,2,4],
        "Explication_Attributs" : {2 : "Test pour voire si je retrouve dans ma base de connaissances l'explication de 2"}
         }
    )
    data_map["Australian"].describeAttributs({
        "Categorique":[0, 1, 4, 6],
        "Categorique_Ordre":[1, 4],
        "Nom":{3: "Salaire"}}
    )
    data_map["Heart"].describeAttributs({
        "Categorique":[0, 1, 4, 6],
        "Categorique_Ordre":[1, 4]
    }
    )


def printDescriptions():
    for i in data_map:
        print(data_map[i].getDesc())


def loadDatasets(mung = False,describe = False):
    for i in os.listdir("..\\Local_Datasets"):
        ref = i.replace("_", "").replace(".csv", "")
        data_map[ref] = Dataset(ref)
        if(mung):
            data_map[ref].dataMung()
        if(describe):
            data_map[ref].loadDescription()


def init():
    for i in data_map:
        data_map[i].dataMung()
        data_map[i].loadDescription()


download()