from NaturePackage.Nature import Nature as nat
from Calculs.Destin import Destiny as dest
from sklearn.svm import SVC
class Configuration:

    def __init__(self,nature=None,destiny=None,mesures=None,dataset=None):
        if nature==None:
            self.nature={"Psupp":0.2,"Pstop":0.5,"MaxA":1,"MaxH":5,"MaxP":100,"MaxS":3,"nb_promotions":3,"nb_cluster":5,"Tol":3,"tol_evolutivite":0.25,
                         "scrutin":"Condorcet","metric":"accuracy","evolve_strategies":True,"only_global_crossing":False,
                         "random_initialisation":False,"train_all":False,"strat":[[0.1, 0.7, 0.5, 0.8], [0.5, 0.3, 0.5, 0.7], [0.3, 0.6, 0.5, 0.7]]}
        else:
            self.nature=nature
        if destiny==None:
            self.destiny={"modele":"AdaBoost","seuillage":"unique","max_iterations":5,"treshold":0.5,"coef1":1,"coef2":1,"coef3":1,
                          "mesures":["FScore","DML" ,"GainRatio" ,"FCC"],
                          "all_mesures":["FScore","RST","ReliefF","FCS",'GainInformation' ,"GainRatio" , "SymetricalIncertitude" ,"MutualInformation" , "UH" , "US" , "DML","AdaBoost","RST","FCC","Chisquare"]}
        else:
            self.destiny=destiny
        self.mesures=mesures
        self.dataset=dataset

    def get_nature(self):       return self.nature
    def get_destiny(self):
        return self.destiny
    def get_dataset(self):
        return self.dataset
    def get_mesures(self):
        return self.mesures
    def set_nature(self,dicte):
        self.nature=dicte
    def set_destiny(self,dicte):
        self.destiny=dicte