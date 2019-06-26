from nltk import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from Interface import Configuration as conf



class Core:
    ho=0
    def __init__(self,ui,datasets=None,configs=None):
        self.config=conf.Configuration()
        self.datasets=None
        self.configs=[self.config]
        self.actual_ui=ui


    def set_ui(self,ui):
        self.actual_ui=ui
        
    def bake_nature(self,ui):
        self.config.get_nature()["MaxH"] = ui.interfac.heuristics.count()
        self.config.get_nature()["Pstop"] = float(ui.interfac.prob_stop.text())
        self.config.get_nature()["Psupp"] = float(ui.interfac.prob_supp.text())
        self.config.get_nature()["MaxA"] = int(ui.interfac.max_attribute.text())
        self.config.get_nature()["MaxP"] = int(ui.interfac.max_population.text())
        self.config.get_nature()["nb_promo"] = int(ui.interfac.nb_promotions.text())
        self.config.get_nature()["tol_evolutivité"] = float(ui.interfac.tol_evolutivite.text())
        self.config.get_nature()["Tol"] = int(ui.interfac.tol.text())
        self.config.get_nature()["scrutin"] = ui.interfac.ballot.currentText()
        self.config.get_nature()["metric"] = ui.interfac.metrics.currentText()
        #model = ui.interfac.training_model.currentText()
        #self.config.get_nature()["modele"]=model
        self.config.get_nature()["evolve_strategies"] = ui.interfac.evolve_strategies.isChecked()
        self.config.get_nature()["only_global_crossing"] = ui.interfac.only_global_crossing.isChecked()
        self.config.get_nature()["random_initialisation"] = ui.interfac.random_initalisation.isChecked()
        self.config.get_nature()["train_all"] = ui.interfac.train_all.isChecked()
    
    def bake_destiny(self,ui):
        model = ui.interfac.training_model.currentText()
        self.config.get_destiny()["modele"]=model
        i = 0
        self.config.get_destiny()["mesures"].clear()
        while i < ui.interfac.heuristics.count():
            self.config.get_destiny()["mesures"].append(ui.interfac.heuristics.item(i).text())
            i = i + 1

        self.config.get_destiny()["seuillage"] = ui.interfac.treshold_type.currentText()
        self.config.get_destiny()["coef1"] = float(ui.interfac.f1.text())
        self.config.get_destiny()["coef2"] = float(ui.interfac.f2.text())
        self.config.get_destiny()["coef3"] = float(ui.interfac.f3.text())

        if ui.interfac.treshold_type.currentText() == "union_intersection":
            self.config.get_destiny()["max_iteration"] = int(ui.interfac.nb_tresholding_union.text())
        if ui.interfac.treshold_type.currentText() == "unique":
            self.config.get_destiny()["max_iteration"] = int(ui.interfac.nb_tresholding_unique.text())
        if ui.interfac.treshold_type.currentText() == "manual":
            self.config.get_destiny()["treshold"] = float(ui.interfac.manual_treshold.text())

    def drink_nature(self, ui):
        ui.interfac.prob_stop.setText(str(self.config.get_nature()["Pstop"]))
        ui.interfac.prob_supp.setText(str(self.config.get_nature()["Psupp"]))
        ui.interfac.max_attribute.setText(str(self.config.get_nature()["MaxA"]))
        ui.interfac.max_population.setText(str(self.config.get_nature()["MaxP"]))
        ui.interfac.nb_promotions.setText(str(self.config.get_nature()["nb_promotions"]))
        ui.interfac.tol_evolutivite.setText(str(self.config.get_nature()["tol_evolutivite"]))
        ui.interfac.tol.setText(str(self.config.get_nature()["Tol"]))
        ui.interfac.nb_cluster.setText(str(self.config.get_nature()["nb_cluster"]))
#        ui.interfac.ballot.setInsertPolicy(str(self.config.get_nature()["scrutin"]))
#        ui.interfac.setInsertPolicy.setInsertPolicy(str(self.config.get_nature()["modele"]))
        ui.interfac.evolve_strategies.setChecked(self.config.get_nature()["evolve_strategies"])
        ui.interfac.only_global_crossing.setChecked(self.config.get_nature()["only_global_crossing"])
        ui.interfac.random_initalisation.setChecked(self.config.get_nature()["random_initialisation"])
        ui.interfac.train_all.setChecked(self.config.get_nature()["train_all"])
#        ui.interfac.metrics.setInsertPolicy(str(self.config.get_nature()["metric"]))
    
    def drink_destiny(self,ui):
        ui.interfac.manual_treshold.setText(str(self.config.get_destiny()["treshold"]))
        ui.interfac.f1.setText(str(self.config.get_destiny()["coef1"]))
        ui.interfac.f2.setText(str(self.config.get_destiny()["coef2"]))
        ui.interfac.f3.setText(str(self.config.get_destiny()["coef3"]))
        ui.interfac.add_heuristics.addItems(self.config.get_destiny()["all_mesures"])
        ui.interfac.heuristics.addItems(self.config.get_destiny()["mesures"])

        
