from Calculs import Destin as dest
from Calculs.Destin import Destiny
from DataSets.load_promoters_dataset import load_promoter_dataset
from NaturePackage import Nature as nat
from PyQt5 import QtWidgets,uic,QtGui
import sys
import json
from threading import Thread,RLock
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
import time
from sklearn.ensemble import RandomForestClassifier
from Interface import Running as run
from Interface import Configuration as conf
from Interface import Core
from Interface import Per_heuristic
from Interface import Per_treshold
from Interface import Treshold_type
from DataSets import Interface_Datasets
from Interface import Hyper_heuristic_test
from Interface import interface

class model_ui:

    def check_main_ui(self):
        try:
            float(self.interfac.prob_stop.text())
        except ValueError:
            self.popup("Prob stop doit être un réel")
            return False
        if(float(self.interfac.prob_stop.text())<0.1 or float(self.interfac.prob_stop.text())> 1):
            self.popup("Prob stop doit être entre 0.1 et 1 ")
            return False
        try:
            float(self.interfac.prob_supp.text())
        except ValueError:
            self.popup("Prob supp doit être un réel")
            return False
        if(float(self.interfac.prob_supp.text())<0 or float(self.interfac.prob_supp.text())> 0.9):
            self.popup("Prob supp doit être entre 0 et 0.9 ")
            return False
        try:
            int(self.interfac.max_attribute.text())
        except ValueError:
            self.popup("max attribute doit être un entier")
            return False
        if(int(self.interfac.max_attribute.text())<1 or int(self.interfac.max_attribute.text())> 4):
            self.popup("max attribute doit être entre 1 et 4")
            return False
        try:
            int(self.interfac.max_population.text())
        except ValueError:
            self.popup("max population doit être un entier")
            return False
        if(int(self.interfac.max_population.text())<10 or int(self.interfac.max_population.text())> 999999):
            self.popup("max population doit être entre 10 et 999999 ")
            return False
        try:
            int(self.interfac.nb_promotions.text())
        except ValueError:
            self.popup("nb promotions doit être un entier")
            return False
        if(int(self.interfac.nb_promotions.text())<1 or int(self.interfac.nb_promotions.text())> 10):
            self.popup("nb promotions doit être entre 1 et 10 ")
            return False
        try:
            float(self.interfac.tol_evolutivite.text())
        except ValueError:
            self.popup("tol évolutivité doit être un réel")
            return False
        if(float(self.interfac.tol_evolutivite.text())<0 or float(self.interfac.tol_evolutivite.text())> 0.8):
            self.popup("tol évolutivité doit être entre 0 et 0.8 ")
            return False
        try:
            int(self.interfac.max_iteration.text())
        except ValueError:
            self.popup("max itérations doit être un entier")
            return False
        if(int(self.interfac.max_iteration.text())<0 or int(self.interfac.max_iteration.text())> 999999999):
            self.popup("max itérations doit être entre 0 et 999999999 ")
            return False
        try:
            int(self.interfac.nb_cluster.text())
        except ValueError:
            self.popup("nb cluster doit être un entier")
            return False
        if(int(self.interfac.nb_cluster.text())<0 or int(self.interfac.nb_cluster.text())> 10):
            self.popup("nb cluster doit être entre 0 et 10 ")
            return False
        return True


    def save_file(self):
        dicte={}
        try:
            self.core.bake_destiny(self)
            self.core.bake_nature(self)
            dicte["nature"]=self.core.config.get_nature()
            dicte["destiny"]=self.core.config.get_destiny()
            j=json.dumps(dicte)
        except Exception as e:
            print(e)
        window = QtWidgets.QFileDialog()
        nom_fichier=window.getSaveFileName(None,'Save file','Configs/',filter='JSON (*.json)')
        print(nom_fichier[0])
        if(nom_fichier[0]!=""):
            with open(nom_fichier[0],'w') as f:
               f.write(j)
            try:
                self.popup("Configuration enregistrée dans "+nom_fichier[0])
            except Exception as e:
                print(e)

    def open_file(self):
        try:
            window=QtWidgets.QFileDialog()
            name=window.getOpenFileName(None,'Open file','Configs/',filter='JSON (*.json)')
            print(name[0])
            with open(name[0],'r') as f:
                dicte=json.load(f)
            try:
                self.core.config.set_destiny(dicte["destiny"])
                self.core.config.set_nature(dicte["nature"])
                self.core.drink_destiny(self)
                self.core.drink_nature(self)
                self.gentel_popup("Configuration chargée a partir de "+name[0])
            except Exception as e:
                self.popup("Format du fichier invalide")
        except Exception as e:
            print(e)


    def popup(self,strr):
        self.interfac.statusBar().showMessage('Error: '+ strr)

    def gentel_popup(self,strr):
        self.interfac.statusBar().showMessage(strr)

    def test_dataset(self):
        if(self.check_main_ui()):
            per=Hyper_heuristic_test.hyper_heuristic_test(self,self.core)
            per.start()
    def per_treshold(self):
        if (self.check_main_ui()):
            per=Per_treshold.per_treshold(self,self.core)
            per.start()

    def per_heuristic(self):
        if (self.check_main_ui()):
            per=Per_heuristic.per_heuristic(self,self.core)
            per.start()

    def treshold_type(self):
        per=Treshold_type.Treshold_type(self,self.core)
        per.start()

    def active_run(self,bol):
        self.interfac.per_heuristic.setEnabled(bol)
        self.interfac.per_treshold.setEnabled(bol)
        self.interfac.treshold_type_2.setEnabled(bol)
        self.interfac.run.setEnabled(bol)
        self.interfac.test_dataset.setEnabled(bol)

    def clear(self):
        self.DM.clear()
        nat.Nature.clear()

    def refresh(self,clear=True):
        if(clear):
            self.clear()
        self.core.bake_destiny(self)
        self.core.bake_nature(self)
        self.DM.refresh_dict(self.core.config)
        nat.Nature.refresh_dict(self.core.config)


    def changer_scrutin(self):
        if self.interfac.ballot.currentText()=="Condorcet":
            self.interfac.tol.setEnabled(True)
        else:
            self.interfac.tol.setEnabled(False)


    def add_heuritistic(self):
        heuristic=self.interfac.add_heuristics.currentText()
        i = 0
        lebool=False
        while i < self.interfac.heuristics.count():
            if  self.interfac.heuristics.item(i).text()==heuristic:
                lebool=True
            i=i+1
        if lebool==False:
            self.interfac.heuristics.addItem(heuristic)

    def clear_heuristics(self):
        self.interfac.heuristics.clear()

    def delete_heuristic(self):
            self.interfac.heuristics.takeItem(self.interfac.heuristics.currentRow())

    def affichier_running(self):
        if(self.interfac.Running.isVisible()==False):
            self.interfac.Running.setVisible(True)
        else:
            self.interfac.Running.setVisible(False)
    def gerer_seuil(self):
        if self.interfac.treshold_type.currentText()=="union_intersection":
            self.interfac.union_treshold_label.setEnabled(True)
            self.interfac.nb_tresholding_union.setEnabled(True)
            self.interfac.unique_treshold_label.setEnabled(False)
            self.interfac.manual_treshold_label.setEnabled(False)
            self.interfac.manual_treshold.setEnabled(False)
            self.interfac.nb_tresholding_unique.setEnabled(False)

        if self.interfac.treshold_type.currentText()=="manual":
            self.interfac.union_treshold_label.setEnabled(False)
            self.interfac.nb_tresholding_union.setEnabled(False)
            self.interfac.unique_treshold_label.setEnabled(False)
            self.interfac.nb_tresholding_unique.setEnabled(False)
            self.interfac.manual_treshold_label.setEnabled(True)
            self.interfac.manual_treshold.setEnabled(True)
        if self.interfac.treshold_type.currentText()=="unique":
            self.interfac.union_treshold_label.setEnabled(False)
            self.interfac.nb_tresholding_union.setEnabled(False)
            self.interfac.unique_treshold_label.setEnabled(True)
            self.interfac.nb_tresholding_unique.setEnabled(True)
            self.interfac.manual_treshold_label.setEnabled(False)
            self.interfac.manual_treshold.setEnabled(False)

    def run(self):
        if(self.check_main_ui()):
            self.interfac.statusBar().showMessage('Running...')
            nat.Nature.stop=False
            dest.Destiny.stop=False
            self.isruning=True
            self.running_thread=run.Running(self)
            self.running_thread.start()
            self.active_run(False)
            self.interfac.cancel.setEnabled(True)


    def cancel(self):
        self.isruning=False
        nat.Nature.stop=True
        dest.Destiny.stop=True
        self.intialised=False
        self.iteration=0
        self.interfac.S.setEnabled(True)
        self.interfac.P.setEnabled(True)
        self.clear()
        self.interfac.cancel.setEnabled(False)
        self.time=0




    def init_interface1(self):
        self.interfac.treshold_type.addItem("manual")
        self.interfac.treshold_type.addItem("union_intersection")
        self.interfac.treshold_type.addItem("complexity")
        self.interfac.treshold_type.addItem("unique")
        self.interfac.metrics.addItem("accuracy")
        self.interfac.ballot.addItem("None")
        self.interfac.ballot.addItem("Condorcet")
        self.interfac.training_model.addItems(["SVM","MLP","DTC","RF","Adaboost","KNN"])
        self.interfac.max_iteration.setText(str(0))
        self.interfac.strategies.setColumnCount(4)
        self.interfac.strategies.setShowGrid(True)
        self.interfac.nb_tresholding_unique.setText("10")
        self.interfac.nb_tresholding_union.setText("10")
        self.interfac.max_iteration.setText("5")
        self.interfac.iteration_type.setText("5")
        self.core.drink_destiny(self)
        self.interfac.per_heuristic_iteration.setText("5")
        self.interfac.nb_sbudivisions.setText("10")
        self.core.drink_nature(self)
        self.interfac.heuristic_test.addItems(self.core.config.get_destiny()["all_mesures"])



    def init_interface2(self):
        self.interfac.heuristics_button.clicked.connect(self.add_heuritistic)
        self.interfac.clear.clicked.connect(self.clear_heuristics)
        self.interfac.per_heuristic.clicked.connect(self.per_heuristic)
        self.interfac.ballot.currentIndexChanged.connect(self.changer_scrutin)
        self.interfac.treshold_type.currentIndexChanged.connect(self.gerer_seuil)
        self.interfac.delete_h.clicked.connect(self.delete_heuristic)
        self.interfac.run.clicked.connect(self.run)
        self.interfac.cancel.clicked.connect(self.cancel)
        self.interfac.sequential_run.stateChanged.connect(self.cancel)
        self.interfac.per_treshold.clicked.connect(self.per_treshold)
        self.interfac.treshold_type_2.clicked.connect(self.treshold_type)
        self.interfac.actionSave.triggered.connect(self.save_file)
        self.interfac.actionOpen.triggered.connect(self.open_file)
        self.interfac.test_dataset.clicked.connect(self.test_dataset)
        self.interfac.dataset_combo.addItems(self.dataset.keys())
        self.interfac.add_heuristics.addItems(self.core.config.get_destiny()["all_mesures"])
        self.refresh_dataset()




    def refresh_dataset(self):
        for i in self.dataset.keys():
            self.interfac.list_dataset.addItem(i)

    def __init__(self):
        app = QtWidgets.QApplication(sys.argv)
        self.running_thread=run.Running(self)
        self.core=Core.Core(self)
        self.isruning=False
        self.config=conf.Configuration()
        self.interfac = uic.loadUi("interface.ui")
        Interface_Datasets.init()
        self.dataset = Interface_Datasets.data_map
        self.init_interface1()
        self.DM = dest.Destiny("SVM","manual")
        self.init_interface2()
        self.intialised=False
        self.iteration=0
        self.interfac.show()

        app.exec_()

dlg=model_ui()



#back

# nat.Nature.evolve_strategies = self.interfac.evolve_strategies.isChecked()
# nat.Nature.only_global_crossing = self.interfac.only_global_crossing.isChecked()
# nat.Nature.random_initialisation = self.interfac.random_initalisation.isChecked()
# nat.Nature.train_all = self.interfac.train_all.isChecked()
# dest.Destiny.maxH=self.interfac.heuristics.count()-1
# nat.Nature.Pstop=float(self.interfac.prob_stop.text())
# nat.Nature.Psupp=float(self.interfac.prob_supp.text())
# nat.Nature.maxA=int(self.interfac.max_attribute.text())
# nat.Nature.maxP=int(self.interfac.max_population.text())
# nat.Nature.nb_promo=int(self.interfac.nb_promotions.text())
# nat.Nature.tol_evolutivite=float(self.interfac.tol_evolutivite.text())
# nat.Nature.Tol=int(self.interfac.tol.text())
# nat.Nature.scrutin=self.interfac.ballot.currentText()
# self.DM.setSeuillage(self.interfac.treshold_type.currentText())
# if self.interfac.treshold_type.currentText()=="union_intersection":
#    self.DM.setMax_iterations(int(self.interfac.nb_tresholding_union.text()))
# if self.interfac.treshold_type.currentText()=="unique":
#    self.DM.setMax_iterations(int(self.interfac.nb_tresholding_unique.text()))
# if self.interfac.treshold_type.currentText() == "manual":
#    self.DM.setTreshold(float(self.interfac.manual_treshold.text()))
#    self.DM.__Treshold=float(self.interfac.manual_treshold.text())
#    self.DM.seuillage="manual"
#    print("DM.Seuillage",self.DM.seuillage)
#    print("DM.Treshold",self.DM.__Treshold)
# if self.interfac.treshold_type.currentText()=="complexity":
#    dest.Destiny.coef3=float(self.interfac.f3.text())
#    dest.Destiny.coef2=float(self.interfac.f2.text())
#    dest.Destiny.coef1=float(self.interfac.f1.text())

# nat.Nature.metric=self.interfac.metrics.currentText()
# model=self.interfac.training_model.currentText()
# if model == "SVM":
#    nat.Nature.modele=SVC()
# if model == "DTC":
#    nat.Nature.modele=DecisionTreeClassifier()
# if model == "MLP":
#    nat.Nature.modele=MLPClassifier()
# if model == "Adaboost":
#    nat.Nature.modele=AdaBoostClassifier()
# if model == "KNN":
#    nat.Nature.modele=KNeighborsClassifier()
# if model == "RF":
#    nat.Nature.modele=RandomForestClassifier()

# while i < self.interfac.heuristics.count():
#    if self.DM.Mmesures_classification.__contains__(self.interfac.heuristics.item(i).text()):
#        self.DM.mesures_classification.append(self.interfac.heuristics.item(i).text())
#    if self.DM.Mmesures_consistance.__contains__(self.interfac.heuristics.item(i).text()):
#        self.DM.mesures_consistance.append(self.interfac.heuristics.item(i).text())
#    if self.DM.Mmesures_dependance.__contains__(self.interfac.heuristics.item(i).text()):
#        self.DM.mesures_dependance.append(self.interfac.heuristics.item(i).text())
#    if self.DM.Mmesures_distance.__contains__(self.interfac.heuristics.item(i).text()):
#        self.DM.mesures_distance.append(self.interfac.heuristics.item(i).text())
#    if self.DM.Mmesures_information.__contains__(self.interfac.heuristics.item(i).text()):
#        self.DM.mesures_information.append(self.interfac.heuristics.item(i).text())
#    i=i+1



# init
#self.interfac.prob_stop.setText(str(nat.Nature.Pstop))
#        self.interfac.prob_supp.setText(str(nat.Nature.Psupp))
#        self.interfac.max_attribute.setText(str(nat.Nature.maxA))
#        self.interfac.max_population.setText(str(nat.Nature.maxP))
#        self.interfac.nb_promotions.setText(str(nat.Nature.nb_promo))
#        self.interfac.tol_evolutivite.setText(str(nat.Nature.tol_evolutivite))
#        self.interfac.tol.setText(str(nat.Nature.Tol))
#        self.interfac.nb_cluster.setText(str(nat.Nature.nb_cluster))#

#self.interfac.manual_treshold.setText(str(0.5))
#self.interfac.f1.setText(str(dest.Destiny.coef1))
#self.interfac.f2.setText(str(dest.Destiny.coef2))
#self.interfac.f3.setText(str(dest.Destiny.coef3))
# self.interfac.add_heuristics.addItems(["FScore","RST","ReliefF","FCS",'GainInformation' ,"GainRatio" , "SymetricalIncertitude" ,"MutualInformation" , "UH" , "US" , "DML","AdaBoost","RST"])


#init 2
#
#self.interfac.heuristics.addItems(self.DM.mesures_consistance)
#        self.interfac.heuristics.addItems(self.DM.mesures_classification)
#        self.interfac.heuristics.addItems(self.DM.mesures_dependance)
#        self.interfac.heuristics.addItems(self.DM.mesures_distance)
#        self.interfac.heuristics.addItems(self.DM.mesures_information)#