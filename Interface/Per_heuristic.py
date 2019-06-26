from PyQt5.QtGui import QPixmap

from Calculs import Destin as dest
from DataSets.load_promoters_dataset import load_promoter_dataset
from NaturePackage import Nature as nat
from threading import Thread,RLock
import matplotlib.pyplot as plt

class per_heuristic(Thread):

    def __init__(self,modele,core):
        Thread.__init__(self)
        self.model=modele
        self.core=core

    def run(self):
        self.model.interfac.statusBar().showMessage('Running...')
        self.model.active_run(False)
        dict = {}
        dest.Destiny.stop = False
        dest.Destiny.mesures_consideres = self.core.config.get_destiny()["all_mesures"]
        D = dest.Destiny(self.model.interfac.training_model.currentText(),"manual")
        data, target = self.model.dataset[self.model.interfac.dataset_combo.currentText()].getDataTarget()
        D.fit(data, target)

        for i in range(len(self.core.config.get_destiny()["all_mesures"])):
            b = int(self.model.interfac.per_heuristic_iteration.text())
            a = D.return_criteron(i, b)
            dict[self.core.config.get_destiny()["all_mesures"][i]] = a
        stre = ""
        #self.model.interfac.rapport.clear()
        #for k in dict.keys():
        #    stre = str(k) + ": " + str(dict[k]) + "\n"
        #    self.model.interfac.rapport.addItem(stre)
        print(stre)
        dest.Destiny.stop = True
        self.model.active_run(True)
        x = dict.keys()
        xx=[x for x in range(len(dict.keys()))]
        y = dict.values()
        print("___dict",dict)
        plt.xticks(xx,x)
        plt.bar(x, y, label="bar chart",width=0.27)

        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Heuristic by threshold')
        plt.legend()
        plt.savefig('foo.png')
        self.model.interfac.charte.setPixmap(QPixmap('foo.png'))
        self.model.interfac.charte.setScaledContents(True)
        plt.show()

