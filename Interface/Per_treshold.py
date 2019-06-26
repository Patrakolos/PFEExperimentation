from PyQt5.QtGui import QPixmap

from Calculs import Destin as dest
from DataSets.load_promoters_dataset import load_promoter_dataset
from NaturePackage import Nature as nat
from threading import Thread,RLock
import matplotlib.pyplot as plt


class per_treshold(Thread):

    def __init__(self,modele,core):
        Thread.__init__(self)
        self.model=modele
        self.core=core

    def run(self):
        self.model.interfac.statusBar().showMessage('Running...')
        self.model.active_run(False)
        pas=1.0/int(self.model.interfac.nb_sbudivisions.text())
        dictt={}
        i=pas
        h=self.core.config.get_destiny()["all_mesures"].index(self.model.interfac.heuristic_test.currentText())
        D = dest.Destiny(self.model.interfac.training_model.currentText(),"manual")
        dest.Destiny.stop = False
        data, target = self.model.dataset[self.model.interfac.dataset_combo.currentText()].getDataTarget()
        D.fit(data, target)
        D.mesures_consideres = self.core.config.get_destiny()["all_mesures"]
        #self.model.interfac.rapport.clear()
        while(i<1):
            dictt[i]=D.criteron_heursitique_unique(h,i)
            print("h")
        #    self.model.interfac.rapport.addItem(str(i)+": "+str(dictt[i]))
            i=i+pas
        print("i")
        self.model.active_run(True)
        print("j")
        dest.Destiny.stop=True
        x=dictt.keys()
        y=dictt.values()
        plt.xlabel("Treshold")
        plt.ylabel("accuracy")
        plt.bar(x, y, label="bar chart",width=0.05)

        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Heuristic by threshold')
        plt.legend()
        plt.savefig('foo.png')
        self.model.interfac.charte.setPixmap(QPixmap('foo.png'))
        self.model.interfac.charte.setScaledContents(True)
        plt.show()





