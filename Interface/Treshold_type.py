from PyQt5.QtGui import QPixmap

from Calculs import Destin as dest
from DataSets.load_promoters_dataset import load_promoter_dataset
from NaturePackage import Nature as nat
from threading import Thread,RLock
import matplotlib.pyplot as plt
from NaturePackage import Nature as nat
from Calculs import Destin as dest

class Treshold_type(Thread):

    def __init__(self,modele,core):
        Thread.__init__(self)
        self.model=modele
        self.core=core

    def run(self):
        self.model.interfac.statusBar().showMessage('Running...')
        self.model.active_run(False)
        types=["manual","union_intersection","complexity","unique"]
        dictt={}
        for i in types:
            dictt[i]=self.intance(i)
            print("__I: ",i)
        print(dictt)
        x = dictt.keys()
        y = dictt.values()
        plt.xlabel("Treshold type")
        plt.ylabel("accuracy")
        plt.bar(x, y, label="bar chart", width=0.05)

        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('acuracy by treshold type')
        plt.legend()
        plt.savefig('foo.png')
        self.model.interfac.charte.setPixmap(QPixmap('foo.png'))
        self.model.interfac.charte.setScaledContents(True)
        plt.show()
        self.model.active_run(True)

    def intance(self,strr):
        nat.Nature.stop=False
        dest.Destiny.stop=False
        self.model.isruning = True
        self.model.interfac.S.setEnabled(False)
        self.model.interfac.P.setEnabled(False)
        data, target = load_promoter_dataset()
        self.model.DM = dest.Destiny(strr)
        self.model.refresh()
        self.model.DM.fit(data, target)
        if (self.model.isruning):
            nat.Nature.init(self.model.DM)
        for i in range(int(self.model.interfac.iteration_type.text())):
            if (self.model.isruning):
                nat.Nature.evolve()
        self.model.interfac.cancel.setEnabled(False)
        self.model.interfac.S.setEnabled(True)
        self.model.interfac.P.setEnabled(True)
        self.model.isruning=False
        return nat.Nature.actual_precision

