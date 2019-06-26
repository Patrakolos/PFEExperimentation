from PyQt5.QtGui import QPixmap

from Calculs import Destin as dest
from DataSets.load_promoters_dataset import load_promoter_dataset
from NaturePackage import Nature as nat
from threading import Thread,RLock
import matplotlib.pyplot as plt
from Calculs import Destin
import operator

class hyper_heuristic_test(Thread):

    def __init__(self,modele,core):
        Thread.__init__(self)
        self.model=modele
        self.core=core

    def run(self):
        self.model.interfac.statusBar().showMessage('Running...')
        self.model.active_run(False)

        dict = {}

        for l in self.model.dataset.keys():
            self.model.DM=Destin.Destiny("manual")
            self.model.refresh(True)
            D = self.model.DM
            dict[l]={}
            self.model.dataset[l].dataMung(True)
            print("______DATAST",l)
            print("______maping",D.mapping_mesures)
            print("____anterieures",D.get_mesures_anterieures())
            self.model.dataset[l].loadDescription()
            data, target = self.model.dataset[l].transform(self.model.dataset[l].getDataTarget()[0],self.model.dataset[l].getDataTarget()[1])
            dest.Destiny.stop = False
            D.fit(data, target)
            for i in range(len(self.model.DM.mesures_consideres)):
                b = int(self.model.interfac.iterations_mesures_visual.text())
                a = D.return_criteron(i, b)
                dict[l][self.model.DM.mesures_consideres[i]] = a
            if(self.model.interfac.include_nature.isChecked()):
                self.model.DM = dest.Destiny(self.model.interfac.training_model.currentText(),self.model.interfac.treshold_type.currentText())
                self.model.refresh()
                self.model.DM.fit(data, target)
                nat.Nature.stop = False
                nat.Nature.init(self.model.DM)
                for it in range(int(self.model.interfac.iterations_nature_visual.text())):
                    print("_____iteration :",it)
                    nat.Nature.evolve()
                dict[l]["Nature"]=nat.Nature.actual_precision
                nat.Nature.stop = True
                dest.Destiny.stop = True


        clasement=[]
        for lp in self.model.DM.mesures_consideres:
            a=0
            for k in dict.keys():
                a=a+int(dict[k][lp])
            clasement.append((lp,a))
        les_mesures=[a[0] for a in (sorted(clasement, key=operator.itemgetter(1), reverse=True)[0:3])]

        x=dict.keys()
        mes=[]
        m=0
        while m<3:
            for k in dict.keys():
                mes.append(dict[k][les_mesures[m]])
            m=m+1

        plt.bar(x,mes[0], label=les_mesures[0],width=0.05,color='red')
        plt.bar(x, mes[1], label=les_mesures[1], width=0.05, color='green')
        plt.bar(x, mes[2], label=les_mesures[2], width=0.05, color='blue')
        if (self.model.interfac.include_nature.isChecked()):
            nature = {}
            for k in dict.keys():
                nature[k] = dict[k]["Nature"]
            plt.bar(x, nature.values(), label="Nature", width=0.05, color='yelow')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Testing Datasets')
        plt.legend()
        plt.savefig('foo.png')
        self.model.interfac.charte_2.setPixmap(QPixmap('foo.png'))
        self.model.interfac.charte_2.setScaledContents(True)
        plt.show()







        print("dict: ",dict)


        self.model.active_run(True)

