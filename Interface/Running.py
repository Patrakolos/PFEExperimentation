from Calculs import Destin as dest
from DataSets.load_promoters_dataset import load_promoter_dataset
from NaturePackage import Nature as nat
from threading import Thread,RLock


class Running(Thread):

    def __init__(self,modele):
        Thread.__init__(self)
        self.model=modele

    def run(self):
        if(not self.model.interfac.sequential_run.isChecked()):
            self.model.interfac.S.setEnabled(False)
            self.model.interfac.P.setEnabled(False)
            self.model.dataset[self.model.interfac.dataset_combo.currentText()].loadDescription()
            self.model.dataset[self.model.interfac.dataset_combo.currentText()].dataMung(True)
            data, target = self.model.dataset[self.model.interfac.dataset_combo.currentText()].getDataTarget()
            self.model.DM = dest.Destiny(self.model.interfac.training_model.currentText(),self.model.interfac.treshold_type.currentText())
            self.model.refresh()
            self.model.DM.fit(data, target)
            if(self.model.isruning):
                nat.Nature.init(self.model.DM)
                self.model.interfac.iteration.setText("0")
                self.model.interfac.tresh.setText(str(self.model.DM.getTreshold()))
                self.model.interfac.quality.setText(str(nat.Nature.actual_precision))
                self.model.interfac.alpha.setText(str(nat.Nature.actualalpha.identity))
                for i in nat.Nature.population:
                    self.model.interfac.list_identity.addItem(str(i.identity))
                    self.model.interfac.list_incarnation.addItem(str(i.incarnation))
                    self.model.interfac.list_subset.addItem(str(i.resultat))
            for i in range(int(self.model.interfac.max_iteration.text())):
                if(self.model.isruning):
                    nat.Nature.evolve()
                    self.model.interfac.iteration.setText(str(i))
                    self.model.interfac.quality.setText(str(nat.Nature.actual_precision))
                    self.model.interfac.tresh.setText(str(self.model.DM.getTreshold()))
                    self.model.interfac.alpha.setText(str(nat.Nature.actualalpha.identity))

            self.model.interfac.cancel.setEnabled(False)
            self.model.interfac.S.setEnabled(True)
            self.model.interfac.P.setEnabled(True)
        else:
            if(not self.model.intialised):
                self.model.interfac.S.setEnabled(False)
                self.model.interfac.P.setEnabled(False)
                self.model.iteration=0
                self.model.intialised=True
                data, target = load_promoter_dataset()
                self.model.DM = dest.Destiny(self.model.interfac.treshold_type.currentText())
                self.model.refresh()
                self.model.DM.fit(data, target)
                if (self.model.isruning):
                    nat.Nature.init(self.model.DM)
                    self.model.interfac.iteration.setText("0")
                    self.model.interfac.quality.setText(str(nat.Nature.actual_precision))
                    self.model.interfac.tresh.setText(str(self.model.DM.getTreshold()))
                    self.model.interfac.alpha.setText(str(nat.Nature.actualalpha.identity))
                self.model.interfac.cancel.setEnabled(True)
            else:
                nat.Nature.evolve()
                self.model.iteration=self.model.iteration+1
                self.model.interfac.iteration.setText(str(self.model.iteration))
                self.model.interfac.quality.setText(str(nat.Nature.actual_precision))
                self.model.interfac.tresh.setText(str(self.model.DM.getTreshold()))
                self.model.interfac.alpha.setText(str(nat.Nature.actualalpha.identity))
        self.model.active_run(True)
        self.model.isruning=False
        self.model.interfac.statusBar().showMessage('')
        nat.Nature.stop=True
        dest.Destiny.stop=True
