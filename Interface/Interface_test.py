from PyQt5 import QtWidgets, uic
import sys
from Destiny.DataSets import german_dataset



def passer():
    interface.listWidget.addItem(interface.lineEdit.text())
app=QtWidgets.QApplication(sys.argv)
interface=uic.loadUi("interface.ui")
interface.show()
interface.pushButton.clicked.connect(passer)
interface.lineEdit.returnPressed.connect(passer)
app.exec_()


