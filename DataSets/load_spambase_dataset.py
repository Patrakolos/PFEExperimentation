import requests
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.ensemble import RandomForestClassifier , AdaBoostClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from Destiny import Evaluateur_Precision , Embedded_Thresholding
from Destiny.Evaluateur_Precision import Evaluateur_Precision


def load_spambase_dataset():
    r = requests.get(r"https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data")
    L = str(r.content).replace("b'","").replace("\\r","").split(r'\n')
    X = []
    for i in L[:-1]:
        K = np.array(i.split(',')).astype("float")
        X.append(K)
    D = np.array(X[:-1])
    D = D.transpose()
    Y = D[-1]
    X = np.array(X)[:-2]
    return X,Y

train,target = load_spambase_dataset()
print(train.shape)
print(target.shape)
target = target[:-1]
print(target)
E = Evaluateur_Precision(train,target)
E.train(SVC())
print(E.vecteur_precision())