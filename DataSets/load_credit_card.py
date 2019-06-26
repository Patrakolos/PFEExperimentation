import pandas as pd
import numpy as np


def load():
    Pand = pd.read_csv (r"C:\Users\Geekzone\IdeaProjects\untitled10\Fichiers\creditcard.csv")
    dataset = np.array (Pand)[:10000]
    clas = dataset.transpose ()[-2]
    X = dataset[:-2]
    Y = clas.transpose ()[:-2]
    return X,Y

