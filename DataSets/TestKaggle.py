import pandas as pd
import numpy as np
from sklearn.preprocessing import KBinsDiscretizer



class KaggleDataSet:
    @staticmethod
    def getKaggleDataset():
        train = pd.read_csv ('train.csv')
        print ("dumped")
        target = train['target']
        features = [c for c in train.columns if c not in ['ID_code' , 'target']]
        train = train[features]
        train = np.array (train)
        target = np.array (target)
        train = train[0:100]
        target = target[0:100]
        return train,target




