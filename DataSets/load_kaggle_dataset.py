import pandas as pd


class load_kaggle_dataset:
    @staticmethod
    def load_dateset():
        train = pd.read_csv (r'train.csv')
        target = train['target']
        features = [c for c in train.columns if c not in ['ID_code' , 'target']]
        train = train[features]
        return train,target