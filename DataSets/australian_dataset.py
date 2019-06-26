import numpy as np
import requests
from sklearn.naive_bayes import GaussianNB



def load_australian_dataset():
    print('h')
    r = requests.get(r"http://archive.ics.uci.edu/ml/machine-learning-databases/statlog/australian/australian.dat")
    L = str(r.content).replace("b'","").replace("\\r","").replace("'","").split(r'\n')
    X = []
    for i in L:
        try:
            K = np.array(i.split(' ')).astype("float")
            X.append (K)
        except:
            continue
    D = np.array(X[:-1])
    D = D.transpose()
    Y = D[-1]
    X = np.array(X)[:-2]
    Y = Y[:-1]
    return X,Y

def save_dataset_on_disc():
    try:
        with open("australian_dataset.txt","w") as f:
            r = requests.get (r"http://archive.ics.uci.edu/ml/machine-learning-databases/statlog/australian/australian.dat")
            f.write(str(r.content).replace("\n","\n"))
            f.close()
    except(Exception):
        print("retry")
        save_dataset_on_disc()



#train,target = load_australian_dataset()
#print(train.shape)
#print(target.shape)
#print(target)
#E = Evaluateur_Precision(train,target)
#E.train(GaussianNB())
#print(E.vecteur_precision())