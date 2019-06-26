from itertools import combinations

from sklearn import linear_model , svm
from sklearn.ensemble import AdaBoostClassifier , RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error , r2_score , mean_absolute_error
from sklearn.model_selection import train_test_split , KFold
from sklearn.svm import SVR , SVC
from sklearn.tree import DecisionTreeClassifier

from Calculs.Destin import Destiny
from DataSets.german_dataset import load_german_dataset
from Utilitaire.Evaluateur_Precision import Evaluateur_Precision
import pandas as pd
import numpy as np
from sklearn.datasets import load_diabetes


def GenererFichDonnees(long_comb):
    data , target = load_german_dataset ()
    print (len (data[0]))
    c = combinations (list (range (0 , len (data[0]))) , long_comb)
    w = []
    for i in c:
        w.append (list (i))

    c = combinations (list (range (0 , len (data[0]))) , long_comb-1)
    for i in c:
        w.append (list (i))

    c = w
    total = len (list (c))
    D = []
    T = []
    Tot = []
    k = 0
    print (list (c))
    Dest = Destiny ()
    Dest.fit (data , target)
    E = Evaluateur_Precision (data , target)
    E.train (AdaBoostClassifier())
    for i in c:
        k = k + 1
        pr = list (Dest.Projection (i))
        eval = E.Evaluer (i)
        D.append (pr)
        T.append (eval)
        KW = list (pr) + [eval]
        Tot.append (KW)
        print (k , " / " , total , " termines ")

    dat = np.array (D)
    tar = np.array (T)

    print (dat.shape)
    print (tar.shape)
    print (Tot)
    Df = pd.DataFrame (Tot)
    Df.to_csv ("TestProjectionPrediction" + str (long_comb))
    print (Df)

def GenererGrosFichDonnees(long_comb):
    data , target = load_german_dataset ()
    print (len (data[0]))
    w = []
    for i in range(2,long_comb+1):
        c = combinations (list (range (0 , len (data[0]))) , long_comb)
        for i in c:
            w.append (list (i))
    c = w
    total = len (list (c))
    D = []
    T = []
    Tot = []
    k = 0
    print (list (c))
    Dest = Destiny ()
    Dest.fit (data , target)
    E = Evaluateur_Precision (data , target)
    E.train (SVC (gamma="auto"))
    for i in c:
        k = k + 1
        pr = list (Dest.Projection (i))
        eval = E.Evaluer (i)
        D.append (pr)
        T.append (eval)
        KW = list (pr) + [eval]
        Tot.append (KW)
        print (k , " / " , total , " termines ")

    dat = np.array (D)
    tar = np.array (T)

    print (dat.shape)
    print (tar.shape)
    print (Tot)
    Df = pd.DataFrame (Tot)
    Df.to_csv ("TestProjectionPredictionXX" + str (long_comb))
    print (Df)


def TrainModele(data,target):
    regr = linear_model.LinearRegression ()

    data_train , data_test , target_train , target_test \
        = train_test_split (data , target , test_size=0.4 , random_state=0)

    regr.fit (data_train , target_train)
    print("data test",data_test)
    target_pred = regr.predict (data_test)
    #print ("Predictions : ")
    #print (target_pred)
    #print ("Réality : ")
    #print (target_test)

    #print ('Coefficients: \n' , regr.coef_)
    print ("Mean squared error: %.2f" , mean_squared_error (target_test , target_pred))
    # Explained variance score: 1 is perfect prediction
    print ('Variance score: %.2f' % r2_score (target_test , target_pred))
    return mean_squared_error (target_test , target_pred) , r2_score (target_test , target_pred)

def Test():
    import pandas as pd
    df = pd.read_csv ("TestProjectionPrediction4")
    mat = np.array (df)
    mat = mat.T
    mat = mat[1:]
    data = mat[:-1]
    target = mat[-1]
    mat = mat.T
    data = data.T
    print (data.shape)
    print (target.shape)
    pas = 50
    t = int(data.shape[0] / pas)
    rez = []
    for i in range(1,pas+1):
        Kf = KFold(n_splits=25,shuffle=True)
        data_sc = data[0:i*t]
        target_sc = target[0:i*t]
        regr = svm.SVR (gamma = "auto")
        for regr in [(svm.SVR(gamma="auto"),"SVR gamma = auto"),(LinearRegression(),"Linear Regression"),(svm.SVR(gamma="scale"),"SVR gamma = scale")]:
            for t_index,test_index in Kf.split(data_sc,target_sc):

                data_train , target_train = data_sc[t_index] , target_sc[t_index]
                data_test , target_test = data_sc[test_index] , target_sc[test_index]

                regr[0].fit(data_train,target_train)

                target_pred = regr[0].predict(data_test)

                mse = (mean_squared_error (target_test , target_pred),"MSE")
                r2 = (r2_score (target_test , target_pred),"R2")
                mae = (mean_absolute_error(target_test,target_pred),"MAE")
                for j in [mae]:
                    T = [data_train.shape[0] , j[0], j[1], regr[1]]
                    rez.append (T)

    import seaborn as sns
    sns.set ()
    import matplotlib.pyplot as plt
    ax = sns.lineplot (x=0 , y=1 ,hue=3 ,data=pd.DataFrame(rez,dtype="float64"))
    plt.show ()

Test()