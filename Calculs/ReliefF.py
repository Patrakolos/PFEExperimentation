# -*- coding: utf-8 -*-

"""
Copyright (c) 2016 Randal S. Olson

Permission is hereby granted, free of charge, to any person obtaining a copy of this software
and associated documentation files (the "Software"), to deal in the Software without restriction,
including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial
portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT
LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

from __future__ import print_function
import numpy as np
from sklearn.neighbors import KDTree

from Calculs.FCS import FCS
from DataSets.german_dataset import load_german_dataset
from Utilitaire.Dimension_Reductor import Dimension_Reductor
from Utilitaire.Mesure import Mesure


class ReliefF(Mesure):

    """Feature selection using data-mined expert knowledge.

    Based on the RankingFunctions algorithm as introduced in:

    Kononenko, Igor et al. Overcoming the myopia of inductive learning
    algorithms with RELIEFF (1997), Applied Intelligence, 7(1), p39-55

    """

    def __init__(self, n_neighbors=100, n_features_to_keep=10):
        """Sets up RankingFunctions to perform feature selection.

        Parameters
        ----------
        n_neighbors: int (default: 100)
            The number of neighbors to consider when assigning feature
            importance scores.
            More neighbors results in more accurate scores, but takes longer.

        Returns
        -------
        None

        """
        super().__init__()
        self._liste_mesures = ["ReliefF"]
        self.feature_scores = None
        self.top_features = None
        self.tree = None
        self.n_neighbors = n_neighbors
        self.n_features_to_keep = n_features_to_keep
        self.scores = {}
        self.__data = None
        self.__target = None

    def ranking_function_constructor(self,motclef):
        if(motclef == "ReliefF"):
            return self.score

    def fit(self, data, target):
        super().fit(data,target)
        self.__data = data
        self.__target = target
        """Computes the feature importance scores from the training data.

        Parameters
        ----------
        X: array-like {n_samples, n_features}
            Training instances to compute the feature importance scores from
        y: array-like {n_samples}
            Training labels
        }

        Returns
        -------
        None

        """
        self.feature_scores = np.zeros(data.shape[1])
        self.tree = KDTree(data)
        for source_index in range(data.shape[0]):
            distances, indices = self.tree.query(
                data[source_index].reshape(1, -1), k=self.n_neighbors+1)
            # Nearest neighbor is self, so ignore first match
            indices = indices[0][1:]
            # Create a binary array that is 1 when the source and neighbor
            #  match and -1 everywhere else, for labels and features..
            labels_match = np.equal(target[source_index], target[indices]) * 2. - 1.
            features_match = np.equal(data[source_index], data[indices]) * 2. - 1.
            # The change in feature_scores is the dot product of these  arrays
            self.feature_scores += np.dot(features_match.T, labels_match)
        self.top_features = np.argsort(self.feature_scores)[::-1]

    def score(self,x):
        if(len(x)>1):
            DR = Dimension_Reductor()
            DR.fit(self.__data,self.__target)
            L = DR.getPCA(x)
            LL = []
            LL.append(L)
            LL = np.array(LL)
            LL = LL.transpose()
            R = ReliefF()
            R.fit(LL,self.__target)
            return R.feature_scores[0]
        else:
            if(len(self.scores.keys())==0):
                self.scores = {}
                cpt = 0
                for r in self.feature_scores:
                    tu = cpt,r
                    self.feature_scores[cpt] = r
                    cpt = cpt + 1
            return self.feature_scores[x[0]]



    def transform(self, X):
        """Reduces the feature set down to the top `n_features_to_keep` features.
        Parameters
        ----------
        X: array-like {n_samples, n_features}
            Feature matrix to perform feature selection on
        Returns
        -------
        X_reduced: array-like {n_samples, n_features_to_keep}
            Reduced feature matrix
        """
        return X[:, self.top_features[:self.n_features_to_keep]]

    def fit_transform(self, X, y):
        """Computes the feature importance scores from the training data, then
        reduces the feature set down to the top `n_features_to_keep` features.

        Parameters
        ----------
        X: array-like {n_samples, n_features}
            Training instances to compute the feature importance scores from
        y: array-like {n_samples}
            Training labels

        Returns
        -------
        X_reduced: array-like {n_samples, n_features_to_keep}
            Reduced feature matrix

        """
        self.fit(X, y)
        return self.transform(X)


noms_mesures = ["ReliefF"]
classe_mesure = ReliefF

#data,target = load_german_dataset()
#R = ReliefF()#
#R.fit(data,target)
#print(R.rank_with(["ReliefF"],n=1))