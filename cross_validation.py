
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np

from sklearn.utils import shuffle


def cv_error(estimator, dataset, datalabel, cv=1):
    dataset = pd.DataFrame(dataset)
    datalabel = pd.DataFrame(datalabel)
    length = len(datalabel) / cv
    left = 0
    right = length
    error = 0.0
    dataset, datalabel = shuffle(dataset, datalabel, random_state=0)
    dataset.reset_index(drop = True)
    datalabel.reset_index(drop = True)
    for i in range(cv):
        valset = pd.DataFrame(dataset.iloc[left:right], columns=dataset.columns).reset_index(drop=True)
        trainset = dataset.drop([k for k in range(left, right)]).reset_index(drop=True)
        vallabel = datalabel.iloc[left:right].reset_index(drop=True)
        trainlabel = datalabel.drop([k for k in range(left, right)]).reset_index(drop=True)
        for col in trainset.columns:
            mean = np.mean(trainset[col])
            std = np.std(trainset[col])
            trainset[col] = (trainset[col] - mean) / std
            valset[col] = (valset[col] - mean) / std
        # for row in range(trainset.shape[0]):
        #     for l in trainset.iloc[row]:
        #         # if np.isnan(l):
        #         #     print "yes trainset"
        #         print trainset.iloc[row]

        estimator.fit(trainset, trainlabel)
        #predlabel = estimator[0].predict(valset)
        error += (1 - estimator.score(valset, vallabel))
        left = right
        right += length
        if right > len(datalabel):
            right = len(datalabel)
    return error / float(cv)

