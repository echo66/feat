import pandas as pd
import numpy as np
from feat import Feat
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve, auc
from sklearn.linear_model import LogisticRegression as LR

df = pd.read_csv('../d_heart.csv')
df.describe()
X = df.drop('class',axis=1).values
y = df['class'].values
n_splits = 5 
kf = KFold(n_splits=n_splits)
kf.get_n_splits(X)

clf = Feat(max_depth=6,
        # max_dim=X.shape[1],
        max_dim=min(50,2*X.shape[1]),
        pop_size=500,
        verbosity=0,
        shuffle=True,
        classification=True,
        functions="+,-,*,/,exp,log,and,or,not,xor,=,<,>,ite,gauss,gauss2d,sign,logit,tanh",
        random_state=42)
lr = LR()
rocs = []
aucs = []
lr_rocs = []
lr_aucs = []

for train_idx, test_idx in kf.split(X):
    clf.fit(X[train_idx],y[train_idx])
    lr.fit(X[train_idx],y[train_idx])
    
    probabilities = clf.predict_proba(X[test_idx])
    lr_probabilities = lr.predict_proba(X[test_idx])
    print('lr_probabilities shape:',lr_probabilities.shape) 
    fpr,tpr,_ = roc_curve(y[test_idx], probabilities)
    lr_fpr,lr_tpr,_ = roc_curve(y[test_idx], lr_probabilities[:,1])

    aucs.append( auc(fpr,tpr) )
    lr_aucs.append( auc(lr_fpr,lr_tpr) )
   

    rocs.append((fpr,tpr))
    lr_rocs.append((lr_fpr,lr_tpr))


import matplotlib.pyplot as plt

colors = plt.cm.Reds(np.linspace(.5, 1, n_splits))

fig = plt.figure()
for i, (fpr,tpr) in enumerate(rocs):
    plt.plot(fpr,tpr, color=colors[i], marker='s',
            linestyle='-', linewidth=1, label='FEAT Fold {:d} (AUC = {:0.2f})'.format(i,aucs[i]))

colors = plt.cm.Blues(np.linspace(0.5, 1, n_splits))
for i, (fpr,tpr) in enumerate(lr_rocs):
    plt.plot(fpr,tpr, color=colors[i],marker='>',
            linestyle='-', linewidth=1, label='LR Fold {:d} (AUC = {:0.2f})'.format(i,lr_aucs[i]))
plt.legend()

plt.show() 
