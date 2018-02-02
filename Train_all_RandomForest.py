import numpy as np
import pandas as pd
from dispersion_model_feature import extract_features

#======================Find Zeta values and wave properties======================================================
properties = pd.DataFrame([], columns=['W','L','C','RTF'])
# zetas = pd.DataFrame([], columns=['zeta0','dz','shelf break','wb'])
zetas = pd.DataFrame([], columns=['zeta0','dz','wb'])
n = 10 # numbers of training set for 0 mode
for i in range(n):
    s = 'semi{}.txt'.format(i)
    
    # wave properties
    properties = properties.append(extract_features(s).properties())
    properties = properties.reset_index(drop=True)
    
    # zeta
    zetas = zetas.append(extract_features(s).zetas())
    zetas = zetas.reset_index(drop=True)

m = 10 # 1st mode
for i in range(m):
    s = 'w{}.txt'.format(i)
    
    # wave properties
    properties = properties.append(extract_features(s).properties())
    properties = properties.reset_index(drop=True)
    
    # zeta
    zetas = zetas.append(extract_features(s).zetas())
    zetas = zetas.reset_index(drop=True)

l = 10 # 2st mode
for i in range(l):
    s = 'r{}.txt'.format(i)
    
    # wave properties
    properties = properties.append(extract_features(s).properties())
    properties = properties.reset_index(drop=True)
    
    # zeta
    zetas = zetas.append(extract_features(s).zetas())
    zetas = zetas.reset_index(drop=True)
    
k = 10 # complex solution, labeled as 9
for i in range(k):
    s = 'q{}.txt'.format(i)
    
    # wave properties
    properties = properties.append(extract_features(s).properties())
    properties = properties.reset_index(drop=True)
    
    # zeta
    zetas = zetas.append(extract_features(s).zetas())
    zetas = zetas.reset_index(drop=True)
#======================Train a model============================================================================
# label
y0 = np.zeros(n)
y1 = np.ones(m)
y2 = np.ones(l)*2
y9 = np.ones(k-1)*9
y = np.concatenate((y0,y1,y2,y9))
x = zetas.values

np.random.seed(21)
shuffled = np.random.permutation(len(y))
x = x[shuffled]
y = y[shuffled]
xtrain = x[:int(len(y)*.8)]
ytrain = y[:int(len(y)*.8)]
xtest = x[int(len(y)*.8):]
ytest = y[int(len(y)*.8):]

from sklearn.ensemble import RandomForestClassifier
rnd_clf = RandomForestClassifier(n_estimators = 500, max_leaf_nodes=16, n_jobs=-1)
rnd_clf.fit(x,y)

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
yprd = rnd_clf.predict(x)
cf = confusion_matrix(y, yprd)
plt.matshow(cf, cmap=plt.cm.gray)
plt.xticks([0,1,2,3],rnd_clf.classes_.astype(int).astype(str))
plt.yticks([0,1,2,3],rnd_clf.classes_.astype(int).astype(str))
plt.show()

#==============================================Pickle========================================================
import cPickle
with open('dispersion_modes.pkl', 'wb') as f:
    cPickle.dump(rnd_clf, f)  