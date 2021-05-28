# -*- coding: utf-8 -*-
"""
Created on Sun Jan  3 20:12:13 2021

@author: HO18971
"""
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.layers import Dense
from keras.models import Sequential
from keras.regularizers import l1, l2
from keras.optimizers import Adam, SGD
from keras.wrappers.scikit_learn import KerasClassifier
from keras.layers import Dropout
from keras.activations import relu
from keras.activations import sigmoid
from numpy.random import seed
import random
import matplotlib as mpl
import matplotlib.pyplot as pl
import pandas as pd
import copy
import numpy as np
import tensorflow as tf
mpl.rcParams['xtick.labelsize'] = 9
mpl.rcParams['axes.labelsize'] = 4
mpl.rcParams['font.size'] = 8
mpl.rcParams['axes.labelsize'] = 6
pd.get_option('display.max_columns')
random.seed(0)
seed(0)
tf.random.set_seed(0)


n_samples = 100000
n_features = 3
X = np.random.binomial(1, 0.5, (n_samples, n_features))
y = []
for i in range(X.shape[0]):  
    v1 = 1 * (X[i][0]==1)
    v2 = 1 * (X[i][1]==1)
    v3 = 1 * (X[i][2]==1)
    y.append(v3*v1 + (1-v3)*v2)



n_noise = 7
X_noise = np.random.binomial(1, 0.5, (n_samples, n_noise))
n_features += n_noise
X = np.hstack((X, X_noise))


y = np.array(y)

train_size = 0.8
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, random_state=0)





best_param_nn = 0.02#.25#.21# 1
# best_param_nn_lasso = 0.35#.21# 1
p = 2 # number of relu layer
n_neurons = 6

def nn_model():
    mod = Sequential()
    #mod.add(Dropout(0.01))
    mod.add(Dense(4, use_bias=True, input_dim=X_train.shape[1], activation='relu', #,# kernel_initializer=Constant(0.001),
                       #kernel_regularizer=l1(best_param_nn_lasso)))
                       activity_regularizer=l1(best_param_nn)))#,# bias_regularizer=l1(best_param_nn),
                       #activity_regularizer=l2(best_param_nn),))
                       #kernel_initializer=Constant(0.0001) ))
#    mod.add(Dropout(0.5))
    mod.add(Dense(2, use_bias=True, activation='relu'))#, kernel_initializer=Constant(0.001)))

    #mod.add(Dense(2, use_bias=True, activation='relu'))#, kernel_initializer=Constant(0.001)))
                       #kernel_regularizer=l1(best_param_nn),))# bias_regularizer=l1(best_param_nn),
                       #activity_regularizer=l1(best_param_nn),))
#    mod.add(Dropout(0.5))    
    
    mod.add(Dense(1, use_bias=True, activation='sigmoid'))
    #opt = SGD(lr=0.01) #
    opt=Adam(lr=0.01)#learning_rate=0.01)
#    opt=Adagrad(lr=0.01)#learning_rate=0.01)

    mod.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
#    model_nn.fit(X_train, y_train, epochs=10, class_weight=class_weights_dict)
#    model_nn.evaluate(X_train, y_train)
    return mod


keras_model = nn_model()
history = keras_model.fit(X_train, y_train, epochs=10, batch_size=100)#, class_weight=class_weights_dict)
print('performance on train set: ', keras_model.evaluate(X_train, y_train))
print('performance on test set: ', keras_model.evaluate(X_test, y_test))

# collect weights and bias
W = []
for i_relu_layer in np.arange(p+2)[::2]:
    W.append(keras_model.weights[i_relu_layer].numpy())
W.append(keras_model.weights[p+2].numpy())

B = []
for i_relu_layer in np.arange(1,p+3)[::2]:
    B.append(np.expand_dims(keras_model.weights[i_relu_layer].numpy(), 0))
B.append(keras_model.weights[(p*2)+1].numpy())

# compute activations/z
A = []
Z = []
for k in range(p):
    if k == 0:
        A.append(np.dot(X_train, W[0]) + B[0])
        Z.append(relu(A[0]).numpy())
    else:
        A.append(np.dot(Z[k-1], W[k]) + B[k])
        Z.append(relu(A[k]).numpy())
A.append(np.dot(Z[p-1], W[p]) + B[p])
Z.append(sigmoid(A[p]))

# check activations/z
pred = keras_model.predict(X_train)
assert np.round(Z[-1].numpy().sum(), 0) == np.round(pred.sum(), 0)

# compute effective weights/bias
W_eff = np.zeros(X_train.shape)
for en, x in enumerate(X_train):
    w_temp = W[0]
    for k in range(p):
        w_temp = np.dot(w_temp * (A[k][en]>0), W[k+1])
    W_eff[en, :] = w_temp.T[0]
    
B_eff = np.zeros(X_train.shape[0])
for en, x in enumerate(X_train):
    for k in range(0, p):
        b_temp = B[k]
        for h in range(k, p):
            b_temp = np.dot(b_temp * (A[h][en]>0), W[h+1])
        B_eff[en] = B_eff[en] + copy.deepcopy(b_temp)[0]
B_eff = B_eff + B[p][0]

# check effective weights
zz = np.zeros(X_train.shape[0])
for g in range(X_train.shape[0]):
    zz[g] = np.dot(X_train[g], W_eff[g]) + B_eff[g]
import math
def sigmoid_(x):
  return 1 / (1 + math.exp(-x))
pred_effective = [sigmoid_(zz_) for zz_ in zz]
assert np.round(Z[-1].numpy().sum(), 0) == np.round(np.sum(pred_effective), 0)

# create clusters based on Z
mat = Z[0]
for z in range(1, p):
    mat = np.hstack((mat, Z[z]))
mat = mat>0
clu = mat.dot(1 << np.arange(mat.shape[-1]))



#fig, axes = pl.subplots(int(np.sqrt(np.unique(clu).shape)+1), int(np.sqrt(np.unique(clu).shape)+1))
df_res = pd.DataFrame()
fig, axes = pl.subplots(2, 4)

df_metaresults = pd.DataFrame()
for en, kk in enumerate(np.unique(clu)):
    explaination_ = W_eff[clu==kk, :] #####+ np.expand_dims(B_eff[clu==kk], 1)
    pred_cluster = 1*(keras_model.predict(X_train)[clu==kk]>0.5)
    positive = np.round(np.sum(pred_cluster)/pred_cluster.shape[0], 2)
    df_res['Feature'] = np.arange(1, n_features+1)
    df_res['Feature Importance'] = np.mean(explaination_, 0)
    df_res['Abs score'] = np.abs(df_res['Feature Importance'])
    df_plot = df_res.sort_values(['Abs score'], ascending=False)
    
    seq = "{0:b}".format(kk).zfill(n_neurons)[::-1]
    df_metaresults.loc[seq, 'Support'] = pred_cluster.shape[0]/(n_samples*train_size)#np.round(pred_cluster.shape[0]/(n_samples*train_size),2)
    df_metaresults.loc[seq, 'Positive'] = positive
    df_metaresults.loc[seq, 'Avg instance'] = str(list( np.round((X_train)[clu==kk].mean(0), 2) )) #np.round(
    df_metaresults.loc[seq, np.arange(1, n_features+1)] = df_res['Feature Importance'].values
    #string_title = "neur: " + "{0:b}".format(kk).zfill(n_neurons)[::-1] + ' tot: ' + str(np.round(pred_cluster.shape[0]/(n_samples*train_size),2)) + ' pos: ' + str(positive) + ' example: ' + str(list((X_train)[clu==kk].mean(0)))
    #print(string_title)
    df_plot.iloc[:].plot.bar(color='blue', fontsize=20, x='Feature', y='Feature Importance', rot=0, legend=False)#, ax=axes.flatten()[en])
    #df_plot.iloc[:5].plot.bar(fontsize=10, x='Feature', y='Feature Importance', rot=0, legend=False)



print(df_metaresults.round(2))







def nn_model():
    mod = Sequential()
    mod.add(Dense(4, use_bias=True, input_dim=X_train.shape[1], activation='relu'))
    mod.add(Dense(2, use_bias=True, activation='relu'))
    mod.add(Dense(1, use_bias=True, activation='sigmoid'))
    opt=Adam(lr=0.01)
    mod.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    return mod



