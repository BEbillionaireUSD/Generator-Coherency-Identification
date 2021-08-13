#! -*- coding: utf-8 -*-
from util import load_data, add_noise
from DTC import DTC
from warnings import simplefilter
from keras import optimizers
from sklearn.decomposition import KernelPCA
import os
import csv
import argparse
import time
import pandas as pd
import numpy as np
simplefilter(action='ignore', category=FutureWarning)

MODELS = ['DTC', 'LSTM_AE', 'Conv_AE', 'SAE', 'VAE']
gamma, alpha = 1.0, 1.0
kernel_size, pool_size, strides = 10, 10, 1
epochs, eval_epochs, save_epochs = 1, 1, 10
tol, patience = 1e-10, 3

n_units = [128,1]
n_filters = 50
batch_size = 512
pretrain_epochs = 300

cluster_init = 'kmeans' #'hierarchical'
dist_metric = 'eucl'  # eucl', 'cid'

pretrain_optimizer = 'adam'
optimizer = 'adam'
#optimizer = optimizers.Adam(lr=0.00001, clipnorm=1.)

def get_res(y_pred, shift=2):
    results = {}
    for idx, cluster in enumerate(y_pred):
        if cluster not in results:
            results[cluster] = []
        results[cluster].append(str(idx+shift))
    return results

def init(X_train, MODEL='DTC', n_clusters=5, ae_weights=None, save_dir=''):
    dtc = DTC(n_clusters=n_clusters, input_dim=X_train.shape[-1], timesteps=X_train.shape[1], n_filters=n_filters, kernel_size=kernel_size,
              strides=strides, pool_size=pool_size, n_units=n_units, alpha=alpha, dist_metric=dist_metric, cluster_init=cluster_init)

    dtc.initialize(MODEL)
    #dtc.model.summary()
    # print(dtc.model.get_config())
    dtc.compile(gamma=gamma, optimizer=optimizer)

    # Load pre-trained AE weights or pre-train
    mse_loss = None
    if ae_weights is None and pretrain_epochs > 0:
        dtc.pretrain(X=X_train, optimizer=pretrain_optimizer, epochs=pretrain_epochs, batch_size=batch_size, save_dir=save_dir)
    elif ae_weights is not None:
        dtc.load_ae_weights(ae_weights)
        #dtc.autoencoder.compile(optimizer=optimizer, loss='mse')
        #dtc.autoencoder.evaluate(X_train, X_train)
    '''
    from sklearn.feature_selection import mutual_info_regression 
    y = dtc.encoder.predict(X_train)
    y = y.squeeze()
    X_in = X_train.squeeze()
    print(y.shape)
    res = 0
    for i in range(y.shape[1]):
        lie = y[:, i].squeeze()
        mi = mutual_info_regression(X_in, lie).sum() / X_in.shape[1]
        res += mi
    print('IAE Munual Information =', res)
    '''
    start_time = time.time()
    labels = dtc.init_cluster_weights(X_train)
    loss = dtc.fit(X_train, epochs, eval_epochs, save_epochs, batch_size, tol, patience, save_dir) # KL loss
    identify_time = time.time()-start_time

    return labels, loss[2], loss[1], identify_time

if __name__ == "__main__":
    X, names = load_data(file_path='../scope-511/all.csv', sheet_idx=0, time=[], time_interval=0, snr=0, show=False)
    print(X.shape)
    #X_mi = X[:,:pool_size*8,:]
    #print(X_mi.shape)
    save_dir = 'weights/发电机511/all'
    MODEL = 'DTC'

    for n_cluster in range(3,8):
        with open('result/scope-511-all_'+str(n_cluster)+'组.txt', 'w') as log:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            ae_weights_path = '{}/ae_weights-epoch{}.h5'.format(save_dir, pretrain_epochs)
            print('#####', n_cluster)
            if not os.path.exists(ae_weights_path):
                y_pred, kl_loss, mse_loss,_ = init(X, MODEL=MODEL, n_clusters=n_cluster, save_dir=save_dir)
            else:
                y_pred, kl_loss, mse_loss,_ = init(X, MODEL=MODEL, ae_weights=ae_weights_path, n_clusters=n_cluster)
            results = get_res(y_pred, shift=1)
            log.write('## 模型：'+MODEL+'##\n')
            log.write('KL, MSE: '+str(kl_loss)+'\t'+str(mse_loss)+'\n')
            c = 1
            for cluster in results:
                log.write('组'+str(c)+'\t'+'\t'.join(results[cluster])+'\n')
                c+=1
            log.write('##############################')
    exit()
