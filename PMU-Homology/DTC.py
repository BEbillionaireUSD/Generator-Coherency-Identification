#! -*- coding: utf-8 -*-
import os
import csv
import argparse
from time import time
import pandas as pd
import numpy as np

from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

from keras.models import Model
from keras.layers import Dense, Reshape, UpSampling2D, Conv2DTranspose, GlobalAveragePooling1D, Softmax
#from keras.losses import kullback_leibler_divergence
import keras.backend as K


import sklearn.metrics as sm
from sklearn.cluster import AgglomerativeClustering, KMeans
import tensorflow as tf

import tsdistances
from TSClusteringLayer import TSClusteringLayer
from TAE import *


def kullback_leibler_divergence(y_true, y_pred):
    y_true = K.clip(y_true, K.epsilon(), 1)
    y_pred = K.clip(y_pred, K.epsilon(), 1)
    kl = K.sum(y_true * K.log(y_true / y_pred + 1e-10), axis=-1)
    return kl


class DTC:
    def __init__(self, n_clusters, input_dim, timesteps, n_filters=50, kernel_size=10, strides=1, pool_size=10, n_units=[50, 1], alpha=1.0, dist_metric='eucl', cluster_init='kmeans'):
        try:
            assert(timesteps % pool_size == 0)
        except:
            print(timesteps, pool_size)
            exit()
        self.n_clusters = n_clusters
        self.input_dim = input_dim
        self.timesteps = timesteps
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.pool_size = pool_size
        self.n_units = n_units
        self.latent_shape = (self.timesteps // self.pool_size, self.n_units[1])
        self.alpha = alpha
        self.dist_metric = dist_metric
        self.cluster_init = cluster_init
        self.pretrained = False
        self.model = self.autoencoder = self.encoder = self.decoder = None

    def initialize(self, MODEL):
        if MODEL == 'DTC':
            self.autoencoder, self.encoder, self.decoder = temporal_autoencoder(input_dim=self.input_dim, timesteps=self.timesteps, \
                                                                                n_filters=self.n_filters, kernel_size=self.kernel_size, \
                                                                                strides=self.strides, pool_size=self.pool_size, n_units=self.n_units)
        elif MODEL == 'LSTM_AE':            
            self.autoencoder, self.encoder, self.decoder = temporal_autoencoder_lstm_ae(input_dim=self.input_dim, timesteps=self.timesteps, n_units=self.n_units)
        
        elif MODEL == 'Conv_AE':
            self.autoencoder, self.encoder, self.decoder = temporal_autoencoder_cnn_ae(
                input_dim=self.input_dim, timesteps=self.timesteps, n_filters=self.n_filters, kernel_size=self.kernel_size,
                strides=self.strides, pool_size=self.pool_size, n_units=[100,2])
        elif MODEL == 'SAE':
            self.autoencoder, self.encoder, self.decoder = temporal_autoencoder_sae(input_dim=self.input_dim, timesteps=self.timesteps)
        elif MODEL == 'VAE':
            self.autoencoder, self.encoder, self.decoder=temporal_autoencoder_vae(input_dim=self.input_dim, timesteps=self.timesteps)
        else:
            print("Unknown Model Type!")
            exit()

        clustering_layer = TSClusteringLayer(self.n_clusters, alpha=self.alpha, dist_metric=self.dist_metric, name='TSClustering')(self.encoder.output)

        self.model = Model(inputs=self.autoencoder.input,
                            outputs=[self.autoencoder.output, clustering_layer])

    def cluster_centers_(self):
        return self.model.get_layer(name='TSClustering').get_weights()[0]

    def weighted_kld(loss_weight):
        #Custom KL-divergence loss with a variable weight parameter
        def loss(y_true, y_pred):
            return loss_weight * kullback_leibler_divergence(y_true, y_pred)
        return loss

    def compile(self, gamma=1.0, optimizer='adam'):
        #gamma: coefficient of TS clustering loss
        self.model.compile(loss=['mse', kullback_leibler_divergence], loss_weights=[
                           1.0, gamma], optimizer=optimizer)

    def load_weights(self, weights_path):
        self.model.load_weights(weights_path)
        self.pretrained = True

    def load_ae_weights(self, ae_weights_path):
        self.autoencoder.load_weights(ae_weights_path)
        self.pretrained = True

    def dist(self, x1, x2):
        if self.dist_metric == 'eucl':
            return tsdistances.eucl(x1, x2)
        elif self.dist_metric == 'cid':
            return tsdistances.cid(x1, x2)
        else:
            raise ValueError('Available distances are eucl or cid!')

    def init_cluster_weights(self, X):
        #X: numpy array containing training set or batch
        assert(self.cluster_init in ['hierarchical', 'kmeans'])
        print('Initializing cluster...')

        features = self.encode(X)
        X_train = features.reshape(features.shape[0], -1)

        if self.cluster_init == 'hierarchical':
            if self.dist_metric == 'eucl':  # use AgglomerativeClustering off-the-shelf
                hc = AgglomerativeClustering(n_clusters=self.n_clusters, affinity='euclidean', linkage='complete').fit(X_train)
            else:  # compute distance matrix using dist
                d = np.zeros((features.shape[0], features.shape[0]))
                for i in range(features.shape[0]):
                    for j in range(i):
                        d[i, j] = d[j, i] = self.dist(features[i], features[j])
                hc = AgglomerativeClustering(n_clusters=self.n_clusters, affinity='precomputed', linkage='complete').fit(d)
            cluster_centers = np.array([features[hc.labels_ == c].mean(axis=0) for c in range(self.n_clusters)]) # compute centroid

        elif self.cluster_init == 'kmeans':
            # fit k-means on flattened features
            km = KMeans(n_clusters=self.n_clusters, tol=1e-10, max_iter=3000).fit(X_train)
            cluster_centers = km.cluster_centers_.reshape(self.n_clusters, features.shape[1], features.shape[2])
        #print(cluster_centers)

        self.model.get_layer(name='TSClustering').set_weights([cluster_centers])
        return km.labels_

    def encode(self, x):
        return self.encoder.predict(x)

    def decode(self, x):
        return self.decoder.predict(x)

    def predict(self, x):
        q = self.model.predict(x, verbose=0)[1]
        return q.argmax(axis=1)

    def target_distribution(q):  # target distribution p which enhances the discrimination of soft label q
        weight = q ** 2 / q.sum(0)
        return (weight.T / weight.sum(1)).T

    def pretrain(self, X, optimizer='adam', epochs=10, batch_size=128, save_dir='results/tmp', verbose=1):
        self.autoencoder.compile(optimizer=optimizer, loss='mse')

        t0 = time()
        history = self.autoencoder.fit(X, X, batch_size=batch_size, epochs=epochs, verbose=verbose)
        #print('Pretraining time: ', time() - t0)
        self.autoencoder.save_weights('{}/ae_weights-epoch{}.h5'.format(save_dir, epochs))
        self.pretrained = True
        return history.history['loss'][-1]

    def fit(self, X_train, epochs=100, eval_epochs=10, save_epochs=10, batch_size=64, tol=0.001, patience=5, save_dir='results/tmp'):

        if not self.pretrained:
            print('Autoencoder was not pre-trained!')
        '''
        # Logging file
        logfile = open(save_dir + '/dtc_log.csv', 'w')
        fieldnames = ['epoch', 'T', 'L', 'Lr', 'Lc']
        logwriter = csv.DictWriter(logfile, fieldnames)
        logwriter.writeheader()
        '''
        y_pred_last = None
        patience_cnt = 0
        Loss_no_drop = 0
        cur_loss, pre_loss = 0, 0
       
        for epoch in range(epochs):

            # Compute cluster assignments for training set
            q = self.model.predict(X_train, batch_size=batch_size)[1] #predicted
            p = DTC.target_distribution(q) # Real
            y_pred = q.argmax(axis=1)

            logdict = dict(epoch=epoch)
            
            loss = self.model.evaluate(X_train, [X_train, p], batch_size=batch_size, verbose=False)
            logdict['L'], logdict['Lr'], logdict['Lc'] = loss[0], loss[1], loss[2]
            #logwriter.writerow(logdict)
            cur_loss = logdict['L']

            print('[Train] - Lr={:f}, Lc={:f} - total loss={:f}'.format(logdict['Lr'], logdict['Lc'], logdict['L']))
            print(y_pred)
            
            if y_pred_last is not None:
                assignment_changes = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
            y_pred_last = y_pred
                
            if epoch > 0 and assignment_changes < tol:
                patience_cnt += 1
                print('Assignment changes {} < {} tolerance threshold. Patience: {}/{}.'.format(assignment_changes, tol, patience_cnt, patience))
                if patience_cnt >= patience:
                    print('Reached max patience. Stopping training.')
                    #logfile.close()
                    break
            else:
                patience_cnt = 0

            self.model.fit(X_train, [X_train, p], epochs=1, batch_size=batch_size, verbose=True)
            '''
            print('*** 逐层输出',epoch, '***')
            intermediate = Model(inputs=self.model.input, outputs=self.model.get_layer('Conv_encode').output)
            out = intermediate.predict(X_train)
            print('Conv1d_encoded', out)
            print('Conv1d_encoded weight', self.model.get_layer(name='Conv_encode').get_weights())
            '''
        #logfile.close()
        #print('Saving model to:', save_dir + '/DTC_model_final.h5')
        #self.model.save_weights(save_dir + '/DTC_model_final.h5')
        return loss

