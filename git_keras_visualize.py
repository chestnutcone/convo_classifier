# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 23:58:58 2019

@author: Oliver
"""
import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from keras.models import load_model, Model

NUM_COLS = 7

def visualize_keras(model_path, input_path):
    with open(input_path, 'rb') as f:
        data = pickle.load(f)
        
    x_train, x_test, y_train, y_test = data['train_data']
    model = load_model(model_path)
    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['binary_accuracy'])
    
    intermediate_model = Model(inputs=model.input,
                               outputs=model.get_layer('layer32').output)
    
    mid_out = intermediate_model.predict([x_test[:,:NUM_COLS],
                                          x_test[:,NUM_COLS:]])
    
    pca_2d = PCA(n_components=2)
    transformed_2d = pca_2d.fit_transform(mid_out)
    print('variance', pca_2d.explained_variance_ratio_)
    
    np_y_test = y_test.to_numpy()
    ones = np_y_test.nonzero()
    zeros = np.where(np_y_test==0)
    
    plt.scatter(transformed_2d[ones,0], transformed_2d[ones,1], alpha=0.05, label='Me')
    plt.scatter(transformed_2d[zeros,0], transformed_2d[zeros,1], alpha=0.05,label='Not Me')
    plt.legend()
    plt.show()