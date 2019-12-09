# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 00:29:41 2019

@author: Oliver
"""

import pickle
import matplotlib.pyplot as plt
import numpy as np
from keras.models import load_model
from sklearn.metrics import confusion_matrix


NUM_COLS = 7

def graph_roc(model_path, input_path):
    with open(input_path, 'rb') as f:
        data = pickle.load(f)
        
    x_train, x_test, y_train, y_test = data['train_data']
    model1 = load_model(model_path)
    model1.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['binary_accuracy'])
    
    
    model1_pred = model1.predict([x_test[:,:NUM_COLS],
                                  x_test[:,NUM_COLS:]])
    
    total_cm = {'model1':[]}
    for threshold in np.arange(0.1,1,0.1):
        mod_model1_pred = (model1_pred >= threshold).astype(int)
        cm_m1 = confusion_matrix(y_test, mod_model1_pred)
        total_cm['model1'].append(cm_m1)
        
    def get_roc(vals):
        model1_roc = []
        for cm in vals:
            fp = cm[0][1]
            tn = cm[0][0]
            
            fn = cm[1][0]
            tp = cm[1][1]
            
            fpr = fp/(tn+fp)
            tpr = tp/(fn+tp)
            model1_roc.append([fpr,tpr])
        return model1_roc
        
        
    model1_roc = get_roc(total_cm['model1'])
    model1_roc = np.array(model1_roc)
    diag = np.linspace(0,1,200)
    
    plt.scatter(model1_roc[:,0], model1_roc[:,1], label='model1')
    plt.scatter(diag, diag, label='ref', s=10)
    plt.xlim(right=1, left=0)
    plt.ylim(top=1, bottom=0)
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.legend()
    plt.show()

    