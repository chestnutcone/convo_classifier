# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 16:16:10 2019

@author: Oliver
"""

import pickle
import matplotlib.pyplot as plt
from keras import layers, models
from sklearn.metrics import f1_score, confusion_matrix


NUM_COLS = 7
MAX_LEN = 50
VOCAB_SIZE = 1000

def build_model():
    num_inp = layers.Input(shape=(NUM_COLS,))
    word_inp = layers.Input(shape=(MAX_LEN,))
    
    embed = layers.Embedding(input_dim=VOCAB_SIZE, 
                             output_dim=8, 
                             input_length=MAX_LEN)(word_inp)
    #process embed on conv1d then LSTM
    text = layers.Conv1D(filters=32,
                          kernel_size=3,
                          activation='relu')(embed)
    text = layers.MaxPooling1D(3)(text)
    text = layers.Flatten()(text)
    # text = layers.LSTM(16)(text)
    
    combined = layers.concatenate([num_inp, text], axis=1)
    final = layers.Dense(16, activation='relu', name='layer32')(combined)
    final = layers.Dense(1, activation='sigmoid')(final)
    
    model = models.Model(inputs=[num_inp, word_inp],
                         outputs=final)
    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['binary_accuracy'])
    # print('model_summary')
    # print(model.summary())
    return model

def graph_history(history):
    acc = history.history['binary_accuracy']
    val_acc = history.history['val_binary_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    run_length = range(len(acc))
    plt.plot(run_length, acc, label='binary_accuracy')
    plt.plot(run_length, val_acc, label='val_binary_accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('accuracy')
    plt.legend()

    plt.figure()
    plt.plot(run_length, loss, label='loss')
    plt.plot(run_length, val_loss, label='val_loss')
    plt.xlabel('Epochs')
    plt.ylabel('loss')
    plt.legend()
    plt.show()

def plot_confusion_matrix(cm, cmap=plt.cm.Blues):
    fig, ax = plt.subplots(1,1)
    img = ax.imshow(cm, cmap=cmap, extent=[-1,1,-1,1])
    labels = ['Not Me', 'Me']
    ax.xaxis.tick_top()
    ax.set_xticks([-0.5,0.5])
    ax.set_xticklabels(labels)
    ax.set_yticks([-0.5,0.5])
    ax.set_yticklabels(labels[::-1])
    ax.set_xlabel('Predicted')
    ax.xaxis.set_label_position('top')
    ax.set_ylabel('True')
    fig.colorbar(img)
    
def plot_data(y_pred, y_true):
    THRESHOLD = 0.5
    mod_predict = (y_pred>=THRESHOLD).astype(int)
    f1_val = f1_score(y_true, mod_predict)
    print('f1 score', f1_val)
    cm = confusion_matrix(y_true, mod_predict)
    print('confusion matrix')
    print(cm)
    fp = cm[0][1]
    tp = cm[1][1]
    fn = cm[1][0]
    precision = tp/(tp+fp)
    recall = tp/(fn+tp)
    print('precision: {}  recall: {}'.format(precision, recall))
    plot_confusion_matrix(cm)
    
    
def try_model(neural_inp_path, model_name, save_model=True):
    with open(neural_inp_path, 'rb') as f:
        data = pickle.load(f)
        
    x_train, x_test, y_train, y_test = data['train_data']
    model = build_model()
    history = model.fit(x=[x_train[:,:NUM_COLS],
                           x_train[:,NUM_COLS:]],
                        y=y_train,
                        validation_data=([x_test[:,:NUM_COLS],
                                          x_test[:,NUM_COLS:]],y_test),
                        epochs=3,
                        batch_size=128)
    
    graph_history(history)
    predict = model.predict([x_test[:,:NUM_COLS],
                             x_test[:,NUM_COLS:]])
    plot_data(predict, y_test)
    model.save(model_name)