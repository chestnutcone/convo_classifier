# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 16:16:19 2019

@author: Oliver
"""
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
from keras import preprocessing


def keras_process(pd_fpath, save_name, main_char, sub_char, max_words=1000, max_len=50):
    with open(pd_fpath, 'rb') as f:
        df = pickle.load(f)
        
    mod_df = df.drop('sender_name', axis=1)
    y_label = pd.get_dummies(df['sender_name'])
    y_label.drop(sub_char, axis=1, inplace=True)
    mod_df = pd.concat((mod_df, y_label), axis=1)
    
    word_msgs = mod_df['ori_msg'].to_numpy()
    words_msgs = [' '.join(x) for x in word_msgs]
    
    input_tokenizer = preprocessing.text.Tokenizer(num_words=max_words)
    input_tokenizer.fit_on_texts(words_msgs)
    input_seq = input_tokenizer.texts_to_sequences(words_msgs)
    input_data = preprocessing.sequence.pad_sequences(input_seq, max_len)
    
    # normalize x data
    y_label = mod_df[main_char]
    x_only_df = mod_df.drop([main_char, 'ori_msg'], axis=1)
    
    transform_col = x_only_df.shape[1]
    x_only_df = np.concatenate((x_only_df.to_numpy(), input_data), axis=1)
    
    x_train, x_test, y_train, y_test = train_test_split(x_only_df, y_label)
    
    x_scaler = StandardScaler()
    x_train_scaled = x_scaler.fit_transform(x_train[:,:transform_col])
    x_test_scaled = x_scaler.transform(x_test[:,:transform_col])
    
    x_train_scaled = np.concatenate((x_train_scaled, x_train[:,transform_col:]), axis=1)
    x_test_scaled = np.concatenate((x_test_scaled, x_test[:,transform_col:]), axis=1)
    
    with open('neural_inp_{}.pickle'.format(save_name), 'wb') as f:
        pickle.dump({'tokenizer':input_tokenizer,
                     'train_data': (x_train_scaled, x_test_scaled, y_train, y_test)}, f)