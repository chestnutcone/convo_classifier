# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 12:32:59 2019

@author: Oliver
"""

import pandas as pd
import pickle

def process_pd(f_path, save_name):
    # process data
    with open(f_path, 'rb') as f:
        data = pickle.load(f)
        
    # do block-wise operation
    df = {'msg_count':[],
          'word_count':[],
          'char_count':[],
          'avg_msg_len':[],
          'upper_cs_count':[],
          'upper_cs_avg':[],
          'msg_time_diff':[],
          'ori_msg':[],
          'sender_name':[]}
    
    # first timestamp will be 0
    last_ts = data[0]['timestamp_s']
    for msg in data:
        df['msg_count'].append(len(msg['content']))
        words = ' '.join(msg['content'])
        words = words.split()
        df['word_count'].append(len(words))
        df['avg_msg_len'].append(len(words)/len(msg['content']))
        
        words = ''.join(words)
        df['char_count'].append(len(words))
        upper_count = 0
        for w in words:
            if w.isupper():
                upper_count += 1
        
        df['upper_cs_count'].append(upper_count)
        df['upper_cs_avg'].append(upper_count/len(words))
        df['ori_msg'].append(msg['content'])
        df['msg_time_diff'].append(msg['timestamp_s']-last_ts)
        df['sender_name'].append(msg['sender_name'])
        last_ts = msg['timestamp_s']
        
    pd_df = pd.DataFrame(df)
    pd_df.to_pickle('pd_{}.pickle'.format(save_name))