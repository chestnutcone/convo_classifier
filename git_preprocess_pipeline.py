# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 11:14:57 2019

@author: Oliver
"""

import json
import datetime
import pickle
import os

def binary_search(data, val):
    # takes the json_convo['messages'] as data
    # val is start_time value of timestamp in seconds
    
    # binary search for msg start
    l = 0
    r = len(data)-1
    best_idx = l
    while l<r:
        mid = (l+r) //2
        timestamp = data[mid]['timestamp_ms'] // 1000 # get seconds
        if timestamp < val:
            # move right
            l = mid + 1
        elif timestamp > val:
            r = mid-1
        else:
            best_idx = mid
            break

        if abs(data[mid]['timestamp_ms'] // 1000 - val) < abs(data[best_idx]['timestamp_ms'] // 1000 - val):
            best_idx = mid
    return best_idx
        
def preprocess(path, start_time, save_name):
    # start_time will be timestamp in seconds
    # path is path to file
    
    with open(path, 'r') as f:
        json_convo = json.load(f)
    assert len(json_convo['participants']) == 2
    participants = {p['name']:i for i,p in enumerate(json_convo['participants'])}     
    
    chron_message = json_convo['messages'][::-1]
    convo_start = binary_search(chron_message, start_time)

    last_msg = None
    total_msg = []

    for i in range(convo_start, len(chron_message)):
        json_msg = chron_message[i]
        if 'content' not in json_msg.keys(): continue
    
        if last_msg != participants[json_msg['sender_name']]:
            try:
                total_msg.append(temp_aggre)
            except:
                # except first one
                pass 
            temp_aggre = {'content':[json_msg['content']]}
            temp_aggre['timestamp_s'] = json_msg['timestamp_ms'] // 1000
            temp_aggre['sender_name'] = json_msg['sender_name']
        else:
            # keep aggregating
            temp_aggre['content'].append(json_msg['content'])
        last_msg = participants[json_msg['sender_name']]
        
    # append last round
    total_msg.append(temp_aggre)
    with open('preprocess_{}.pickle'.format(save_name), 'wb') as f:
        pickle.dump(total_msg, f)