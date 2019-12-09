# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 12:24:54 2019

@author: Oliver
"""

import datetime
from git_preprocess_pipeline import preprocess
from git_process import process_pd
from git_keras_preprocessing import keras_process
from git_keras_train import try_model
from git_keras_visualize import visualize_keras
from git_plot_roc import graph_roc

PATH_TO_MSG_JSON = r""
SAVE_NAME = ''
MAIN_CHAR = '' # sender name of main char as appeared in facebook sender
SUB_CHAR = '' # sender name of the other person 
MODEL_NAME = 'something.h5'
START_TIME = datetime.datetime(2014,1,1).timestamp()

preprocess(PATH_TO_MSG_JSON, START_TIME, SAVE_NAME)
process_pd('preprocess_{}.pickle'.format(SAVE_NAME), SAVE_NAME)
keras_process('pd_{}.pickle'.format(SAVE_NAME), SAVE_NAME, MAIN_CHAR, SUB_CHAR)
try_model('neural_inp_{}.pickle'.format(SAVE_NAME), MODEL_NAME, save_model=True)
visualize_keras(MODEL_NAME, 'neural_inp_{}.pickle'.format(SAVE_NAME))
graph_roc(MODEL_NAME, 'neural_inp_{}.pickle'.format(SAVE_NAME))
