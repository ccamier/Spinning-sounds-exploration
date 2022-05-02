# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 14:55:51 2021

@author: C7881534
"""

import numpy as np

def max_wavetype(signal):
    while hasattr(signal,'__len__'):
        signal=signal[0]
    wavemax = 1
    if type(signal) is np.float32 :
        #wavemax = np.finfo(np.float32).max
        wavemax = 1
    if (type(signal) is  np.int16 or  type(signal) is np.int32 ):
        wavemax=np.iinfo(type(signal)).max
    return float(wavemax)