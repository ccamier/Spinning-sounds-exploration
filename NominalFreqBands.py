# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 23:55:33 2020

@author: C7881534
"""

import numpy as np

NOMINAL_OCTAVE_CENTER_FREQUENCIES = np.array([31.5, 63.0, 125.0, 250.0, 
                                              500.0, 1000.0, 2000.0, 
                                              4000.0, 8000.0, 16000.0])
"""Nominal octave center frequencies.
"""

NOMINAL_THIRD_OCTAVE_CENTER_FREQUENCIES = np.array([25.0, 31.5, 40.0, 50.0, 63.0, 80.0, 100.0, 125.0, 160.0, 
                                                    200.0, 250.0, 315.0, 400.0, 500.0, 630.0, 800.0, 1000.0, 
                                                    1250.0, 1600.0, 2000.0, 2500.0, 3150.0, 4000.0, 5000.0, 
                                                    6300.0, 8000.0, 10000.0, 12500.0, 16000.0, 20000.0
                                                    ])
NOMINAL_A_Weighting = {'25.0':-44.7,'31.5': -39.4,'40.0':-34.6,'50.0':-30.2,'63.0':-26.2,'80.0':-22.5,'100.0':-19.1,'125.0':-16.1,
                       '160.0':-13.4,'200.0':-10.9,'250.0':-8.6,'315.0':-6.6,'400.0':-4.8,'500.0':-3.2,'630.0':-1.9,'800.0':-0.8,'1000.0':0.,
                       '1250.0':0.6,'1600.0':1.,'2000.0':1.2,'2500.0':1.3,'3150.0':1.2,'4000.0':1.,'5000.0':0.5,'6300.0':-0.1,'8000.0':-1.1,
                       '10000.0':-2.5,'12500.0':-4.3,'16000.0':-6.6,'20000.0':-9.3}

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx
    idx = find_nearest([1.25,2],value=bands[1]/bands[0])

def third(low_freq=25.,high_freq=20000.):
    bands = NOMINAL_THIRD_OCTAVE_CENTER_FREQUENCIES
    bands1 = bands[bands <=high_freq]
    bands2 = bands1[bands1>=low_freq]
    return bands2

def octave(low_freq=25.,high_freq=20000.):
    bands = NOMINAL_OCTAVE_CENTER_FREQUENCIES
    bands1 = bands[bands <=high_freq]
    bands2 = bands1[bands1>=low_freq]
    return bands2

def check_band_type(bands):
    idx = find_nearest([1.25,2],value=bands[1]/bands[0])
    return ['third','octave'][idx]

def octave_low(lowband,highband):
    """
    low limits of octave frequency bands.
    
    :param bands: Bands of calculation (optional). Only support standard octave and third-octave bands.
    :type bands: :class:`np.ndarray`
    """
    i_start,i_end = find_nearest(NOMINAL_OCTAVE_CENTER_FREQUENCIES, lowband) , find_nearest(NOMINAL_THIRD_OCTAVE_CENTER_FREQUENCIES, highband)
    bands=NOMINAL_OCTAVE_CENTER_FREQUENCIES[i_start:i_end+1]
    return np.array([center_freq/2.**(1/2) for center_freq in iter(bands)])

def octave_high(lowband,highband):
    """
    high limits of octave frequency bands.
    
    :param bands: Bands of calculation (optional). Only support standard octave and third-octave bands.
    :type bands: :class:`np.ndarray`
    """
    i_start,i_end = find_nearest(NOMINAL_OCTAVE_CENTER_FREQUENCIES, lowband) , find_nearest(NOMINAL_THIRD_OCTAVE_CENTER_FREQUENCIES, highband)
    bands=NOMINAL_OCTAVE_CENTER_FREQUENCIES[i_start:i_end+1]
    return np.array([center_freq*2.**(1/2) for center_freq in iter(bands)])

def third_low(lowband,highband):
    """
    low limits of octave frequency bands.
    
    :param bands: Bands of calculation (optional). Only support standard octave and third-octave bands.
    :type bands: :class:`np.ndarray`
    """
    i_start,i_end = find_nearest(NOMINAL_THIRD_OCTAVE_CENTER_FREQUENCIES, lowband) , find_nearest(NOMINAL_THIRD_OCTAVE_CENTER_FREQUENCIES, highband)
    bands=NOMINAL_THIRD_OCTAVE_CENTER_FREQUENCIES[i_start:i_end+1]
    return np.array([center_freq/2.**(1/6) for center_freq in iter(bands)])

def third_high(lowband,highband):
    """
    high limits of octave frequency bands.
    
    :param bands: Bands of calculation (optional). Only support standard octave and third-octave bands.
    :type bands: :class:`np.ndarray`
    """
    i_start,i_end = find_nearest(NOMINAL_THIRD_OCTAVE_CENTER_FREQUENCIES, lowband) , find_nearest(NOMINAL_THIRD_OCTAVE_CENTER_FREQUENCIES, highband)
    bands=NOMINAL_THIRD_OCTAVE_CENTER_FREQUENCIES[i_start:i_end+1]
    return np.array([center_freq*2.**(1/6) for center_freq in iter(bands)])

def wgA(lowband,highband):
    """
    A-weighting array for 1/3 octave bands.
    
    :param bands: Start and end bands of calculation
    :type bands: :class:`np.ndarray`
    """
    i_start,i_end = find_nearest(NOMINAL_THIRD_OCTAVE_CENTER_FREQUENCIES, lowband) , find_nearest(NOMINAL_THIRD_OCTAVE_CENTER_FREQUENCIES, highband)
    return np.array([NOMINAL_A_Weighting[str(freq)] for freq in iter(NOMINAL_THIRD_OCTAVE_CENTER_FREQUENCIES[i_start:i_end+1])])
    