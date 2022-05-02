# -*- coding: utf-8 -*-
"""
Created on Thu May  6 14:45:29 2021

@author: C7881534
"""

from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
from acoustics.signal import bandpass
from acoustics.atmosphere import Atmosphere
from NominalFreqBands import (check_band_type, octave_low, octave_high,
                                  third_low, third_high,wgA)
import pandas as pd

from scipy import signal
from distcolors import get_distinguishable_colors
import scipy.io.wavfile as wa
from wave_norm import max_wavetype


SOUNDSPEED = 343.0
WAV2PA =   1e6 # Leq  numerical unity ~  Leq acoustic pressure of 120 dB [ref]

def Leq(p,T=1,fs=48000,t_init = [0],ref=WAV2PA):
    # Return the list of Leq,T computed from the list of initial time
    # T: Computation time range [s]
    # t_init: list of initial times [s]
    ref = ref #[Pa]
    L=np.zeros(len(t_init))
    if len(p)< (t_init[-1]+T)*fs:
        print('p is not long enough for the requested t_init and T')
        return None
    for ind,ti in enumerate(t_init):
        p_temp = p[ int(np.round(ti*fs)) : int(np.round((ti+T)*fs))]
        #L[ind] = 10*np.log10(1./T * np.trapz([pow(p_temp[i],2)/(pow(p0,2)) for i in range(len(p_temp))],dx = 1./fs))
        L[ind] = 10*np.log10(1./(T*fs)*sum(p_temp*p_temp)*(ref*ref))
    return L

def Leq_per_band(sig, fs=48000,T=1,t_init = 0, bands=None,ref=WAV2PA):
    band_type = check_band_type(bands)
    if band_type=='octave':
        low = octave_low(bands[0], bands[-1])
        high = octave_high(bands[0], bands[-1])
    elif band_type=='third':
        low = third_low(bands[0], bands[-1])
        high = third_high(bands[0], bands[-1])
        
    levels = np.zeros(bands.size)
    for band in range(bands.size):
        filtered_sig = bandpass(sig, low[band],high[band],
                                                 fs, order=8)
        h2 = filtered_sig**2.0
        levels[band] = Leq(filtered_sig,T=T,fs=fs,t_init = [t_init],ref=ref)
    return levels

def LAeq(sig, fs=48000,T=1,t_init = 0, bands=None,ref=WAV2PA):
    band_type = check_band_type(bands)
    if band_type=='octave':
        low = octave_low(bands[0], bands[-1])
        high = octave_high(bands[0], bands[-1])
        #Corresponding_A_weighting = wg.a_weighting(bands[0], bands[-1])[::3]
        Corresponding_A_weighting = wgA(bands[0], bands[-1])[::3]
    elif band_type=='third':
        low = third_low(bands[0], bands[-1])
        high = third_high(bands[0], bands[-1])
        #Corresponding_A_weighting = wg.a_weighting(bands[0], bands[-1])
        Corresponding_A_weighting = wgA(bands[0], bands[-1])
    levels = np.zeros(bands.size)
    for band in range(bands.size):
        filtered_sig = bandpass(sig, low[band],high[band],
                                                 fs, order=8)
        h2 = filtered_sig**2.0
        levels[band] = Leq(filtered_sig,T=T,fs=fs,t_init = [t_init],ref=ref) + Corresponding_A_weighting[band]
    level =  10.*np.log10(np.sum(10**(levels/10.)))
    return np.round(level,2)

def hist_levels(bins,levels,**kwargs):
    fig, ax = plt.subplots(figsize=kwargs.get('figsize', [8,4]))
    x_label = [str(int(item)) for item in iter(bins)]
    x_ticks= np.arange(len(x_label))
    ax.grid('True')
    ax.bar(x_ticks,levels, facecolor='slategrey')
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_label,rotation=80)
    plt.title(kwargs.get('title', None))
    plt.ylim(kwargs.get('ylim', None))
    plt.xlabel('Frequency bands [Hz]')
    plt.ylabel(kwargs.get('ylabel','[dB]'))
    plt.tight_layout(rect=(0.01,0.01,0.99,0.99))
    
def plot_levels(bins,levels,**kwargs):
    fig, ax = plt.subplots(figsize=kwargs.get('figsize', [8,4]))
    x_label = [str(int(item)) for item in iter(bins)]
    x_ticks= np.arange(len(x_label))
    ax.grid('True')
    ax.plot(x_ticks,levels,'s-',color= kwargs.get('color', [0,0,0]))
    if kwargs.get("plot_gabarit",False):
        gabarit_bands = kwargs.get("gabarit_bands",None)
        gabarit_levels = kwargs.get("gabarit_levels",None)
        ind_min=np.where(bins>=gabarit_bands[0])[0][0]
        ax.plot(x_ticks[ind_min:ind_min+len(gabarit_bands)],gabarit_levels,'-',linewidth=4,color=[0,0,0],alpha=0.5)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_label,rotation=80)
    plt.title(kwargs.get('title', None))
    plt.ylim(kwargs.get('ylim', None))
    plt.xlabel(kwargs.get('xlabel','Frequency bands [Hz]'))
    plt.ylabel(kwargs.get('ylabel','[dB]'))
    plt.tight_layout(rect=(0.01,0.01,0.99,0.99))
    
def plots_levels(bins,Levels,**kwargs):
    fig, ax = plt.subplots(figsize=[8,4])
    x_label = [str(int(item)) for item in iter(bins)]
    x_ticks= np.arange(len(x_label))
    ax.grid('True')
    colors = get_distinguishable_colors(len(Levels))
    for i_levels, levels in enumerate(Levels):
        ax.plot(x_ticks,levels,'s-', color=colors[i_levels],alpha=0.6,label=kwargs.get("labels",None)[i_levels])
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_label,rotation=80)
    plt.title(kwargs.get('title', None))
    plt.xlabel('Frequency bands [Hz]')
    plt.ylabel(kwargs.get("ylabel",'[dB]'))
    if kwargs.get("labels",None)!=['']:
        #plt.legend(loc=0,fontsize=8)
        from matplotlib.font_manager import FontProperties
        fontP = FontProperties()
        fontP.set_size('xx-small')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', prop=fontP)
    plt.tight_layout(rect=(0.01,0.01,0.99,0.99))


def propagation(sig,fs,distance,delay=True):
    A=Atmosphere()
    #ir= A.impulse_response(distance = distance, fs = fs, ntaps = (1024))
    ir= A.impulse_response(distance = distance, fs = fs, ntaps = (next_power_of_2(distance/340*fs)))
    if delay:
        sig = signal.fftconvolve(sig,ir,mode='same')
    return sig /distance

def retropropagation(sig,fs,distance,delay=True):
    A=Atmosphere()
    ir= A.impulse_response(distance = distance, fs = fs, ntaps = (next_power_of_2(distance/340*fs)),inverse = True)
    #print('max ir = {}'.format(np.max(ir)))
    if delay:
        sig = signal.fftconvolve(sig,ir,mode='same')
    return sig*distance


def db2lin(delta_level):
    return 10**(delta_level/20.)

def lin2db(factdist):
    return 20*np.log10(factdist)

def max_wave(sig):
    m = 1
    if isinstance(sig[0],int):
        m = np.iinfo(type(sig[0])).max
    return m         
    
def next_power_of_2(x):  
    return 1 if x == 0 else 2**(int(x) - 1).bit_length()

def FIRfromsensitivity(file,fs=48000,plot_on=False):
    def band_edges_from_bands(bands):
        width = (bands[1]-bands[0])/2
        return np.array([[b-width , b+width] for b in iter(bands)])
    df=pd.read_csv(file, sep="\t", skiprows=[0], names=['Bands','Gains'])
    gains = df['Gains'].to_numpy()
    bands= df['Bands'].to_numpy()

    lenband = len(gains)//2*2+1
    numtaps=513
    taps = signal.firls(numtaps,bands[:lenband-1], np.array([10**((-gain-10)/10) for gain in iter(gains[:lenband-1])]), weight=None, fs=fs)   
    freq_fir, response_fir = signal.freqz(taps)    
    if plot_on:
        plt.figure()
        plt.plot(taps,'.-')
        plt.show()
        plt.figure()
        plt.semilogx(bands,-gains-10,'.-')
        plt.semilogx(0.5*fs*freq_fir/np.pi,10*np.log10( np.abs(response_fir)),'.-',color = 'red')
        plt.axis([20,20e4,-15,10])
        plt.show()
    return taps

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = signal.lfilter(b, a, data)
    return y

def applyFIR(outfilename, infilename,taps):
    fs,sig = wa.read(infilename)   
    #print('type record:{} | max: {}'.format(type(sig[0]), np.max(sig) ))   
    filter_sig = butter_bandpass_filter(signal.fftconvolve(sig,taps,mode='full')  , 25, 15000, fs, order=6)  
    #print('type filtered_record:{} | max: {}'.format(type(filter_sig[0]),np.max(filter_sig)))
    filter_sig2 = np.array([np.int32(np.round(item)) for item in filter_sig])
    #print('type filtered_record2:{} | max: {}'.format(type(filter_sig2[0]),np.max(filter_sig2)))
    wa.write(outfilename,fs,filter_sig2)
     
    return sig
    


