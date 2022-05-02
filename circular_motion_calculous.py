# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 12:31:55 2022

@author: C7881534
"""

import streamlit as st
import numpy as np
import scipy
from pylab import plt
import os
import scipy.io.wavfile as wa
from Levels import Leq,Leq_per_band,plot_levels

#streamlit layout
col1,col2 = st.columns([1.5,1])

#global physical parameters
parameters={}
parameters['fs']= 44100 #Hz
parameters['signal_length']= 3. #s
parameters['rho'] =1.# 415. # acoustical impedance at T=20°C
parameters['c']= 343. # sound celerity at T=20°C
parameters['t'] = np.arange(0,parameters['signal_length'],1./parameters['fs'])

#Emission signal parameters
parameters['A_emission'] = 1.
parameters['freq_emission'] = 100. # [Hz]

#Motion parameters
parameters['R_motion']= 5. # [m]
parameters['freq_rotation'] = 99. # rot per sec ~ [Hz] 

#Receiver parameters
parameters['theta_receiver']=1/2*2*np.pi # [radians]
parameters['R_receiver'] = 3.5# [m]

#Computation parameters
parameters['list_options']= ['emission','phased_emis','propag_phased_emis','doppler_propag_phased_emis']
parameters['compute']={}
parameters['audio_folder_name'] = os.path.join(os.getcwd(),'circular_sounds/')

def signals_compute(parameters):
    signals = {}
    if parameters['compute']['emission']:
        emission = np.array([parameters['A_emission']/(4*np.pi*parameters['rho'] * parameters['R_motion']) * np.sin(2*np.pi*parameters['freq_emission']*t_i) for t_i in iter(parameters['t'])])
        signals.update({'emission':emission})
    if parameters['compute']['phased_emis']:
        phase = 2*np.pi/parameters['c'] * np.sqrt(parameters['R_motion']**2 + parameters['R_receiver']**2 + parameters['R_motion']*parameters['R_receiver']*np.cos(2*np.pi*parameters['freq_rotation']*parameters['t']))+parameters['theta_receiver']
        phased_emis = np.array([parameters['A_emission']/(4*np.pi*parameters['rho'] * parameters['R_motion']) * np.sin(2*np.pi*parameters['freq_emission']*t_i+phase[i_t]) for i_t,t_i in enumerate(parameters['t'])])
        signals.update({'phased_emis':phased_emis})
    if parameters['compute']['propag_phased_emis']:
        propag_phased_emis = np.array([phased_emis[i_t]*(4*np.pi*parameters['rho']*parameters['R_motion']) /(4*np.pi*parameters['rho'] * np.sqrt((parameters['R_motion']**2+parameters['R_receiver']**2)))*np.sqrt((1-2*(parameters['R_motion']*parameters['R_receiver'])/(parameters['R_motion']**2+parameters['R_receiver']**2)*np.sin(2*np.pi*parameters['freq_rotation']*t_i))) for i_t,t_i in enumerate(parameters['t'])])
        signals.update({'propag_phased_emis':propag_phased_emis})
    if parameters['compute']['doppler_propag_phased_emis']:   
        doppler_propag_phased_emis = np.array([propag_phased_emis[i_t]/(1+2*np.pi*parameters['freq_rotation']*parameters['R_receiver']/parameters['c']*np.sin(2*np.pi*parameters['freq_rotation']*t_i)) for i_t,t_i in enumerate(parameters['t'])])
        signals.update({'doppler_propag_phased_emis':doppler_propag_phased_emis})
    return signals
    
def fft_compute(parameters,signals):
    freq = np.fft.rfftfreq(parameters['t'].shape[-1],1/parameters['fs'])
    fft_signals = {'freq':freq,'results':{}}
    for key in iter(signals):
        fft_signals['results'].update({'fft_'+key:np.fft.rfft(signals[key])})
    return fft_signals

def signals_plot(parameters,signals):
    plt.figure(figsize=[7,5]);
    figure, axes = plt.subplots()
    for key in iter(signals):
            plt.plot(parameters['t'],signals[key], alpha=0.4,label = key)
    plt.axis([parameters['t'][0],parameters['t'][-1],-5e-2,5e-2])
    plt.legend(loc=1)
    plt.grid()
    plt.xlabel('Time [s]')
    plt.xlabel('Magnitude [ref 1]')
    plt.title('Freq_rotation: {} Hz, Freq_emission: {} Hz'.format(parameters['freq_rotation'],parameters['freq_emission']))
    
    axins = axes.inset_axes([0.7, -0.3, 0.5, 0.5])
    for key in iter(signals):
            axins.plot(parameters['t'],signals[key], alpha=0.4,label = key)
    # sub region of the original image
    x1, x2, y1, y2 = parameters['t'][0],parameters['t'][1000],-5e-2,5e-2;
    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)
    axins.yaxis.tick_right()
    axins.grid('minor')
    [axins.spines[item].set_color('lightgrey') for item in iter(['bottom','top','left','right'])];
    axes.indicate_inset_zoom(axins, edgecolor="lightgrey")
    col1.pyplot(plt);
    
def Leq_plot(parameters,signals):
    plt.figure(figsize=[7,5]);
    figure, axes = plt.subplots()
    t_integr = 0.01; # 1/ parameters['freq_rotation']
    L_t = np.arange(parameters['t'][0],parameters['t'][-1],t_integr)
    signals.update({'L_t':L_t})
    for key in iter(parameters['compute'].keys()):
        if parameters['compute'][key]:
            L = Leq(signals[key],T=t_integr,fs=parameters['fs'],t_init = L_t)
            signals.update({'L_'+key:L})
            plt.plot(L_t,L, alpha=0.4,label = key)
    
    plt.axis([L_t[0],L_t[-1],0,140])
    plt.legend(loc=3)
    plt.grid()
    plt.title('Leq  (over 0.01s-periods)')
    plt.xlabel('Time [s]')
    plt.ylabel('[dB]')
    
    axins = axes.inset_axes([0.7, -0.3, 0.5, 0.5])
    for key in iter(parameters['compute'].keys()):
        if parameters['compute'][key]:
            axins.plot(signals['L_t'],signals['L_'+key], alpha=0.4,label = key)
            last_key=key
    
    # sub region of the original image
    x1, x2, y1, y2 = signals['L_t'][-1]*2/3,signals['L_t'][-1],np.min(signals['L_'+last_key])-2.5 , np.max(signals['L_'+last_key])+2.5;
    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)
    axins.yaxis.tick_right()
    axins.grid('minor')
    [axins.spines[item].set_color('grey') for item in iter(['bottom','top','left','right'])];
    axes.indicate_inset_zoom(axins, edgecolor="grey")
    col1.pyplot(plt);
    #plt.show()
def fft_plot(parameters,fft_signals):
    plt.figure(figsize=[7,5]);
    for key in iter(fft_signals['results']):
            plt.loglog(fft_signals['freq'],np.abs(fft_signals['results'][key]),alpha=0.8,label=key)
    plt.legend(loc=4)
    plt.xlabel('Magnitude [dB]')
    plt.xlabel('Frequency [Hz]')
    plt.grid()
    plt.axis([fft_signals['freq'][1],fft_signals['freq'][-1],1e-21,1e4])
    col1.pyplot(plt);
    #plt.show()
    
def scheme_plot(parameters):
    plt.figure(figsize = [3,3])
    figure, axes = plt.subplots()
    circle = plt.Circle( (0., 0. ), parameters['R_motion'] ,fill = False,color='red',alpha=0.7)
    arrow = plt.arrow(parameters['R_motion'], 0,0, parameters['freq_rotation']/200*9, width = 0.05,head_width = 0.5,color='red',alpha=0.7)
    plt.plot([0, parameters['R_receiver']*np.cos(parameters['theta_receiver'])],[0, parameters['R_receiver']*np.sin(parameters['theta_receiver'])],'-+',color='darkblue')
    plt.plot([0],[0],'o',color='black',markersize=4)
    plt.plot([parameters['R_motion']],[0],'o',color='darkred',markersize=4,label='Source')
    plt.axis([-10,10 ,-10, 10])
    axes.set_aspect( 1 )
    axes.add_artist(circle)
    axes.add_artist(arrow)
    axes.legend(loc=3)
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.title( 'Geometric config.' )
    
    axins = axes.inset_axes([0.7, -0.3, 0.5, 0.5])
    circle2 = plt.Circle( (0., 0. ), parameters['R_motion'] ,fill = False )
    axins.plot([0, parameters['R_receiver']*np.cos(parameters['theta_receiver'])],[0, parameters['R_receiver']*np.sin(parameters['theta_receiver'])],'-+',color='darkblue',label='Receiver')
    axins.plot([0],[0],'o',color='black',markersize=4,label = 'Center')
    axins.add_artist(circle2)
    axins.add_artist(circle2)
    axins.legend(loc=4)
    # sub region of the original image
    x1, x2, y1, y2 = -parameters['R_receiver']*np.sqrt(2), parameters['R_receiver']*np.sqrt(2), -parameters['R_receiver']*np.sqrt(2), parameters['R_receiver']*np.sqrt(2)
    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)
    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)
    axins.yaxis.tick_right()
    axins.grid('minor')
    [axins.spines[item].set_color('lightgrey') for item in iter(['bottom','top','left','right'])];
    axes.indicate_inset_zoom(axins, edgecolor="lightgrey")
    col2.pyplot(figure)
def produce_audio(parameters,signals):
    for signal_key in iter(parameters['compute']):
        if parameters['compute'][signal_key]:
            wa.write(parameters['audio_folder_name']+signal_key+'.wav', parameters['fs'] , np.array([np.int32(val) for val in iter((scipy.signal.windows.tukey(len(signals[signal_key]),.1)*signals[signal_key]/np.max(signals[signal_key]))*(2**30))]));
            audio_file = open(parameters['audio_folder_name']+signal_key+'.wav', 'rb');
            audio_bytes = audio_file.read();
            col2.audio(audio_bytes, format='audio/ogg',title = signal_key);

option = st.sidebar.selectbox("Computed signals",parameters['list_options'],index=3);
# pour changer le nombre de graphes dessinés à l'initialisation, changer l'index de la fonction st.sidebar.selectbox
parameters['freq_rotation'] = st.sidebar.slider('freq_rotation', min_value=0., max_value=200., value=parameters['freq_rotation'], step=0.1);
parameters['R_motion'] = st.sidebar.slider('R_motion', min_value=0., max_value=10., value=parameters['R_motion'], step=0.1);
parameters['freq_emission'] = st.sidebar.slider('freq_emission', min_value=0., max_value=200., value=parameters['freq_emission'], step=0.1);
parameters['R_receiver'] = st.sidebar.slider('R_receiver', min_value=0., max_value=10., value=parameters['R_receiver'], step=0.1);
parameters['theta_receiver'] = st.sidebar.slider('theta_receiver', min_value=0., max_value=2*np.pi, value=parameters['theta_receiver'], step=0.1);

[parameters['compute'].update({key:[True if i_index <=parameters['list_options'].index(option) else False for i_index in range(len(parameters['list_options']))][i_key] }) for i_key,key in enumerate(parameters['list_options'])];
#parameters['list_options'] = [list_options[item] if parameters['compute'][key] else None for item,key in enumerate(list_options)]
#parameters['list_options'].remove(None)
signals = signals_compute(parameters);
signals_plot(parameters,signals);
fft_signals = fft_compute(parameters,signals);
fft_plot(parameters,fft_signals);
scheme_plot(parameters);
Leq_plot(parameters, signals);
produce_audio(parameters,signals);