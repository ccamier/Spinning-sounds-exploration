# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 12:31:55 2022

@author: C7881534
"""

import streamlit as st
import numpy as np
import wave, struct
import matplotlib.pyplot as plt
import os,glob
from PIL import Image

col1,col2 = st.columns([1.5,1]);

#global physical parameters
parameters={}
parameters['fs']= 44100 #Hz
parameters['signal_length']= 3. #s
parameters['rho'] =1.# 415. # acoustical impedance at T=20°C
parameters['c']= 343. # sound celerity at T=20°C
parameters['t'] = np.arange(0,parameters['signal_length'],1./parameters['fs']);

#Emission signal parameters
parameters['A_emission'] = 1.
parameters['freq_emission'] = 100. # [Hz]

#Motion parameters
parameters['R_motion']= 5. # [m]
parameters['freq_rotation'] = 0. # rot per sec ~ [Hz] 

#Receiver parameters
parameters['theta_receiver']=0*2*np.pi # [radians]
parameters['R_receiver'] = .1# [m]

#Computation parameters
parameters['list_options']= ['emission','phased_emis','propag_phased_emis','doppler_propag_phased_emis'];
parameters['compute']={};
parameters['audio_folder_name'] = os.path.join(os.getcwd(),'circular_sounds/');

def wa_write(name,fs=None,data=np.array([])):
    sampleRate = np.float32(fs) # hertz
    obj = wave.open(name,'w')
    obj.setnchannels(1) # mono
    obj.setsampwidth(2)
    obj.setframerate(sampleRate)
    for value in iter(data):
        data = struct.pack('<h', value)
        obj.writeframesraw( data )
    obj.close()

def tukey_win(N,ratio=1.):
    N_ratio = np.int32(N*ratio/2)*2
    hann = np.hanning(N_ratio)
    return np.append(np.append(hann[0:int(N_ratio/2)] , np.ones(N-N_ratio)), hann[int(N_ratio/2):] )

def Leq(p,T=1,fs=48000,t_init = [0],ref=1):
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

def signals_compute(parameters):
    signals = {}
    if parameters['compute']['emission']:
        emission = np.array([parameters['A_emission']/(parameters['rho'] * parameters['R_motion']) * np.sin(2*np.pi*parameters['freq_emission']*t_i) for t_i in iter(parameters['t'])]);
        signals.update({'emission':emission});
    if parameters['compute']['phased_emis']:
        phase = 2*np.pi/parameters['c'] * np.sqrt(parameters['R_motion']**2 + parameters['R_receiver']**2 + parameters['R_motion']*parameters['R_receiver']*np.cos(2*np.pi*parameters['freq_rotation']*parameters['t']))+parameters['theta_receiver'];
        phased_emis = np.array([parameters['A_emission']/(parameters['rho'] * parameters['R_motion']) * np.sin(2*np.pi*parameters['freq_emission']*t_i+phase[i_t]) for i_t,t_i in enumerate(parameters['t'])]);
        signals.update({'phased_emis':phased_emis});
    if parameters['compute']['propag_phased_emis']:
        propag_phased_emis = np.array([phased_emis[i_t]*(parameters['rho']*parameters['R_motion']) /(parameters['rho'] * np.sqrt((parameters['R_motion']**2+parameters['R_receiver']**2)))*np.sqrt((1-2*(parameters['R_motion']*parameters['R_receiver'])/(parameters['R_motion']**2+parameters['R_receiver']**2)*np.sin(2*np.pi*parameters['freq_rotation']*t_i))) for i_t,t_i in enumerate(parameters['t'])]);
        signals.update({'propag_phased_emis':propag_phased_emis});
    if parameters['compute']['doppler_propag_phased_emis']:   
        doppler_propag_phased_emis = np.array([propag_phased_emis[i_t]/(1+2*np.pi*parameters['freq_rotation']*parameters['R_receiver']/parameters['c']*np.sin(2*np.pi*parameters['freq_rotation']*t_i)) for i_t,t_i in enumerate(parameters['t'])]);
        signals.update({'doppler_propag_phased_emis':doppler_propag_phased_emis});
    return signals
    
def fft_compute(parameters,signals):
    freq = np.fft.rfftfreq(parameters['t'].shape[-1],1/parameters['fs'])
    signals.update({'freq':freq})
    for key in iter(parameters['compute'].keys()):
        if parameters['compute'][key]:
            signals.update({'fft_'+key:10.*np.log10(np.abs(np.fft.fft([2/parameters['fs']*val**2 for val in iter(signals[key][:len(signals['freq'])])])))});
    return signals

def signals_plot(parameters,signals):
    col1.subheader("Simulated signals")
    col1.text('Computed with:\n Source frequency: {} Hz\n Rotation frequency: {} Hz\n R receiver: {} m\n theta receiver: {} m\n  '.format(parameters['freq_emission'],parameters['freq_rotation'],parameters['R_receiver'],parameters['theta_receiver']))
    plt.figure(figsize=[7,5]);
    figure, axes = plt.subplots()
    for key in iter(signals):
            plt.plot(parameters['t'],signals[key], alpha=0.4,label = key);
    plt.axis([parameters['t'][0],parameters['t'][-1],-np.max(signals['emission'])*1.4,np.max(signals['emission'])*1.4])
    plt.legend(loc=1)
    plt.grid()
    plt.xlabel('Time [s]')
    plt.ylabel('Magnitude [ref 1]')
    axins = axes.inset_axes([0.7, -0.3, 0.5, 0.5])
    for key in iter(signals):
            axins.plot(parameters['t'],signals[key], alpha=0.4,label = key);
    # sub region of the original image
    x1, x2, y1, y2 = parameters['t'][0],parameters['t'][1000],-np.max(signals[key])*1.1, np.max(signals[key])*1.1;
    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)
    axins.yaxis.tick_right()
    for item in iter(['bottom','top','left','right']):
        axins.spines[item].set_color('grey');
    axes.indicate_inset_zoom(axins, edgecolor=[0.2,0.2,0.2])
    axins.grid('minor')
    col1.pyplot(plt);
    
def Leq_plot(parameters,signals):
    col2.subheader("Leq  (over 0.01s-periods)")
    plt.figure(figsize=[7,5]);
    figure, axes = plt.subplots()
    t_integr = 0.01; # 1/ parameters['freq_rotation']
    L_t = np.arange(parameters['t'][0],parameters['t'][-1],t_integr)
    signals.update({'L_t':L_t})
    for key in iter(parameters['compute'].keys()):
        if parameters['compute'][key]:
            L = Leq(signals[key],T=t_integr,fs=parameters['fs'],t_init = L_t,ref=2e5);
            signals.update({'L_'+key:L});
            plt.plot(L_t,L, alpha=0.4,label = key);
    
    plt.axis([L_t[0],L_t[-1],50,120])
    plt.legend(loc=3)
    plt.grid()
    plt.xlabel('Time [s]')
    plt.ylabel('Leq [dB]')
    
    axins = axes.inset_axes([0.7, -0.3, 0.5, 0.5])
    for key in iter(parameters['compute'].keys()):
        if parameters['compute'][key]:
            axins.plot(signals['L_t'],signals['L_'+key], alpha=0.4,label = key);
            last_key=key;
    
    # sub region of the original image
    x1, x2, y1, y2 = signals['L_t'][-1]*2/3,signals['L_t'][-1],np.min(signals['L_'+last_key])-1 , np.max(signals['L_'+last_key])+1;
    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)
    axins.yaxis.tick_right()
    axins.grid('minor')
    for item in iter(['bottom','top','left','right']):
        axins.spines[item].set_color('grey');
    axes.indicate_inset_zoom(axins, edgecolor=[0.2,0.2,0.2])
    col2.pyplot(plt);
    #plt.show()
def fft_plot(parameters,signals):
    col2.subheader("Fourier transforms")
    plt.figure(figsize=[7,5]);
    for key in iter(parameters['compute'].keys()):
        if parameters['compute'][key]:
            plt.semilogx(signals['freq'],signals['fft_'+key],alpha=0.8,label=key);
    plt.legend(loc=4)
    plt.axis([signals['freq'][1],2e4,-120,10])
    plt.ylabel('Magnitude [dB]')
    plt.xlabel('Frequency [Hz]')
    plt.grid('minor')
    
    col2.pyplot(plt);
    #plt.show()
    
def scheme_plot(parameters):
    col1.subheader("Geometric configuration")
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
    
    axins = axes.inset_axes([0.7, -0.3, 0.5, 0.5])
    circle2 = plt.Circle( (0., 0. ), parameters['R_motion'], color='red',fill = False )
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
    for item in iter(['bottom','top','left','right']):
        axins.spines[item].set_color('lightgrey');
    axes.indicate_inset_zoom(axins, edgecolor="lightgrey")
    col1.pyplot(figure)
    #plt.show()
    
def produce_audio(parameters,signals):
    col2.subheader("Audio files")
    signals.update({'audio_files':{}})
    for signal_key in iter(parameters['compute']):
        if parameters['compute'][signal_key]:
            wa_write(signal_key+'.wav', parameters['fs'] , np.array([np.int16(val) for val in iter((tukey_win(len(signals[signal_key]),.1)*signals[signal_key]/np.max(signals[signal_key]))*(2**15-1))]) );
            col2.text(signal_key)
            col2.audio(signal_key+'.wav', format='wav' );

st.sidebar.header('Circular trajectory moving source at high velocities')

option = st.sidebar.selectbox("Computed signals",parameters['list_options'],index=3);
 # pour changer le nombre de graphes dessinés à l'initialisation, changer l'index de la fonction st.sidebar.selectbox
parameters['freq_rotation'] = st.sidebar.slider('freq_rotation', min_value=0., max_value=200., value=parameters['freq_rotation'], step=0.1);
parameters['R_motion'] = st.sidebar.slider('R_motion', min_value=0., max_value=10., value=parameters['R_motion'], step=0.1);
parameters['freq_emission'] = st.sidebar.slider('freq_emission', min_value=0., max_value=200., value=parameters['freq_emission'], step=0.1);
parameters['R_receiver'] = st.sidebar.slider('R_receiver', min_value=0., max_value=10., value=parameters['R_receiver'], step=0.1);
parameters['theta_receiver'] = st.sidebar.slider('theta_receiver', min_value=0., max_value=2*np.pi, value=parameters['theta_receiver'], step=0.1);
for i_key,key in enumerate(parameters['list_options']):
    parameters['compute'].update({key:np.array([True if i_index <=parameters['list_options'].index(option) else False for i_index in range(len(parameters['list_options']))])[i_key] });

scheme_plot(parameters);
signals = signals_compute(parameters);
signals_plot(parameters,signals);
Leq_plot(parameters, signals);
signals = fft_compute(parameters,signals);
fft_plot(parameters,signals);
produce_audio(parameters,signals);



#for key in iter(glob.glob('./figures/*30cm.png')):
st.markdown("""---""")
st.subheader('Spectrograms resulted from sources moving at an increasing speed')
im = Image.open(os.path.join(os.path.join(os.getcwd(),'figures'),'Spectro-emission-R30cm.png'))
st.image(im)
im0 = Image.open(os.path.join(os.path.join(os.getcwd(),'figures'),'Spectro-phased_emis-R30cm.png'))
st.image(im0)
im1 = Image.open(os.path.join(os.path.join(os.getcwd(),'figures'),'Spectro-propag_phased_emis-R30cm.png'))
st.image(im1)
im2 = Image.open(os.path.join(os.path.join(os.getcwd(),'figures'),'Spectro-doppler_propag_phased_emis-R30cm.png'))
st.image(im2)
