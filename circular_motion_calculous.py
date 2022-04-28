# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 12:31:55 2022

@author: C7881534
"""

import streamlit as st
import numpy as np
from scipy import signal
from pylab import plt

fs= 44100 #Hz
signal_length= 5. #s
rho = 415. # acoustical impedance at T=20°C
c= 343. # sound celerity at T=20°C

#Emission signal parameters
A = 1.
freq_emission = 100. # Hz

#Motion parameters
Rs= 2.5 #m
freq_rotation = 99 # rot per sec ~Hz 

# reception parameter
thetal=1/4*2*np.pi
Rl = .5#m

t = np.arange(0,signal_length,1./fs)

# effect of time-dependent phase on the initial static signal
emission = np.array([A/(4*np.pi*rho * Rs) * np.sin(2*np.pi*freq_emission*t_i) for t_i in iter(t)])
phase = 2*np.pi/c * np.sqrt(Rs**2 + Rl**2 + Rs*Rl*np.cos(2*np.pi*freq_rotation*t))+thetal
phased_emission = np.array([A/(4*np.pi*rho * Rs) * np.sin(2*np.pi*freq_emission*t_i+phase[i_t]) for i_t,t_i in enumerate(t)])
propag_phased_emission = np.array([phased_emission[i_t]*(4*np.pi*rho*Rs) /(4*np.pi*rho * np.sqrt((Rs**2+Rl**2)))*np.sqrt((1-2*(Rs*Rl)/(Rs**2+Rl**2)*np.sin(2*np.pi*freq_rotation*t_i))) for i_t,t_i in enumerate(t)])
doppler_propag_phased_emission = np.array([propag_phased_emission[i_t]/(1+2*np.pi*freq_rotation*Rl/c*np.sin(2*np.pi*freq_rotation*t_i)) for i_t,t_i in enumerate(t)])

plt.plot(t,emission, alpha=0.5,label = 'emission')
plt.plot(t,phased_emission,alpha=0.5,label = 'phased emission')
plt.plot(t,propag_phased_emission,alpha=0.5,label = 'propagated phased emission')
plt.plot(t,doppler_propag_phased_emission,alpha=0.3,label = 'Dopplerized propagated phased emission')

plt.axis([t[0],t[-1],-2e-4,2e-4])
plt.show()

sp = np.fft.rfft(emission)
phased_sp = np.fft.rfft(phased_emission)
propag_phased_sp = np.fft.rfft(propag_phased_emission)
doppler_propag_phased_sp = np.fft.rfft(doppler_propag_phased_emission)

freq = np.fft.rfftfreq(t.shape[-1],1/fs)

fig= plt.loglog(freq,np.abs(sp),alpha=0.8,label='emission')
plt.loglog(freq, np.abs(phased_sp),alpha=0.8,label='phased emission')
plt.loglog(freq, np.abs(propag_phased_sp),alpha=0.4,label='propag. phased emission')
plt.loglog(freq, np.abs(doppler_propag_phased_sp),alpha=0.3,label='propag. phased emission with Doppler')

plt.legend(loc=4)
plt.grid()
plt.axis([freq[1],freq[-1],1e-24,1e1])
st.pyplot(plt)

