# -*- coding: utf-8 -*-
"""
Created on Mon May  2 15:04:42 2022

@author: C7881534
"""

import streamlit as st

import wave, struct, random
sampleRate = 44100.0 # hertz
#duration = 1.0 # seconds
#frequency = 440.0 # hertz
obj = wave.open('sound.wav','w')
obj.setnchannels(1) # mono
obj.setsampwidth(2)
obj.setframerate(sampleRate)
for i in range(99999):
   value = random.randint(-32767, 32767)
   data = struct.pack('<h', value)
   obj.writeframesraw( data )
obj.close()

st.header('Test Cédric')
st.subheader('coucou')
st.audio('sound.wav',format='wav')