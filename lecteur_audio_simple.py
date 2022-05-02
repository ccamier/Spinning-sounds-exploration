# -*- coding: utf-8 -*-
"""
Created on Thu Jul  1 12:37:48 2021

@author: C7881534
"""
import streamlit as st
import scipy as sc
st.header('Test Cédric')
text_en = "this article is non compliant"
audio_file = open('C:/Users/C7881534\Documents/Sons/Selection_son/Selection_son/Maelys_prevert_mono.wav', 'rb')
audio_bytes = audio_file.read()
st.audio(audio_bytes, format='audio/ogg')
