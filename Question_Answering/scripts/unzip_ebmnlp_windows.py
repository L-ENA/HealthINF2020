# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 13:50:24 2019

Unzipping the ebm-nlp data can take a while on windows, because it contains so many tiny files. 
This script works faster than the normal unzipping process

@author: Lena Schmidt
"""

from zipfile import ZipFile

with ZipFile("C:\Users\xf18155\OneDrive - University of Bristol\Documents\Ebmnlp\ebm_nlp_2_00.zip", 'r') as zipObj:
   # Extract all the contents of zip file in current directory
   zipObj.extractall()
