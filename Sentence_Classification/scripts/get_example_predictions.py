# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 10:14:33 2019

Simple script, to filter the predictions file and save results to csv - this was used to quickly obtain examples described in the tables
@author: Lena Schmidt
"""
import pandas as pd

pathLabels='\\\\smbhome.uscs.susx.ac.uk\\ls612\\Documents\\Dissertation\\Data\\BERT\\dataframes\\dfTESTjin.df'



pathBase='\\\\smbhome.uscs.susx.ac.uk\\ls612\\Documents\\Dissertation\\Data\\BERT\\Results\\JinBase\\predictions.df'

###chose preds and labels to compare
pred = pd.read_pickle(pathBase)
pred= pred.rename(columns={"P": "p_pred", "I": "i_pred", "O": "o_pred", "A": "a_pred", "M": "m_pred", "R": "r_pred", "C": "c_pred"})#rename before concat, since columns are the same
lab = pd.read_pickle(pathLabels) 

combi = pd.concat([pred, lab], axis=1)

print(pred.head())
print(lab.head())
print(combi.head())

#examples = combi.loc[(combi['P']==1)& (combi['p_pred']>=0.2) & (combi['i_pred']>=0.2)]#examples where P was labelled, and P plus I was predicted
#PI_bert=examples.to_csv('PI_bert.csv')

#examples = combi.loc[(combi['R']==1)& (combi['o_pred']>=0.3)]#where r was labelled but O predicted
#PI_bert=examples.to_csv('OnotR_bert.csv')

#examples = combi.loc[(combi['M']==1)& (combi['m_pred']<=0.3)]#where m was labelled but not predicted 
#PI_bert=examples.to_csv('notM_bert.csv')

examples = combi.loc[(combi['I']==1)& (combi['i_pred']<=0.2)]#where i was labelled but not predicted confidently
PI_bert=examples.to_csv('labInotPred_bert.csv')

print(examples.head())
