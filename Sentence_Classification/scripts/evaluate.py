# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 13:14:51 2019

@author: ls612
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import confusion_matrix
import sklearn.metrics as skm
from sklearn.metrics import f1_score
import numpy as np

#########################################datasets predicted by bert models


multilangLabels='dfTESTjin.df'#the dataframe given to the model for training
multilangPred='predictions.df'#model's prediction, automatically saved after training to the G drive

def conMatrix(label, pred, mode = 'single'):#get matrix, works when predicting single class per example
    LABEL_COLUMNS = ['P','I','O', 'A','M','R','C']
    lab = label.drop(columns=['id','comment_text'])#get labels array as one hot
    
    if mode == 'single':
        
        #######concerning the gold standard labels
        print(lab.head())
        lab = lab.values
        labels=[]#will contain labels as list --> [2,1,1,6,4,0, ..]. They originally come as one hot vector per label [0,0,1,0,0,0,0], so need to be converted
        
        for row in lab:
            for i, v in enumerate(row):
                if v != 0:
                    labels.append(i)#use index of first nr found, it corresponds with the right label
                    
                  
        ####concerning the predictions   #make predictions onehot by using only most confident prediction: argmax
        pre = np.zeros_like(pred)#new array, filled with zeros 
        pre[np.arange(len(pred)), pred.values.argmax(1)] = 1#set a one just in the place where the highest probability sits.
        
        
        predictions=[]#get labels as list, same procedure as with gold standard labels
        for row in pre:
            for i, v in enumerate(row):
                if v != 0:
                    predictions.append(i)
        
        #print(labels.shape)
        #print(len(predictions.shape))

        ################make confusion matrix and plot heatmap from it            
        conf_mat = confusion_matrix(labels, predictions)
        fig, ax = plt.subplots(figsize=(10,10))
        sns.heatmap(conf_mat, annot=True, fmt='d',
                    xticklabels=LABEL_COLUMNS, yticklabels=LABEL_COLUMNS)
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.title('Scibert finetuned')
        plt.show()
        print(skm.classification_report(labels, predictions))
        
    else:####explore how multiclass classification influences scores
        lab = lab.values
        unique, counts = np.unique(lab, return_counts=True)
        print(dict(zip(unique, counts)))
        
        #################################################################plot characteristics of prediction: how sure/ unsure is the model with its probabilities.
        pr = np.around(pred.values, decimals=6, out=None)
        unique, counts = np.unique(pr, return_counts=True)
        freqs = dict(zip(unique, counts))
        
        vals=[]
        for key in sorted(freqs.keys()):#sort by key(probability) and add values to list in correct order for plotting
            vals.append(freqs[key])
        
        #plt.yscale('log')#log scale makes it easier to see middle part, 
        plt.plot(sorted(freqs.keys()),vals)
        plt.ylabel('Number of occurrences')
        plt.xlabel('Predicted probability')
        plt.title('Characteristics of BERT predictions')
        plt.savefig('charBERTpred.png')
        plt.show()
        
        
        pre = pred.apply(lambda x: [1 if y >= 0.4 else 0 for y in x])#just to print predictions for a threshold defined in this function
        
        pre = pre.values
        
        conf_mat = multilabel_confusion_matrix(lab, pre)
        
        print('Threshold: {}'.format(0.3))
        print( skm.classification_report(lab, pre))
        ####################################################################metrics curve, get scores for even linspace of thresholds
        thresholds = np.linspace(0.01, 0.99, 50)
        p={'pr':[],'re':[],'f1':[]}#these lists will store precision etc scores for all 50 thresholds, to be plotted
        i={'pr':[],'re':[],'f1':[]}
        o={'pr':[],'re':[],'f1':[]}
        
        
        for thresh in thresholds:#get daata for each classification threshold
             pre = pred.apply(lambda x: [1 if y >= thresh else 0 for y in x])#assign a 1 (positive prediction) if probability exceeds this particular threshold, otherwise assign 0
             pre = pre.values#get values from dataframe
             
             #print('{} {}'.format(pre[12338], thresh))#print classification example at this threshold
             #print(pre.shape)
             cr=skm.classification_report(lab, pre, output_dict=True)#get classification report for these predictions, and output dict.
             
             p['pr'].append(cr['0']['precision'])#add metrics. there might be a more elegant way to do this...
             p['re'].append(cr['0']['recall'])
             p['f1'].append(cr['0']['f1-score'])
             
             i['pr'].append(cr['1']['precision'])
             i['re'].append(cr['1']['recall'])
             i['f1'].append(cr['1']['f1-score'])
             
             o['pr'].append(cr['2']['precision'])
             o['re'].append(cr['2']['recall'])
             o['f1'].append(cr['2']['f1-score'])
             
             #i.append([cr['1']['precision'],cr['1']['recall'], cr['1']['f1-score']])
            # o.append([cr['2']['precision'],cr['2']['recall'], cr['2']['f1-score']])
        
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3)#####plotting of the threee labels of interest
        fig.suptitle('Effect of threshold on metrics')
        fig.set_figheight(5)
        fig.set_figwidth(19)
        
        ax1.set_title('Performance at P label')
        ax1.plot(thresholds, p['pr'],  '-', label='Precision', linewidth=1)
        ax1.plot(thresholds, p['re'],  '-', label='Recall', linewidth=1)
        ax1.plot(thresholds, p['f1'],  '-', label='F1', linewidth=1)
        
        ax2.set_title('Performance at I label')
        ax2.plot(thresholds, i['pr'],  '-', label='Precision', linewidth=1)
        ax2.plot(thresholds, i['re'],  '-', label='Recall', linewidth=1)
        ax2.plot(thresholds, i['f1'],  '-', label='F1', linewidth=1)
        
        ax3.set_title('Performance at O label')
        ax3.plot(thresholds, o['pr'],  '-', label='Precision', linewidth=1)
        ax3.plot(thresholds, o['re'],  '-', label='Recall', linewidth=1)
        ax3.plot(thresholds, o['f1'],  '-', label='F1', linewidth=1)
        
        ax1.legend()
        fig.text(0.5, 0.04, 'Probability threshold for class assignment', ha='center')
        fig.text(0.08, 0.5, 'Score', va='center', rotation='vertical')
        plt.show()    
#########################################choose which prediction to look at             
#pred = pd.read_pickle(pathBase)   
pred = pd.read_pickle(multilangPred)
lab = pd.read_pickle(multilangLabels)    

print(len(pred.index))
print(len(lab.index))


conMatrix(lab, pred, mode ='multi')#do single class prediction with heatmap result (mode ='single'), or multiclass to get threshold graph. Mode= single will automatically only assign strongest class, while ulti mode assigns classes depending on each tested threshold (spacing can be changed in code)
