# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 11:09:13 2019

@author: Lena Schmidt

Script to convert ebm-nlp to squad format, and to mix it with existing squad domains. Change file paths, decide if you want to make training or testing data, and define the questions for each pico type before running this script.
"""

import json
import pandas as pd
import re
from glob import glob
import random
import os
from random import shuffle
import codecs 
from tqdm import tqdm

def span2text(start, end, idDict, context):
                    #gives answer text based on spans of text, but checks that answer does not cross sentence boundaries
                    
                    ans = [idDict[ind] for ind in range(start, end+1)]
                    cont=context.split()
                    ansSpan=0
                    actual_text=''
                    sents=[]
                    for ind, text in enumerate(cont):
                        #print(text + ' ' + str(ans))
                        if ans[0] == text:
#                            print('---------')
                            
                            
                            ending=ind + len(ans)-1
                            try:
                                if (cont[ending]) == ans[-1]:
                                    
#                                    ansSpan = context.index(actual_text)
                                    actual_text=' '.join(ans)
                                    index_start=context.index(actual_text)#test, to see if index is found
                                    break
                                    #print(ansSpan)
                            except:##########this happens if answers were annotated across sentence boundaries. e.g. 'Error with index 8 in ['Iodixanol', 'caused', 'significantly', 'less', 'discomfort', 'than', 'iohexol', '.']: word iohexol span is this long: 3. This is the answer, which includes the same compound name in the next sentence: ['iohexol', '.', 'Iodixanol']'
                                
                                answer=' '.join(ans)##the given answer. might exceed sentence boundaries
                                
                                
                                parts= answer.split('.')#sine sometimes annotations are across sentences
                                sents = [sent.strip() for sent in parts]
                                
                                for candidate in sents:
                                    if candidate in context and candidate !='':
#                                        ansSpan = context.index(candidate.strip())
                                        actual_text = candidate.strip()
                                        
                                        #print('found {} ---- {}'.format(actual_text,context))
                                        break
                                if actual_text == '':
                                    print('not found {} ----\n {}'.format(sents,context))
                                
                    try:
                        return(actual_text, context.index(actual_text))
                    except:
                        
                        print('not found {} ----\n {}'.format(actual_text,context))#should not happen, unless windows/encoding/read errors happen.
                        return(actual_text, 0)
                     
           

makeTest=False#false means that we make training data. true means testing data. difference: paths, and testing data are not sampled, and testing data can have more than 1 entity per sentence

if makeTest:#todo: change paths or have ebm-nlp in working directory. Make sure to give path to correct pico (path to P data entered now)
    span_fnames = glob("ebm_nlp_2_00\\annotations\\aggregated\\hierarchical_labels\\participants\\test\\gold\\*")######for P extra spans
    
    print('Reading test documents...')
    
    
else:    #train data
    span_fnames = glob("ebm_nlp_2_00\\annotations\\aggregated\\starting_spans\\participants\\train\\*")
    print('Reading train documents...')
toks = glob("ebm_nlp_2_00\\documents\\*.tokens")#all token files in a list

tok_fnames=[]
base="ebm_nlp_2_00\\documents\\"

for fname in span_fnames:
    pmid= os.path.splitext(os.path.basename(fname))[0].split('.')[0]#get pubmed id, as in nye github repo
    tok_fnames.append(os.path.join(base, str(pmid)+'.tokens'))

inputs=list(zip(span_fnames, tok_fnames))#list of corresponding tuples of source file and span file



    
    
def addAbsData(spanFile, tokFile, starting_spans=True, spanID=1, undersample_frac = 0.3, pico_type = "P"):
    
    #starting_spans: if the script should use the binary annotations, or the herarchical annotations
    #spanID: the id tht is considered as the positive case. 1 in case of starting_spans, and 1 to x in case of the herarchical labels. see ebm-nlp documentation for more info
    #undersample_frac: this many percent of sentences will be randomly deleted if they dont contain a positive example, e.g. 30% if the fraction is 0.3. Deletions only happen in training data, not in testing data!
    
    
    #in the following you see different variations of pico questions. make sure to use the question that matches your pico datatype (see parameters starting_spas and pico_type) 
   
    
    if starting_spans:#means that we lookat the file that has starting spans (only labels 0 and 1, for each pico type in different folders, defined above)
        if pico_type == "P":
            quests=['Who was treated?','What were the criteria for enrolment?','What were the inclusion criteria?','Who was enrolled in the study?','What were participants diagnosed with?', 'Which participants took part in the study?']
        elif pico_type == "I":
            quests=['Which intervention did the patients receive?','Which intervention did the participants receive?', 'What was the intervention?', 'What did the patients receive?', 'What did the participants receive?', 'What was the intervention given to the participants?']
        elif pico_type == "O":
            quests=['Which outcomes were measured?','What did the study measure?','Which endpoints were measured?', 'What were the measured outcomes?', 'What were the measured endpoints?', 'What were the primary or secondary endpoints?', 'What were the primary or secondary outcomes?']
    
    else:#get the hierarchical labels, pre-programmed here for P labels
        if spanID==4:
            quests = ['Which condition did the participants have?', 'Patients with which medical conditions were included?', 'What was the medical condition?', 'What was the condition?']
        
        ##gender is 2
        if spanID==2:
            quests = ['What sex were the participants?', 'What was the patient\'s sex', 'What gender were the participants?', 'What was the patient\'s gender', 'Were there male or female participants included?']
        
        if spanID==1:
            quests = ['What age were the participants?', 'How old were the participants?', 'What age were the patients?', 'How old were the patients?', 'What was the age in the population?']
        
        if spanID==3:
            quests = ['What was the sample size?','How big was the sample size?','How big was the population','What was the size of the population?', 'How many participants were enrolled?', 'How many partients took part in the study?', 'How many participants took part in the trial?', 'How many partients were enrolled?']
        
    quest = random.sample(quests, k=1)[0]#as this returns list, we take first item of list###random question
    
    
    someI = codecs.open(spanFile, encoding='utf-8').read().split()
    someTok=codecs.open(tokFile, encoding='utf-8').read().split()
    
    #print("-------------------------")print all labels and tokens for debugging
    #print(someI)
    #print("Tokens:")
    #print(someTok)
    
    
    topicString = os.path.splitext(os.path.basename(spanFile))[0].split('.')[0]#pmid as topic
    
    contexts=[]#sent with word index tuples
    text=''#one sent
    
    id_to_sent = {}
    id_to_word = {}
    
    counter=0
    
    end_ids=[]
    for index, word in enumerate(someTok):
        
        
        text = '{} {}'.format(text,word).strip() # restore full sentences, as bert takes whole context sentences and does splitting itself
        id_to_sent[index]=counter#set sentence number for this id    
        id_to_word[index]=word
                
        
        if re.search('^[.!?]$', word):#regex anchors to filter single fullstop or other common end-of-sentence punctuation
            
            contexts.append(text)#get ready for next sent
            text=''
            counter+=1#next sentence
            end_ids.append(index)
            
    domainDict={'title':topicString,'paragraphs':[]}##append many, change this to constructor

       
    starts=[]
    ends=[]
    sents=[]
    
    for ind, value in enumerate(someI):###get start and end span indexes for this pico
        #print(str(ind) + ' ' + str(value) +' ' + someTok[ind])
        
        if int(value) == spanID:#value is string because it is read from txt file
            
            if ind ==0 or int(someI[ind-1]) != spanID:#if we are at start or if previous is differentt, then we have a beginning span
                starts.append(ind)
                
                #print('start appended at ' + str(ind))
            if len(someI)-1 == ind or int(someI[ind+1]) != spanID: #opposite case to above
                ends.append(ind)
                sents.append(id_to_sent[ind])#get sentence number for that span
                #print('end appended at ' + str(ind))
                #print('---')
                
              
    spans = list(zip(starts, ends, sents))#print all identified spans for debugging
    #print('spans:')
    #print(spans)
    
    
    for context in contexts:
        domainDict['paragraphs'].append({'qas':[], 'context':context})
    
             
                    
    to_delete = []#for getting rid of undersampled entries later. 
    
    for index, paragraph in enumerate(domainDict['paragraphs']):
        qaDict={'question':quest, 'id':'{}{}'.format(random.random(), random.random()), 'answers':[], 'is_impossible':False} 
        
        
        for span in spans:
            
            if span[2] == index:#if sentence appears in the span data
                
                txt, spanStart= span2text(span[0], span[1], id_to_word, paragraph['context']) 
                
                if makeTest:#append list of posible answers
                    qaDict['answers'].append({'text': txt, 'answer_start': spanStart})  ### multiple answers
                else:#at training, only one anser can be given per span, so we overwrite and keep latest
                    qaDict['answers']=[{'text': txt, 'answer_start': spanStart}]
#                print('---')

                
        if len(qaDict['answers']) == 0:#no answer was found
            qaDict['is_impossible'] = True
            qaDict['plausible_answers']= [
                {
                  "text": ' '.join(paragraph['context'].split(' ')[:2]),#first 2 words, only for convenience because dataset has no plausible answers
                  "answer_start": 0
                }
              ]
            
            if random.random() < undersample_frac and makeTest == False:#undersampling, but only when we are not in test data mode. test data should represent the reality
                to_delete.append(index)
                
          
        paragraph['qas'].append(qaDict)
        
        
        
        #print(len(paragraph['quas']))     
        #print(qaDict['answers'])

    for index in sorted(to_delete, reverse=True):#do the actual undersampling by deleting the randomly selected sentences that had no answer. reverse sorting to avoid confusion with self-updating indices
        del domainDict['paragraphs'][index]
    #print(domainDict)
    return domainDict
    

#########################get files, loop and create data
versionString="v2.0"#setup the json file dict
data = {'version': versionString, 'data': []}#to have possibility of pure new squad training data



for inp in tqdm(inputs):#loop 
#for inp in inputs:
#    print(inp)
    spf=inp[0]#annotated span comes first in the data
    tkf=inp[1]
    
    
    
    domainDict = addAbsData(spf, tkf, undersample_frac = 0.4)
    data['data'].append(domainDict) 
    

    
  
if makeTest:
    with open('data\\squad\\dev-v2.0.json', encoding='utf-8') as feedsjson:#Original testing data!
        feeds = json.load(feedsjson)
        print(len(feeds['data']))
        try:
            feeds['data']= random.sample(population=feeds['data'],k=0)#take n samples. usually none with testing data becasue evaluating original squad is not objective
        except:
            pass
        
        
        feeds['data'] =feeds['data']+ data['data']#add original plus new data into one dict
        print(len(feeds['data']))
        #shuffle(feeds['data'])  ###no need to shuffle the eval set
        with open('data\\squad\\smallTest.json', mode='w') as f:
            f.write(json.dumps(feeds, indent=2))
else:        

    with open('data\\squad\\train-v2.0.json') as feedsjson:
        feeds = json.load(feedsjson)
        try:
            print(len(feeds['data']))
            feeds['data']= random.sample(population=feeds['data'],k=20)#take n original training domains
        except:
            pass
        
        
        feeds['data'] =feeds['data']+ data['data']#add original and new
        print(len(feeds['data']))
        shuffle(feeds['data'])  #shuffle to avoid having small/big gradients only with same data
        ############training data
        with open('data\\squad\\train_P.json', mode='w') as f:
            f.write(json.dumps(feeds, indent=2))
        
