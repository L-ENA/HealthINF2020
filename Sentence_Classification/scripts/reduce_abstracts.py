# -*- coding: utf-8 -*-
"""
Created on Sun Jul 28 18:15:38 2019
Methods to delete non-complete abstracts (abstracts that do not have P and I and O tags)
    2. to save them is format that can be used to train the sentence classification
@author: lena schmidt
"""

pathOrigTrain='PICO_train.txt'
pathOrigDev='PICO_dev.txt'
pathOrigTest='PICO_test.txt'

def originalDF(doc1,  ids,path,checkCompleteness=False):#gets the original tagged abstract sentences and outputs them as df that can be used for BERT fine tuning
    d1 = codecs.open(doc1,encoding='utf-8', mode='r')
    linesd1= d1.readlines()
    d1.close()#orig
    
      
    #######################################original data: filter out the abstracts that are pico complete
    key=''
    allAbs={}
    p=False
    i=False
    o=False
    
    delCount = 0
    delAbs = []
    for line in linesd1:
        if '###' in line:
            if key != '' and checkCompleteness:
                
                if p==False or i == False or o == False:#to sort out the previous abstract if it is not complete are not complete
                    #print(key) ##see the ones that were excluded
                    #print('---')
                    delAbs.append(allAbs[key])
                    del allAbs[key]
                    delCount+=1
                
            key=line.strip()#to prevent overlaps if id in new and original dataset are identical###create new key
            p=False
            i=False
            o=False
           
                
            allAbs[key]= []#make empty dict entry for this new annotated abstract
        
    
        else:
            
            if key != '':
                ######################OPTIONAL: change uninteresting tags to G tag
#                if re.search(r'\|(A|M|R|C)\|', line):#finds tags to change
#                    line = re.sub(r'(\w+)(\|.\|)', r'\1|G|', line)
#                    if random.random() <=0.33:
#                        allAbs[key].append(line)
                #else:    #append anywaz
                #print(line)
                if checkCompleteness:
                    if re.search(r'\|[P]\|', line):
                        p=True
                    elif re.search(r'\|[I]\|', line):
                        i=True
                    elif re.search(r'\|[O]\|', line):
                        o=True
                        
                allAbs[key].append(line)
                
    rows=[]
    oip=0
    ip=0
    op=0
    oi=0
    empties=0
    MODE='else'
    ########################################################make one hot label vectors
    print('Assigning tags...')
    if MODE=='allTags':############make files that contain all possible tags, including methods, results etc
        for key, item in allAbs.items():
            counter=0#to add nr to ID
            
            for line in item:
                line = re.sub(r'\n', '', line)#get rid of unncecessary linebreaks
                if re.search(r'\|[P]\|', line):
                    line = re.sub(r'(.+)(\|.\|)', '', line)#delete the tag
                    #df.loc[len(df)] = [key+str(counter)+ids,line,1,0,1,0,0,0,0]####performance of loc is horrible here, takes wa to long for train dataframe. write to csv instead
                    rows.append([key+str(counter)+ids,line,1,0,0,0,0,0,0])#add p entry
                elif re.search(r'\|[I]\|', line):
                    line = re.sub(r'(.+)(\|.\|)', '', line)#delete the tag
                    rows.append([key+str(counter)+ids,line,0,1,0,0,0,0,0])
                elif re.search(r'\|[O]\|', line):
                    line = re.sub(r'(.+)(\|.\|)', '', line)#delete the tag
                    rows.append([key+str(counter)+ids,line,0,0,1,0,0,0,0])
                    
                elif re.search(r'\|OIP\|', line):
                    line = re.sub(r'(.+)(\|.+\|)', '', line)#delete the tag
                    rows.append([key+str(counter)+ids,line,1,1,1,0,0,0,0])
                    oip += 1#count how many occurrences we have for the schiz special multitags
                elif re.search(r'\|OI\|', line):
                    line = re.sub(r'(.+)(\|.+\|)', '', line)#delete the tag
                    rows.append([key+str(counter)+ids,line,0,1,1,0,0,0,0])
                    oi += 1
                elif re.search(r'\|IP\|', line):
                    line = re.sub(r'(.+)(\|.+\|)', '', line)#delete the tag
                    rows.append([key+str(counter)+ids,line,1,1,0,0,0,0,0]) 
                    ip += 1
                elif re.search(r'\|OP\|', line):
                    line = re.sub(r'(.+)(\|.+\|)', '', line)#delete the tag
                    rows.append([key+str(counter)+ids,line,1,0,1,0,0,0,0])
                    op += 1
                    
                elif re.search(r'\|[A]\|', line):
                    line = re.sub(r'(.+)(\|.\|)', '', line)#delete the tag
                    rows.append([key+str(counter)+ids,line,0,0,0,1,0,0,0]) 
                elif re.search(r'\|[M]\|', line):
                    line = re.sub(r'(.+)(\|.\|)', '', line)#delete the tag
                    rows.append([key+str(counter)+ids,line,0,0,0,0,1,0,0]) 
                elif re.search(r'\|[R]\|', line):
                    line = re.sub(r'(.+)(\|.\|)', '', line)#delete the tag
                    rows.append([key+str(counter)+ids,line,0,0,0,0,0,1,0])
                elif re.search(r'\|[C]\|', line):
                    line = re.sub(r'(.+)(\|.\|)', '', line)#delete the tag
                    rows.append([key+str(counter)+ids,line,0,0,0,0,0,0,1])
                
                else:
                    if line.strip() != '':#if line has content
                        #print(line)
                        line = re.sub(r'(.+)(\|.+\|)', '', line)#delete the tag
                        rows.append([key+str(counter)+ids,line,0,0,0,0,0,0,0])#all zeros, for schiz abstracts that did not end up with an annotation anywhere
                        empties += 1
                counter += 1   
        df = pd.DataFrame(rows, columns=['id', 'comment_text','P','I','O', 'A','M','R','C'])#list of tagging options
            
    else:####make just PIO annotations, and optionally sample the unannotated
        for key, item in allAbs.items():
            counter=0#to add nr to ID
            
            for line in item:
                line = re.sub(r'\n', '', line)#get rid of unncecessary linebreaks
                if re.search(r'\|[P]\|', line):
                    line = re.sub(r'(.+)(\|.\|)', '', line)#delete the tag
                    #df.loc[len(df)] = [key+str(counter)+ids,line,1,0,1,0,0,0,0]####performance of loc is horrible here, takes wa to long for train dataframe. write to csv instead
                    rows.append([key+str(counter)+ids,line,1,0,0,0])#add p entry
                elif re.search(r'\|[I]\|', line):
                    line = re.sub(r'(.+)(\|.\|)', '', line)#delete the tag
                    rows.append([key+str(counter)+ids,line,0,1,0,0])
                elif re.search(r'\|[O]\|', line):
                    line = re.sub(r'(.+)(\|.\|)', '', line)#delete the tag
                    rows.append([key+str(counter)+ids,line,0,0,1,0])
                elif re.search(r'\|OIP\|', line):
                    line = re.sub(r'(.+)(\|.+\|)', '', line)#delete the tag
                    rows.append([key+str(counter)+ids,line,1,1,1,0])
                    oip += 1
                elif re.search(r'\|OI\|', line):
                    line = re.sub(r'(.+)(\|.+\|)', '', line)#delete the tag
                    rows.append([key+str(counter)+ids,line,0,1,1,0])
                    oi += 1
                elif re.search(r'\|IP\|', line):
                    line = re.sub(r'(.+)(\|.+\|)', '', line)#delete the tag
                    rows.append([key+str(counter)+ids,line,1,1,0,0]) 
                    ip += 1
                elif re.search(r'\|OP\|', line):
                    line = re.sub(r'(.+)(\|.+\|)', '', line)#delete the tag
                    rows.append([key+str(counter)+ids,line,1,0,1,0])
                    op += 1
                elif random.random() > 0.5:     
                    if re.search(r'\|[A]\|', line):
                        line = re.sub(r'(.+)(\|.\|)', '', line)#delete the tag
                        rows.append([key+str(counter)+ids,line,0,0,0,1]) 
                    elif re.search(r'\|[M]\|', line):
                        line = re.sub(r'(.+)(\|.\|)', '', line)#delete the tag
                        rows.append([key+str(counter)+ids,line,0,0,0,1]) 
                    elif re.search(r'\|[R]\|', line):
                        line = re.sub(r'(.+)(\|.\|)', '', line)#delete the tag
                        rows.append([key+str(counter)+ids,line,0,0,0,1])
                    elif re.search(r'\|[C]\|', line):
                        line = re.sub(r'(.+)(\|.\|)', '', line)#delete the tag
                        rows.append([key+str(counter)+ids,line,0,0,0,1])
                    
                    else:
                        if line.strip() != '':#if line has content
                            #print(line)
                            line = re.sub(r'(.+)(\|.+\|)', '', line)#delete the tag
                            rows.append([key+str(counter)+ids,line,0,0,0,1])#all zeros
                            empties += 1
                    counter += 1   
            
            df = pd.DataFrame(rows, columns=['id', 'comment_text','P','I','O', 'G'])#list of tagging options             
    #df.to_pickle(path+'.df')#save as df
    print(df.head())
    print(len(rows))
    print('Special tags added: {}  {}  {}  {}. empties: {}'.format(oip,ip,op,oi, empties))
    df.to_csv(path+'.csv', index=False)
    print(delCount)
    print(len(allAbs.items()))
    return df
    #############################print excluded abstract, if checking function was active
#    for i in range(10):
#        print()
#        for sent in delAbs[i]:
#            print(sent)
            
#use checkCompleteness param for jin sentences, because they contain some irrelevant abstracts    
#print('dev')            
dfDEVcszg=originalDF(pathSchizDev,'cszg','dfDEVcszg')
dfTESTcszg=originalDF(pathSchizTest,'cszg','dfTESTcszg')    

dfTRAINcszg=originalDF(pathSchizTrain,'cszg','dfTRAINcszg')



dfTESTjin=originalDF(pathOrigTest,'org','csvTESTjin', True)
dfTRAINjin=originalDF(pathOrigTrain,'org','csvTRAINjin', True)
dfDEVjin=originalDF(pathOrigDev,'org','csvDEVjin', True)


dfTESTjin.to_pickle('dfTESTjin.df')
#saveothers if needed
