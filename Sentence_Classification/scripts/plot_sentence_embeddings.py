# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 16:48:40 2019
adapted from https://github.com/hanxiao/bert-as-service/blob/master/example/example7.py

run on windows/lab machine with bert server already started and ready

@author: ls612
"""

import pickle
import time
from collections import namedtuple

import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from bert_serving.client import BertClient
from bert_serving.server import BertServer
from bert_serving.server.helper import get_args_parser
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import euclidean_distances
from matplotlib.lines import Line2D
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import adjusted_rand_score
###########>>>>PART1###############load data into df and 
sents = pickle.load(open("C:\\Users\\xf18155\\pico_examples.p", 'rb'))
#sents=pickle.load(open('ALLpico_examplesCSZG.dict', 'rb'))

s=[]
t=[]
for key, value in sents.items():
    for sent in value:
        s.append(sent)
        t.append(key)
        
data = pd.DataFrame(columns=['SENT', 'TAG'])
data['SENT']=s
data['TAG']=t

maxSents = 1000   
print(data.head())    
shuffled = data.reindex(np.random.permutation(data.index))

p = shuffled[shuffled['TAG'] == 'p'][:maxSents]
i = shuffled[shuffled['TAG'] == 'i'][:maxSents]
o = shuffled[shuffled['TAG'] == 'o'][:maxSents]
concated = pd.concat([p, i, o], ignore_index=True)
concated = concated.reindex(np.random.permutation(concated.index))#shuffle again
concated['LABEL'] = 0

# One-hot encode the lab
concated.loc[concated['TAG'] == 'p', 'LABEL'] = 0
concated.loc[concated['TAG'] == 'i', 'LABEL'] = 1
concated.loc[concated['TAG'] == 'o', 'LABEL'] = 2

subset_text = list(concated['SENT'].values)
subset_label = list(concated['LABEL'].values)
num_label = len(set(subset_label))

##print(subset_label)
#########stats about the texts
print('min_seq_len: %d' % min(len(v.split()) for v in subset_text))
print('max_seq_len: %d' % max(len(v.split()) for v in subset_text))
print('mean_seq_len: %d' % np.mean([len(v.split()) for v in subset_text]))
print('unique label: %d' % num_label)

subset_vec_all_layers = []
layersAdded=[]
concated.to_csv('pioSentsScibert.csv', index=True)
#########################>>>>>PART2##################################################################################################get data from the model: once for every bert configuration



layersAdded.append(['-6'])
#layers=[1,2,3,6,8,10,11,13,15,18,21,24]

#6.51 on 1 worker
#4.09 on 2 workers
#3.41 on 3 workers
print('start server')
bc = BertClient()#####sadly, starting bert server from python script did not work in a loop *(gets stuck when exporting graph), so has to be started manually in console and then run this part of code only, other code commented 
subset_vec_all_layers.append(bc.encode(subset_text))

bc.close()
print('done')
print(len(subset_vec_all_layers))
print(layersAdded)
######################################################################################################################execute to save embedding data
####save bert vectors and labels
stacked_subset_vec_all_layers = np.stack(subset_vec_all_layers)
np.save('bertbase',stacked_subset_vec_all_layers)
np_subset_label = np.array(subset_label)
np.save('bertbase_subset_label',np_subset_label)

#

######################>>>>>>PART3###################################################################################################Visualisation
#load bert vectors and labels
subset_vec_all_layers = np.load('BERTlarge.npy')
np_subset_label = np.load('BERTlarge_subset_label.npy')
subset_label = np_subset_label.tolist()


###############PLOTTING################
layersAdded=['-3 to -1','-8 to -6']
def compareDist(pdat, centroidsP, centroidsI, centroidsO, names):
    distsSelf=[]
    distsN1=[]
    distsN2=[]
        
    for x in range(len(pdat.index)):
        comp = pdat.iloc[x].values.reshape(1, -1)
        euclidian= euclidean_distances(centroidsP, comp)
        distsSelf.append(euclidian[0][0])
        euclidian= euclidean_distances(centroidsI, comp)
        distsN1.append(euclidian[0][0])
        euclidian= euclidean_distances(centroidsO, comp)
        distsN2.append(euclidian[0][0])
        
    #print('Distances within cluster: {}'.format(np.mean(distsSelf)))
    #print('Distances to {} cluster: {}. {} cluster: {}'.format(names[0], np.mean(distsN1),names[1], np.mean(distsN2)))
    return np.mean(distsSelf), np.mean(distsN1), np.mean(distsN2)
        
def analyseClusters(layers, labels):
    metrics=[]
    rands=[]
    for layer in layers:
        print(layer.shape)#how many dims
        data = pd.DataFrame(layer)#assign array and labels
        data['LAB']=subset_label
        pdat=data.loc[data['LAB'] == 0]#filter data by label
        idat=data.loc[data['LAB'] == 1]
        odat=data.loc[data['LAB'] == 2]
        pdat = pdat.drop(['LAB'], axis=1)
        idat = idat.drop(['LAB'], axis=1)
        odat = odat.drop(['LAB'], axis=1)
        
        data = data.drop(['LAB'], axis=1)
        ####################################calculate cluster stats
        kmeansP = KMeans(n_clusters=1, random_state=5,  max_iter=600).fit(pdat)####################################################plot centroids for each class
        centroidsP = kmeansP.cluster_centers_###P
        
        
        kmeansI = KMeans(n_clusters=1, random_state=5,  max_iter=600).fit(idat)###I centroid
        centroidsI = kmeansI.cluster_centers_
        
        kmeansO = KMeans(n_clusters=1, random_state=5,  max_iter=600).fit(odat)###O

        centroidsO = kmeansO.cluster_centers_
        
        kmeansAll = KMeans(n_clusters=3, random_state=5,  max_iter=1200).fit(data)####################################################plot centroids for each class
        centroidsAll = kmeansAll.cluster_centers_###all
        rands.append(adjusted_rand_score(subset_label, list(kmeansAll.labels_)))
        
        
        pSelf, pN1, pN2 = compareDist(pdat, centroidsP, centroidsI, centroidsO, ['I','O'])######todo add other cluster distances#get cluster data  for P
        iSelf, iN1, iN2 = compareDist(idat, centroidsI, centroidsP, centroidsO, ['P','O'])
        oSelf, oN1, oN2 = compareDist(odat, centroidsO, centroidsP, centroidsI, ['P','I'])
        
        metrics.append({'P': [pSelf, pN1, pN2], 'I': [iSelf, iN1, iN2], 'O': [oSelf, oN1, oN2]})#append data to return dict
    return rands, metrics
    
    
rands, metrics = analyseClusters(subset_vec_all_layers, np_subset_label)
#    
def vis(embed,subset_label, rands,  vis_alg='PCA', pool_alg='REDUCE_MEAN', num_label=3):
    
    #layers=[1,2,3,6,8,10,11,13,15,18,21,24]
    layers=layersAdded
    plots=list(range(1,len(layers)+1))
    plt.close()
    fig = plt.figure()
    data={}
    plt.figure(figsize=(6, 6))
    for idx, ebd, subpl, title, rand in zip(layers, embed, plots, layersAdded, rands):
        ax = plt.subplot(1, 1, subpl)
        vis_x = ebd[:, 0]
        vis_y = ebd[:, 1]
        #plt.yticks([])
        #plt.xticks([])
        
        plt.scatter(vis_x, vis_y, c=subset_label, cmap=ListedColormap(["blue", "green", "red"]), marker='.',
                    alpha=0.7, s=2)
        
        ##########################################################################################################################plot cluster centres
        
        concated = pd.DataFrame()#assign array and labels
        concated['LAB']=subset_label#get label for each row, and most strongest dimensions
        
        concated['X']=vis_x
        concated['Y']=vis_y
        concated['LAB']=subset_label
        #print(concated.head())
        pdat=concated.loc[concated['LAB'] == 0][['X', 'Y']]#filter data by label
        idat=concated.loc[concated['LAB'] == 1][['X', 'Y']]
        odat=concated.loc[concated['LAB'] == 2][['X', 'Y']]
        
        kmeansAll = KMeans(n_clusters=3, random_state=5,  max_iter=600).fit(concated)####################################################plot centroids for each class
        centroidsAll = kmeansAll.cluster_centers_###all
        score = adjusted_rand_score(subset_label, list(kmeansAll.labels_))
        #
        print(score)
        
        kmeans = KMeans(n_clusters=1, random_state=5,  max_iter=600).fit(pdat)####################################################plot centroids for each class
        centroidsP = kmeans.cluster_centers_###P
        plt.plot(centroidsP[:, 0], centroidsP[:, 1], marker='D', markersize=10, color="blue")
        
        kmeans = KMeans(n_clusters=1, random_state=5,  max_iter=600).fit(idat)###I centroid
        centroidsI = kmeans.cluster_centers_
        plt.plot(centroidsI[:, 0], centroidsI[:, 1], marker='D', markersize=10, color="green")
        
        kmeans = KMeans(n_clusters=1, random_state=5,  max_iter=600).fit(odat)###O
        centroidsO = kmeans.cluster_centers_
        plt.plot(centroidsO[:, 0], centroidsO[:, 1], marker='D', markersize=10, color="red")
        
        #PO_dist= int(euclidean_distances(centroidsP, centroidsO)[0][0])#####optional: to visualise distance...which layer gets biggest distance between classes..
        #po = plt.plot([centroidsP[:, 0], centroidsO[:, 0]], [centroidsP[:, 1], centroidsO[:, 1]], '--k',alpha=0.5)
        if subpl==1:
            legend_elements = [#Line2D([0], [0],linestyle='--', color='k', lw=1, label='P-O distance = {}'.format(PO_dist)),
                              Line2D(range(1), range(1), color="white",label='Outcome', marker='o', markerfacecolor="red", markersize=12),
                              Line2D([0], [0], marker='o', color='white', label='Intervention', markerfacecolor='g', markersize=12),
                              Line2D([0], [0], marker='o', color="white", label='Population', markerfacecolor='b', markersize=12),
                              Line2D([0], [0],linestyle=' ', color='k', lw=1, label='Rand (before tsne) = {}'.format(float('%.2f'%(rand)))),#attach rand scores and truncate them
                              Line2D([0], [0],linestyle=' ', color='k', lw=1, label='Rand (after tsne) = {}'.format(float('%.2f'%(score))))
                              
                      ]
                      
            ax.legend(handles=legend_elements, loc='lower left', prop={'size': 12},framealpha = 0.5)
        else:
            legend_elements = [#Line2D([0], [0],linestyle='--', color='k', lw=1, label='P-O distance = {}'.format(PO_dist)),
                              Line2D([0], [0],linestyle=' ', color='k', lw=1, label='Rand (before tsne) = {}'.format(float('%.2f'%(rand)))),#attach rand scores and truncate them
                              Line2D([0], [0],linestyle=' ', color='k', lw=1, label='Rand (after tsne) = {}'.format(float('%.2f'%(score))))
                              
                      ]
                      
            ax.legend(handles=legend_elements, loc='lower left', prop={'size': 16})

        
        
        ax.set_title('Bert-base-uncased, pool_layer=%s' % (str(title)), fontsize=12)
####################################################################get outliers or points close to centroids as example and saves them in spreadsheet        
        
        if idx == '-3 to -1':###save example sentences of any layer as examples for outliers/points close to centroid, eg. if x value is less than -50 in this layer (idx 6 is layer -6)....
            print('looking for outliers')
            #outliers = concated.loc[(concated['X']<=28)& (concated['X']>=23) & (concated['Y']<=-20) & (concated['LAB']==0)]##print df entries for some points, to look them up and see what sents they are
            outliers = concated.loc[(concated['LAB']==0)]#save all P and their positions
            outliers = outliers[['X', 'Y']]
            print(outliers.head())
            outliers.to_csv('/content/drive/My Drive/bertLayers/outliers.csv', index=True)
        ########################################################################################################
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1, right=0.95, top=0.9)
    cax = plt.axes([0.96, 0.1, 0.01, 0.3])
    cbar = plt.colorbar(cax=cax, ticks=range(num_label))
    cbar.ax.get_yaxis().set_ticks([])
    for j, lab in enumerate(['P', 'I', 'O']):
        cbar.ax.text(.5, (2 * j + 1) / 3.0, lab, ha='center', va='center', rotation=270)
    fig.suptitle('%s visualization of BERT layers using "bert-as-service" (-pool_strategy=%s)' % (vis_alg, pool_alg),
                 fontsize=14)
    
    #plt.yticks([])
    #plt.xticks([])
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig('/content/drive/My Drive/bertLayers/bertbase_annotate.png', bbox_inches='tight', dpi=400)
    plt.show()


#pca_embed = [PCA(n_components=2).fit_transform(v) for v in subset_vec_all_layers]#use different dimensionality reduction
#vis(pca_embed)

########################################################################################################################REDUCE DIMS AND PLOT
tsne_model_en_2d = TSNE(perplexity=12, n_components=2, init='pca', n_iter=1000, random_state=32)
tsne_embed = [tsne_model_en_2d.fit_transform(v) for v in subset_vec_all_layers]###tsne doing list comprehension
print(tsne_embed)
vis(tsne_embed,subset_label, rands, 't-SNE')
