# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 21:31:18 2018

@author: yingl WANG
"""
import networkx as nx
import networkx.algorithms.isomorphism as iso
import os.path
import numpy as np
import collections as cl
from os.path import join
import random
import math
from math import log, exp, sqrt
from scipy import stats, array, linalg, dot
from collections import deque
import matplotlib.pyplot as plt
from operator import itemgetter
from random import choice


xfont=14
yfont=14
titlefont=14
legendfont=13

var = ['bad_amount', 'expire_amount', 'asset',
       'liability', 'confer', 'loan', 'region', 'risk',
       'market', 'bad_rate', 'expire_rate']
numeric_var = ['asset', 'liability', 'confer', 'loan']
categorical_var = ['region', 'market']
other_var = ['risk', 'bad_rate', 'expire_rate']

tags = []
for y in range(7, 12):
    for m in range(1, 13):
        tags.append('%02d%02d'%(y, m)) 
tags.append('1201')
tags.append('1202')
tags.append('1203')

full_tags = []
for tag in tags:
    full_tags.append('20' + tag)

regions=[11,12,13,14,15,21,22,23,31,32,33,34,35,36,37,41,42,
         43,44,45,46,50,51,52,53,54,61,62,63,64,65,66,71,81,82]
    
LEHMAN_IDX =20 #闆锋浖鍏勫紵鐮翠骇鏈堜唤
FOURTRI_IDX = 22 #鍥涗竾浜块�鍑烘湀浠�
FOURTRI_IDX_END = 47


def get_guarantee_network(tag):
    g = nx.DiGraph()
###### Readlines()[1:] should exculde the first line in the original txt, which is the description of variables###### 
    fin1 = open(r'C:\network\data\edges\edge%s.txt'%tag )

    for line in fin1.readlines()[1:]:
 #       print (line)
        tmp_list = line.split()
        if len(tmp_list) == 4:
            customer, guarantor, balance, num = line.split( )
            g.add_edge(guarantor, customer, balance = float(balance), number = int(num))
    fin1.close()   
####### Add informations of nodes #######
    fin2 = open(r'C:\network\data\nodes\network%s.txt'%tag)
    nodes_has_info =[]
    var_name = ['', 'bad_amount', 'expire_amount', 'asset',
                'liability', 'confer', 'loan', 'region', 'risk',
                'market', 'bad_rate', 'expire_rate']

    for line in fin2.readlines()[1:]:
        x = line.split( )     
        if x[0] in g.nodes():
            nodes_has_info.append(x[0])
            for i in range(1, 12):
                g.node[x[0]][var_name[i]] = float(x[i])
    fin2.close()


###### Caculate the default value of whole GN 
    default_number_node=0
    default_number_edge=0
   
    for n in nodes_has_info:
        if g.node[n]['expire_rate']>0:
            default_number_node +=1
            default_number_edge += g.in_degree(n)
    default_ratio_node = default_number_node / g.number_of_nodes()
    default_ratio_edge = default_number_edge / g.number_of_nodes()

            
#   #填充缺失值
#    node_list=[]
#    lnasset_list=[]
#    lnliability_list=[]
#    lnloan_list=[]
#    
#    for node in nodes_has_info:
#        if g.node [node]['asset']>0:
#           node_list.append(node)
#           lnasset_list.append(np.log(g.node[node]['asset']))
#           lnliability_list.append(np.log(g.node[node]['liability']))
#           lnloan_list.append(np.log(g.node[node]['loan']))
#            ### lnP ~ lnC - alpha lnk
#    x = np.vstack([np.array(lnasset_list), np.ones(len(lnasset_list))]).T
#    (b1, a1) = np.linalg.lstsq(x, np.array(lnliability_list))[0]  
#    (b2, a2) = np.linalg.lstsq(x, np.array(lnloan_list))[0]   
#    #compute the mean and sd of log(asset)
#    narray=np.array(lnasset_list)
#    sum1=narray.sum()
#    narray2=narray*narray
#    sum2=narray2.sum()
#    mean_log_asset=sum1/len(lnasset_list)
#    sd_log_asset=sqrt(sum2/len(lnasset_list)-mean_log_asset**2)   
#    sd = 0.8972
#    tmp = np.random.randn(500000)
#    i = 0
#  
#    #fill the missing value 
#    for node in g.nodes():
#        if node not in nodes_has_info or g.node[node]['asset']==0: ####### Fill in the nodes has no infromation ######
#            log_asset = mean_log_asset + sd_log_asset * tmp[i]
#            g.node[node]['asset'] = np.exp(log_asset)
#            g.node[node]['liability'] = np.exp(a1 + b1 * log_asset)
#            g.node[node]['loan'] = np.exp(a2 + b2 * log_asset + sd * tmp[i])
#            i += 1
#        if node in nodes_has_info and g.node[node]['asset']>0 and g.node[node]['liability']==0:
#           g.node[node]['liability'] = np.exp(a1 + b1 * log(g.node[node]['asset'])) 
#        if node in nodes_has_info and g.node[node]['asset']>0 and g.node[node]['liability']>0 and g.node[node]['loan']==0:
#           g.node[node]['loan'] = np.exp(a2 + b2 * log(g.node[node]['asset']) + sd * tmp[i])
#    del tmp
#        print ('orginal_GN_edges_number:', g.number_of_edges())
    
    return (g, nodes_has_info, default_number_node, default_number_edge, default_ratio_node, default_ratio_edge) 
    


def get_overlap_ratio(g, tag):  ####### GN is a guarantee network ######
    '''
    guarantee_network + shareholding_network
    '''
    GN=g.copy()
    N=g.number_of_nodes()
    M=g.number_of_edges()

    g_overlap= nx.DiGraph()
    edgenum_G_H= 0
    
 ###### Initialized the index of over_lap of all edgs to be 0 ######   
 
    for edge in GN.edges():   
        GN[edge[0]][edge[1]]['over_lap'] = 0   ###### edge is a tuple #####
    
###### Add holder relationship #######
        
    fin3 = open(r'C:\network\data\other relations\holder%s.txt' %tag)
    for line in fin3.readlines()[1:]:
        (customer, holder, time) = line.split()
        if GN.has_edge(holder, customer):
           GN[holder][customer]['over_lap'] = 1
           g_overlap.add_edge(holder, customer)
           edgenum_G_H += 1
           
#        if GN.has_edge(customer, holder):
#           #print(g.has_edge(holder, customer))
#           GN[customer][holder]['over_lap'] = 1
#           g_overlap.add_edge(customer, holder)
#           edgenum_G_H += 1

    fin3.close()   
    
####### Add informations of nodes #######
    fin4 = open(r'C:\network\data\nodes\network%s.txt'%tag)
    overlap_nodes_has_info =[]
    var_name = ['', 'bad_amount', 'expire_amount', 'asset',
                'liability', 'confer', 'loan', 'region', 'risk',
                'market', 'bad_rate', 'expire_rate']

    for line in fin4.readlines()[1:]:
        x = line.split( )     
        if x[0] in g_overlap.nodes():
            overlap_nodes_has_info.append(x[0])
            for i in range(1, 12):
                g_overlap.node[x[0]][var_name[i]] = float(x[i])
    fin4.close()
        

    
    
    #print('g_overlap.number_of_nodes:', g_overlap.number_of_nodes())
    #print('g_overlap.number_of_edges:', g_overlap.number_of_edges())
    #print (edgenum_G_H)
    #print ('over_lap_ratio:', edgenum_G_H/g.number_of_edges())
    #print ('over_lap_ratio:', edgenum_G_H/M)
     
    return (GN, g_overlap,overlap_nodes_has_info, edgenum_G_H)



def default_overlap_network (g_overlap, overlap_nodes_has_info):   #### g should be the overlap network
    N=0 ###### Number of default node 
    M=0 ###### Number of default edge
    for node in g_overlap.nodes():
#        print (g.node[node]['expire_rate'])
        #if g.node[node]['expire_rate']:
        if node in overlap_nodes_has_info:
            #print (g.node[node]['expire_rate'])
            if g_overlap.node[node]['expire_rate']>0:
                #print (g.node[node]['expire_rate'])
                
                N+=1
                M+=g_overlap.in_degree(node)
    default_node_ratio=N/g_overlap.number_of_nodes()
    default_edge_ratio=M/g_overlap.number_of_edges()
    #print('default_node_number_in_overlap_netwrok:', N)
    #print('default_edge_number_in_overlap_netwrok:', M)
 
    #print ('default_node_ratio_in_overlap_network', default_node_ratio) 
    #print('default_edge_ratio_in_overlap_network', default_edge_ratio)
    return (g_overlap, N, M, default_node_ratio, default_edge_ratio)
        

def plot_comparison_GN_overlap_network (tags, xfont, yfont, titlefont, legendfont):
    
###### list of entire guarantee network     
    default_number_node_list_GN =[]
    default_number_edge_list_GN =[]
    default_ratio_node_list_GN =[]
    default_ratio_edge_list_GN =[]
    
###### list of guarantee network with both guarantee and shareholding relationship     
    default_node_number_list_overlap =[]
    default_edge_number_list_overlap=[]
    default_node_ratio_list_overlap =[] 
    default_edge_ratio_list_overlap =[]
    
###### list of guarantee network with guarantee minus shareholding relationship     
    default_number_node_list_G_S =[]
    default_number_edge_list_G_S =[]
    default_ratio_node_list_G_S =[]
    default_ratio_edge_list_G_S =[]


    for tag in tags:
        print (tag)
        (g, nodes_has_info, default_number_node, default_number_edge, default_ratio_node, default_ratio_edge) = get_guarantee_network(tag)
        (GN, g_overlap, overlap_nodes_has_info, edgenum_G_H) =get_overlap_ratio(g, tag)
        (g1, default_node_number, default_edge_number, default_node_ratio, default_edge_ratio)=default_overlap_network (g_overlap, overlap_nodes_has_info)    
        
        
        default_number_node_list_GN.append(default_number_node) 
        default_number_edge_list_GN.append(default_number_edge) 
        default_ratio_node_list_GN.append(default_ratio_node)  
        default_ratio_edge_list_GN.append(default_ratio_edge) 

        default_node_number_list_overlap.append(default_node_number)  
        default_edge_number_list_overlap.append(default_edge_number) 
        default_node_ratio_list_overlap.append(default_node_ratio)  
        default_edge_ratio_list_overlap.append(default_edge_ratio)

        default_number_node_list_G_S.append(default_number_node-default_node_number) 
        default_number_edge_list_G_S.append(default_number_edge-default_edge_number) 
        default_ratio_node_list_G_S.append((default_number_node-default_node_number)/(g.number_of_nodes()-g_overlap.number_of_nodes()))  
        default_ratio_edge_list_G_S.append((default_number_edge-default_edge_number)/(g.number_of_edges()-g_overlap.number_of_edges()))  
     
    print ('default_number_node_list_GN =', default_number_node_list_GN)
    print ('default_number_edge_list_GN =', default_number_edge_list_GN) 
    print ('default_ratio_node_list_GN =', default_ratio_node_list_GN)
    print ('default_ratio_edge_list_GN =', default_ratio_edge_list_GN)
    
    print ('default_node_number_list_overlap =', default_node_number_list_overlap)
    print ('default_edge_number_list_overlap=', default_edge_number_list_overlap)
    print ('default_node_ratio_list_overlap =',  default_node_ratio_list_overlap)
    print ('default_edge_ratio_list_overlap =', default_edge_ratio_list_overlap)
    
    
    print ('default_number_node_list_G_S =', default_number_node_list_G_S)
    print ('default_number_edge_list_G_S =', default_number_edge_list_G_S) 
    print ('default_ratio_node_list_G_S =', default_ratio_node_list_G_S)
    print ('default_ratio_edge_list_G_S =', default_ratio_edge_list_G_S)
###### Plot comparion of default node ratio between entire and overlapped network
    plt.figure(figsize=(14,7))
    N = len(tags)
    x = range(N)
    plt.plot(x,  default_ratio_node_list_GN , 'ko',linewidth=1)
    plt.plot(x,  default_ratio_node_list_G_S , 'bo',linewidth=1)    
    plt.plot(x,  default_node_ratio_list_overlap , 'ro',linewidth=1) 
    
    plt.axvline(x = 3, linewidth=2.2, color='k')
    plt.axvline(x = 20, linewidth=2.5, color='k',linestyle='dashed')
    plt.axvline(x = 22, linewidth=2.2,  color='r')
    plt.axvline(x = 47, linewidth=2.5, color= 'r', linestyle='dashed')
    #plt.legend(loc='upper center', bbox_to_anchor=(0.6,0.95),ncol=3,fancybox=True,shadow=True)
  
    plt.legend( ('Guarantee network', 'Guarantee network with only pure guarantee relationship',  'Over lapped network'),loc='best', fontsize=legendfont,
               fancybox=True,shadow=True)
  
    xtick = np.arange(0, N, 6)
    xtickLabel = []
    for i in xtick:
        xtickLabel.append('20' +tags[i])     
    plt.xticks(xtick, xtickLabel, rotation = 45,fontsize=xfont)
    plt.yticks(fontsize=yfont)
    plt.ylabel('default ratio',fontsize=16)
#    plt.xlabel('0.05% initial ratio')
    plt.title('Comparion of default node ratio between entire and overlapped network', fontsize=titlefont)

    #plt.title('Shandong1',fontsize=titlefont) 
    plt.savefig('Comparion of default node ratio between entire and overlapped network.png')
    plt.show()
    plt.close()

  
    
###### Plot comparion of default edge ratio between entire and overlapped network

    plt.figure(figsize=(14,7))
    N = len(tags)
    x = range(N)
    plt.plot(x,  default_ratio_edge_list_GN , 'ko',linewidth=1) 
    plt.plot(x,  default_ratio_edge_list_G_S , 'bo',linewidth=1)    
    plt.plot(x,  default_edge_ratio_list_overlap , 'ro',linewidth=1) 
    
    plt.axvline(x = 3, linewidth=2.2, color='k')
    plt.axvline(x = 20, linewidth=2.5, color='k',linestyle='dashed')
    plt.axvline(x = 22, linewidth=2.2,  color='r')
    plt.axvline(x = 47, linewidth=2.5, color= 'r', linestyle='dashed')
    #plt.legend(loc='upper center', bbox_to_anchor=(0.6,0.95),ncol=3,fancybox=True,shadow=True)
  
    plt.legend( ('Guarantee network', 'Guarantee network with only pure guarantee relationship',  'Over lapped network'),loc='best', fontsize=legendfont,
               fancybox=True,shadow=True)
  
    xtick = np.arange(0, N, 6)
    xtickLabel = []
    for i in xtick:
        xtickLabel.append('20' +tags[i])     
    plt.xticks(xtick, xtickLabel, rotation = 45, fontsize=xfont)
    plt.yticks(fontsize=yfont)
    plt.ylabel('default ratio',fontsize=16)
#    plt.xlabel('0.05% initial ratio')
    plt.title('Comparion of default edge ratio between entire and overlapped network', fontsize=titlefont)

    #plt.title('Shandong1',fontsize=titlefont) 
    plt.savefig('Comparion of default edge ratio between entire and overlapped network.png')
    plt.show()
    plt.close()
    return (default_ratio_node_list_GN, default_ratio_edge_list_GN, default_node_ratio_list_overlap, default_edge_ratio_list_overlap)

    
(default_ratio_node_list_GN, default_ratio_edge_list_GN, default_node_ratio_list_overlap, default_edge_ratio_list_overlap) = plot_comparison_GN_overlap_network (tags, xfont, yfont, titlefont, legendfont)     





#tag='0801'
#(g, nodes_has_info, default_ratio_node, default_ratio_edge) = get_guarantee_network(tag)
#(GN, g_overlap, overlap_nodes_has_info, edgenum_G_H)=get_overlap_ratio(g, tag)
#(g1, default_node_ratio, default_edge_ratio)=default_overlap_network (g_overlap, overlap_nodes_has_info)