from typing import Optional, Tuple, Union
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter
#from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.typing import (
    Adj,
    OptTensor,
    PairTensor,
    SparseTensor,
    #torch_sparse,
)
from torch_geometric.utils import (
    add_self_loops,
    is_torch_sparse_tensor,
    remove_self_loops,
    softmax,
    is_sparse,
)
import warnings
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures,LargestConnectedComponents
import networkx as nx
import torch_geometric.transforms as T
from torch_geometric.data import Data
from torch_geometric.utils import segregate_self_loops,dense_to_sparse,\
index_to_mask,get_laplacian,erdos_renyi_graph,to_networkx
from torchmetrics.functional import pairwise_cosine_similarity
import torch
from torch.nn import Linear
import torch.nn.functional as F 
from torch_geometric.nn import GCNConv,GATv2Conv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import copy
import json
import pprint
import pickle
from scipy import stats
import pandas as pd
import os.path
from math import floor,ceil
from itertools import product
import scipy.sparse as sp
from torch.distributions.multivariate_normal import MultivariateNormal
from tsne_torch import TorchTSNE as TSNE
from sklearn.cluster import KMeans
import heapq
from scipy import stats
#from msgpass import MessagePassing
from torchmetrics.functional.pairwise.helpers import _check_input
from torchmetrics import AUROC
warnings.filterwarnings("ignore")

def set_seeds(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def computeStatSumry(arr,quantiles,device):
  r = {'mean': arr.mean(),
        'std': arr.std()}
  quantiles=torch.cat((torch.tensor([0,1],device=device),quantiles),dim=0)
  p = torch.quantile(arr,quantiles)
  r['min'] = p[0]
  r['max'] = p[1]
  for i in range(2,len(quantiles)):
    r[str(int(quantiles[i]*100))+'%ile'] = p[i]
  return r

def makeDataDimsEven(data,input_dim,output_dim):
    if input_dim%2==1:
        a=torch.zeros((data.x.size()[0],ceil(data.x.size()[1]/2)*2))
        a[:,:input_dim] = data.x
        data.x = a
        input_dim+=1
    output_dim=(ceil(output_dim/2))*2
    return data,input_dim,output_dim

def printExpSettings(expID,expSetting):
    print('Exp: '+str(expID))
    for k,v in expSetting.items():
        for k2,v2 in expSetting[k].items():
            if(k2==expID):
                print(k,': ',v2)

def getDataHet(datasetName,splitID=1):

    print("Loading datasets as npz-file..")
    data = np.load('data/heterophilous-graphs/'+datasetName+'.npz')
    x = torch.tensor(data['node_features'], dtype=torch.float)
    y = torch.tensor(data['node_labels'], dtype=torch.long)
    edge_index = torch.tensor(data['edges'], dtype=torch.long).t().contiguous()
    train_mask = torch.tensor(data['train_masks'], dtype=torch.bool).transpose(0, 1).contiguous()
    val_mask = torch.tensor(data['val_masks'], dtype=torch.bool).transpose(0, 1).contiguous()
    test_mask = torch.tensor(data['test_masks'], dtype=torch.bool).transpose(0, 1).contiguous()
    num_classes = len(torch.unique(y))
    num_targets = 1 if num_classes == 2 else num_classes
    
    print("Converting to PyG dataset...")
    data = Data(x=x, edge_index=edge_index)
    data.y = y
    data.num_classes = num_classes
    data.num_targets = num_targets
    data.train_mask = train_mask[:,splitID] #split_idx = 1, 10 splits provided in dataset
    data.val_mask = val_mask[:,splitID]
    data.test_mask = test_mask[:,splitID]
    print(data.x.shape)
    return data,data.num_features,data.num_classes

def getData(datasetName, dataTransform, randomLabels=False,oneHotFeatures=False,randomLabelCount=None,splitID=1):
   
    if datasetName[:3]=='Syn': # == 'Synthetic':
        synID = datasetName.split("_")[1]
        with open('SyntheticData/D'+str(synID)+'.pkl', 'rb') as f:
            data = pickle.load(f)
            return data,data.x.shape[1],len(torch.unique(data.y))
    if datasetName[:3]=='SNA':
        synID = datasetName.split("_")[1]
        with open('SelectiveNeighborhoodAggregation/SyntheticData/SNA'+str(synID)+'.pkl', 'rb') as f:
            data = pickle.load(f)
            temp = data.edge_index[0].clone()
            data.edge_index[0] = data.edge_index[1].clone()
            data.edge_index[1] = temp
            return data,data.x.shape[1],len(torch.unique(data.y))        

    if datasetName in ['Cora','Citeseer','Pubmed']:
        dataset = Planetoid(root='data/Planetoid', name=datasetName, transform=NormalizeFeatures())
        data = dataset[0]
        if dataTransform=='removeIsolatedNodes':
            out = segregate_self_loops(data.edge_index)
            edge_index, edge_attr, loop_edge_index, loop_edge_attr = out
            mask = torch.zeros(data.num_nodes, dtype=torch.bool, device=data.x.device)
            mask[edge_index.view(-1)] = 1
            data.train_mask = data.train_mask & mask
            data.val_mask = data.val_mask & mask
            data.test_mask = data.test_mask & mask
        if dataTransform=='useLCC':
            transformLCC = LargestConnectedComponents()
            data = transformLCC(data)
        if randomLabels:
            if randomLabelCount==None:
                data.y = torch.randint(low=0,high=len(torch.unique(data.y)),size=data.y.shape)
            else:
                data.y = torch.randint(low=0,high=randomLabelCount,size=data.y.shape)
            
        if oneHotFeatures:
            data.x = torch.tensor(F.one_hot(data.y).clone().detach(),dtype = torch.float32)
        return data,data.x.shape[1],len(torch.unique(data.y))#dataset.num_features,dataset.num_classes
    else:
        return getDataHet(datasetName,splitID)
    
class GATv2(torch.nn.Module):
    def __init__(self, numLayers, dims, heads, concat, weightSharing, attnDropout=0,
                 bias=False,activation='relu',init='xavierN',balanceAtInit=False,
                 device='cpu',useIdMap=False, useResLin=False):
        super().__init__()
        self.numLayers = numLayers
        self.heads = heads
        self.weightSharing = weightSharing
        if activation =='relu':
            self.activation = torch.nn.ReLU()
        self.dropout = attnDropout
        self.device = device
        self.layers = torch.nn.ModuleList(
            [GATv2Conv(dims[j]*heads[j],dims[j+1],bias=bias,
                       heads=heads[j+1],concat=concat[j],share_weights=weightSharing,dropout=attnDropout) 
                       for j in range(self.numLayers)])

        self.params = [0] * numLayers
        for l in range(numLayers):
            self.params[l]={
                'w':self.layers[l].lin_l.weight,
                'a':self.layers[l].att,
            }
        
        self.initialize(init)
        if balanceAtInit:
            self.balanceAtInit()

    def initialize(self,init='xavierN'):
        if init=='xavierN':
            for l in range(self.numLayers):
                for p in self.params[l].keys():
                    torch.nn.init.xavier_normal_(self.params[l][p].data)
        elif(init=='LLortho'):
            for l in range(self.numLayers):
                for p in self.params[l].keys():
                    if p=='a':
                        #torch.nn.init.zeros_(self.params[l][p].data)
                        torch.nn.init.xavier_normal_(self.params[l][p].data)
                    else:
                        if l==0:
                            firstLayerDeltaDim = (ceil(self.params[l][p].data.shape[0]/2),self.params[l][p].data.shape[1])
                            firstLayerDelta = torch.nn.init.orthogonal_(torch.empty(firstLayerDeltaDim[0],firstLayerDeltaDim[1],device=self.device))
                            self.params[l][p].data = torch.cat((firstLayerDelta,-firstLayerDelta),dim=0)
            
                        
                        elif l==self.numLayers-1:
                            finalLayerDeltaDim = (self.params[l][p].data.shape[0],ceil(self.params[l][p].data.shape[1]/2))
                            finalLayerDelta = torch.nn.init.orthogonal_(torch.empty(finalLayerDeltaDim[0],finalLayerDeltaDim[1],device=self.device))
                            self.params[l][p].data = torch.cat((finalLayerDelta,-finalLayerDelta),dim=1)

                        else:
                            dimP = self.params[l][p].shape
                            submatrix = torch.nn.init.orthogonal_(
                                torch.empty(ceil(dimP[0]/2),ceil(dimP[1]/2),device=self.device))
                            submatrix = torch.cat((submatrix, -submatrix), dim=0)
                            submatrix = torch.cat((submatrix, -submatrix), dim=1)
                            self.params[l][p].data = submatrix
        return

    def balanceAtInit(self,sqNorm=2):    
        with torch.no_grad():     
            if self.weightSharing==True:
                incNorm = torch.sqrt(torch.pow(self.params[0]['w'].data,2).sum(axis=1))
                reqRowWiseL2Norm = torch.sqrt(torch.full((self.params[0]['w'].data.size()[0],),
                                                float(sqNorm),device=self.device))
                self.params[0]['w'].data = torch.multiply(
                                            torch.divide(self.params[0]['w'].data,
                                                         incNorm.reshape((len(incNorm),1))),\
                                            reqRowWiseL2Norm.reshape(len(reqRowWiseL2Norm),1))
                self.params[0]['a'].data = torch.zeros(self.params[0]['a'].data.shape,device=self.device)
                for l in range(1,self.numLayers):
                    self.params[l]['a'].data = torch.zeros(self.params[l]['a'].data.shape,device=self.device)
                    incNorm = torch.sqrt(torch.pow(self.params[l-1]['w'].data,2).sum(axis=1))
                    outNorm = torch.sqrt(torch.pow(self.params[l]['w'].data,2).sum(axis=0))
                    self.params[l]['w'].data = torch.multiply(
                                                    torch.divide(self.params[l]['w'].data,
                                                                 outNorm.reshape((1,len(outNorm)))),\
                                                    (incNorm).reshape((1,len(incNorm))))
                        #outSqNorm = torch.pow(self.params[l]['w'].data,2).sum(axis=0)
            else:
                print('----------non-weight-sharing balancing not implemented----------')
        for l in range(self.numLayers):
            for p in self.params[l].keys():
                self.params[l][p].data.requires_grad=True                

    def printInvarianceC(self,using='sqParamNorms',toPrint=True):
        with torch.no_grad():            
            if using=='sqParamNorms':
                inVar = [None] * (self.numLayers-1)
                for l in range(self.numLayers-1):
                    #inVar[l] = np.zeros((self.dims[l+1],))
                    inVar[l] = torch.pow(self.params[l]['w'].data.detach(),2).sum(axis=1) - \
                            torch.pow(self.params[l+1]['w'].data.detach(),2).sum(axis=0) - \
                            torch.pow(self.params[l]['a'].data.detach(),2)
                    if toPrint:
                        print('l: ',l,' c:',inVar[l])
            if using=='sqRelGradNorms':
                inVar = [None] * (self.numLayers-1)
                for l in range(self.numLayers-1):
                    inVar[l] = (torch.pow(torch.divide(self.params[l]['w'].grad.detach(),
                                                      self.params[l]['w'].data.detach()),2).sum(axis=1)) -\
                                (torch.pow(torch.divide(self.params[l+1]['w'].grad.detach(),
                                                        self.params[l+1]['w'].data.detach()),2).sum(axis=0))-\
                                (torch.pow(torch.divide(self.params[l]['a'].grad.detach(),
                                                        self.params[l]['a'].data.detach()),2))
                    if toPrint:
                        print('l: ',l,' c:',inVar[l])
        return inVar
    
    def scaleInvarBalance(self,numIter,using='sqParamNorms',onlyWeights=False,saveForbNormsBeforeAfterScaling=False):
        lambdaRec = [None] * numIter
        for n in range(numIter):
            lambdaRec[n] = [None] * (self.numLayers-1)
        forbNormsBeforeAfterRec = None
        if saveForbNormsBeforeAfterScaling:
            forbNormsBeforeAfterRec = [{p:{x: {y:None  for y in ['before','after']} for x in ['wght','grad','relGrad']}
                             for p in self.params[l].keys()} for l in range(self.numLayers)]
        #forbNormsBeforeRec = [{p:{x:torch.zeros(numEpochs, device=device) for x in ['wght','grad']}
        #                     for p in params[l].keys()} for l in range(numLayers)]
       
            
        with torch.no_grad():
            if using=='sqParamNorms':
                for it in range(numIter):
                    for l in range(self.numLayers-1):
                        #print('--------- layer: ',l)
                        c = torch.pow(torch.divide((torch.pow(self.params[l+1]['w'].data.detach(),2).sum(axis=0)+\
                                                    torch.pow(self.params[l]['a'].data.detach(),2)),\
                                            torch.pow(self.params[l]['w'].data.detach(),2).sum(axis=1)),0.25)
                        #print('lambda : ',c)
                        c = c.squeeze()            
                        c = torch.nan_to_num(c,1)            
                        #c[c==0] = 1
                        c = torch.clamp(c,0.00001,2)
                        lambdaRec[it][l] = c
                        self.params[l]['w'].data = torch.multiply(self.params[l]['w'].data.detach(),
                                                                c.reshape(len(c),1))
                        
                        self.params[l]['a'].data = torch.divide(self.params[l]['a'].data.detach(),
                                                                c)
                        
                        self.params[l+1]['w'].data = torch.divide(self.params[l+1]['w'].data.detach(),
                                                                c.reshape(1,len(c)))

                        #print('c: ',torch.pow(self.params[l+1]['w'].data.detach(),2).sum(axis=0) + \
                        #      torch.pow(self.params[l]['a'].data.detach(),2) - \
                        #        torch.pow(self.params[l]['w'].data.detach(),2).sum(axis=1))
                        
                        #print('Invar after scaling: ')

                        #self.printInvarianceC() 
            if using=='sqRelGradNorms':
                # print('Before: ')
                # self.printInvarianceC(using)
                if saveForbNormsBeforeAfterScaling:
                    for l in range(self.numLayers):
                        for p in self.params[l].keys():
                            if self.params[l][p]!=None:
                                forbNormsBeforeAfterRec[l][p]['wght']['before'] = torch.sqrt(torch.pow(self.params[l][p].data.detach(),2).sum())
                                forbNormsBeforeAfterRec[l][p]['grad']['before'] = torch.sqrt(torch.pow(self.params[l][p].grad.detach(),2).sum())
                                forbNormsBeforeAfterRec[l][p]['relGrad']['before'] = torch.sqrt(torch.pow(self.params[l][p].grad.detach()/self.params[l][p].data.detach(),2).sum())
                
                for it in range(numIter):
                    for l in range(self.numLayers-1):
                        c = torch.pow(torch.divide(torch.pow(torch.divide(self.params[l]['w'].grad.detach(),
                                                                self.params[l]['w'].data.detach()),2).sum(axis=1),
                                                    (torch.pow(torch.divide(self.params[l+1]['w'].grad.detach(),
                                                        self.params[l+1]['w'].data.detach()),2).sum(axis=0)+\
                                                    torch.pow(torch.divide(self.params[l]['a'].grad.detach(),
                                                        self.params[l]['a'].data.detach()),2))),0.125)
                        c = c.squeeze()
                        c = torch.nan_to_num(c,1)            
                        #c[c==0] = 1
                        c = torch.clamp(c,0.00001,2)
                        lambdaRec[it][l] = c 
                        self.params[l]['w'].data = torch.multiply(self.params[l]['w'].data.detach(),
                                                                c.reshape(len(c),1))
                        self.params[l]['a'].data = torch.divide(self.params[l]['a'].data.detach(),
                                                                c)
                        self.params[l+1]['w'].data = torch.divide(self.params[l+1]['w'].data.detach(),
                                                                c.reshape(1,len(c)))
                        if not onlyWeights:
                            self.params[l]['w'].grad = torch.divide(self.params[l]['w'].grad.detach(),
                                                                c.reshape(len(c),1))
                            self.params[l]['a'].grad = torch.multiply(self.params[l]['a'].grad.detach(),
                                                                c)
                            self.params[l+1]['w'].grad = torch.multiply(self.params[l+1]['w'].grad.detach(),
                                                                c.reshape(1,len(c)))
                
                if saveForbNormsBeforeAfterScaling:
                    for l in range(self.numLayers):
                        for p in self.params[l].keys():
                            if self.params[l][p]!=None:
                                forbNormsBeforeAfterRec[l][p]['wght']['after'] = torch.sqrt(torch.pow(self.params[l][p].data.detach(),2).sum())
                                forbNormsBeforeAfterRec[l][p]['grad']['after'] = torch.sqrt(torch.pow(self.params[l][p].grad.detach(),2).sum())
                                forbNormsBeforeAfterRec[l][p]['relGrad']['after'] = torch.sqrt(torch.pow(self.params[l][p].grad.detach()/self.params[l][p].data.detach(),2).sum())

                # print('After: ')
                # self.printInvarianceC(using)        

                        
            return lambdaRec,forbNormsBeforeAfterRec

    def set_curr_device(self,device):
        self.device = device

    def forward(self, x, edge_index,getAttnCoef=False):
        for i in range(self.numLayers):
            x_new,a = self.layers[i](x,edge_index,return_attention_weights=getAttnCoef)
            #add code for skip connections here 
            x=x_new
            if i <(self.numLayers-1):
                x = self.activation(x)
                if self.dropout>0:
                    x = F.dropout(x, p=self.dropout, training=self.training)
        return x
    
    def getParams(self):
        return self.params
    

def runExpWithGradientClipping (expID,runIDs,device,trainLossToConverge,printLossEveryXEpoch,
           saveParamGradStatSumry,saveNeuronLevelL2Norms,saveLayerWiseForbNorms,
           saveInvarC,saveRescalingFactors,saveForbNormsBeforeAfterScaling,expSetting,resultPath):

    quantiles = torch.tensor((np.array(range(1,10,1))/10),dtype=torch.float32,device=device)
    qLabels = [str(int(q*100))+'%ile' for q in quantiles]
    labels = ['min','max','mean','std']+qLabels 

    if len(runIDs)==0:
        runIDs = range(int(expSetting['numRuns'][expID]))
    datasetName = str(expSetting['dataset'][expID])
    dataTransform = str(expSetting['dataTransform'][expID]) #removeIsolatedNodes,useLCC 
    
    optim = str(expSetting['optimizer'][expID])
    numEpochs = int(expSetting['maxEpochs'][expID])
    lr = float(expSetting['initialLR'][expID])
    attnDropout = float(expSetting['attnDropout'][expID])
    #wghtDecay =  float(expSetting['wghtDecay'][expID])

    numLayers = int(expSetting['numLayers'][expID])
    hiddenDims = [int(expSetting['hiddenDim'][expID])] * (numLayers-1)
    mulLastAttHead = bool(expSetting['mulLastAttHead'][expID])
    #data input always has 1 attention head, decide for last layer
    if mulLastAttHead:
        heads = [1] + ([int(expSetting['attnHeads'][expID])] * (numLayers)) 
    else:
        heads = [1] + ([int(expSetting['attnHeads'][expID])] * (numLayers-1)) + [1] 
    #concat attn heads for all layers except the last, avergae for last (doesn't matter when num of heads for last layer=1)
    concat = ([True] * (numLayers-1)) + [False] 
    activation = str(expSetting['activation'][expID])
    weightSharing = bool(expSetting['weightSharing'][expID])

    initScheme=str(expSetting['initScheme'][expID])
    balanceAtInit=bool(expSetting['balanceAtInit'][expID])
    numBalanceIters=None
    if len(str(expSetting['numBalanceIters'][expID]))>0:
        numBalanceIters=int(expSetting['numBalanceIters'][expID])
    rebalanceInterval=None
    if len(str(expSetting['rebalanceInterval'][expID]))>0:
        rebalanceInterval=int(expSetting['rebalanceInterval'][expID])
    balScheme=None
    if len(str(expSetting['balScheme'][expID]))>0:
        balScheme = str(expSetting['balScheme'][expID])
    adaptLR = None
    if len(str(expSetting['adaptLR'][expID]))>0:
        adaptLR = int(expSetting['adaptLR'][expID])
    gradClipNormVal=None
    if len(str(expSetting['gradClipNormVal'][expID]))>0:
        gradClipNormVal = float(expSetting['gradClipNormVal'][expID])
    lossCriterion = str(expSetting['lossCriterion'][expID])
    metric = str(expSetting['metric'][expID])
    print('*******')
    printExpSettings(expID,expSetting)
    print('*******')

    for run in runIDs:#range(numRuns):
        print('-- RUN ID: '+str(run))
        set_seeds(run)
        data,input_dim,output_dim = getData(datasetName,dataTransform,splitID=run) 
        if input_dim%2==1:
            data,input_dim,o=makeDataDimsEven(data,input_dim,output_dim)
        data = data.to(device)    
        dims = [input_dim]+hiddenDims+[output_dim]
        model = GATv2(numLayers,dims,heads,concat, weightSharing,attnDropout, 
                      activation=activation,init=initScheme,balanceAtInit=balanceAtInit).to(device)
        model.set_curr_device(device)
        if optim=='SGD':
            optimizer = torch.optim.SGD(model.parameters(), lr=lr)#, weight_decay=wghtDecay)
        if optim=='Adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)#, weight_decay=wghtDecay)
        if lossCriterion == 'CEL':
            criterion = torch.nn.CrossEntropyLoss()
        if output_dim==2:
            auroc = AUROC(task='binary')
        else:
            auroc = AUROC(task='multiclass',num_classes=output_dim)
        #criterion = torch.nn.CrossEntropyLoss()
        

        trainLoss = torch.zeros(numEpochs, dtype=torch.float32, device = device)
        valLoss = torch.zeros(numEpochs, dtype=torch.float32, device = device)
        testLoss = torch.zeros(numEpochs, dtype=torch.float32, device = device)
        trainMetric = torch.zeros(numEpochs, dtype=torch.float32, device = device)
        valMetric = torch.zeros(numEpochs, dtype=torch.float32, device = device)
        testMetric = torch.zeros(numEpochs, dtype=torch.float32, device = device)
        
        paramStatSumry=None
        neuronSqL2Norms=None
        forbNorms=None
        invarC=None
        params = model.getParams()
        
        if saveParamGradStatSumry:
            paramStatSumry = [{} for _ in range(numLayers)]
            for l in range(numLayers):
                for p in params[l].keys():
                    if params[l][p]!=None:
                        paramStatSumry[l][p] = {x2:{x:torch.zeros(numEpochs,device=device) for x in labels} for x2 in ['wght','grad']}
                    else: 
                        paramStatSumry[l][p] = None
        
        if saveNeuronLevelL2Norms:
            neuronSqL2Norms = [{x:{p:{} for p in params[l].keys()} for x in ['wght','grad']}  for l in range(numLayers)]
            for l in range(numLayers):
                for x in ['wght','grad']:
                    for p in params[l].keys():
                        if p == 'w':
                        #incoming: row-wise of W matrix, and outgoing is col-wise of W matrix
                            neuronSqL2Norms[l][x][p]={'row':torch.zeros((numEpochs,dims[l+1]),device=device), #row is out
                                                'col':torch.zeros((numEpochs,dims[l]),device=device)} #col is inc
                        if p == 'a':
                            neuronSqL2Norms[l][x][p]=torch.zeros((numEpochs,dims[l+1]),device=device)
           
        if saveLayerWiseForbNorms:
            forbNorms = [{p:{x:torch.zeros(numEpochs, device=device) for x in ['wght','grad','relGrad']}
                             for p in params[l].keys()} for l in range(numLayers)]
        
        if saveInvarC:
            invarC = [torch.zeros((numEpochs,dims[l+1]),device=device) for l in range(numLayers-1)]

        if saveRescalingFactors:
            rescalingFactors = {} # key will be epoch and val will be the lambdaRec returned by rescaling function

        if saveForbNormsBeforeAfterScaling:
            forbNormsBeforeAfterScaling = {}

        maxValAcc = 0
        continueTraining = True      
        epoch=0
        prevWghts = [{} for _ in range(numLayers)]
        for l in range(numLayers):
            for p in params[l].keys():
                prevWghts[l][p] = params[l][p].data.detach().clone()

        orglr = lr
        while(epoch<numEpochs and continueTraining):
            if balScheme=='sqParamNorms':
                if saveInvarC:
                    retInvarC = model.printInvarianceC(balScheme,False)
                    for l in range(numLayers-1):
                        invarC[l][epoch] = retInvarC[l]
                if epoch%(rebalanceInterval)==0:
                    r,f = model.scaleInvarBalance(numBalanceIters,balScheme,False,saveForbNormsBeforeAfterScaling)  
                    if saveRescalingFactors:
                        rescalingFactors[epoch] = r
                    if saveForbNormsBeforeAfterScaling:
                        forbNormsBeforeAfterScaling[epoch] = f
                    

            # if epoch>0 and epoch%30==0:
            #     model.scaleInvarBalance(numBalanceIters,'sqParamNorms')

            #optimizer.param_groups[0]['lr'] = orglr
            model.train()
            optimizer.zero_grad()    
            out = model(data.x, data.edge_index)
            loss = criterion(out[data.train_mask], data.y[data.train_mask])  
            trainLoss[epoch] = loss.detach()
            pred = out.argmax(dim=1)  
            if metric=='Accuracy':
                train_correct = pred[data.train_mask] == data.y[data.train_mask] 
                trainMetric[epoch] = int(train_correct.sum()) / int(data.train_mask.sum())  
            # pred = out.argmax(dim=1)  
            # train_correct = pred[data.train_mask] == data.y[data.train_mask] 
            # trainAcc[epoch] = int(train_correct.sum()) / int(data.train_mask.sum())  
            if metric=='AUROC':
                trainMetric[epoch] = auroc(pred[data.train_mask], data.y[data.train_mask]).item()
            loss.backward()  
            
            
            # if epoch == 0 :
            #     r,f  = model.scaleInvarBalance(10,'sqRelGradNorms',False,saveForbNormsBeforeAfterScaling) 
            #     if saveRescalingFactors:
            #         rescalingFactors[epoch] = r
            #     if saveForbNormsBeforeAfterScaling:
            #         forbNormsBeforeAfterScaling[epoch] = f


            
                
                # r,f  = model.scaleInvarBalance(10,'sqRelGradNorms',False,saveForbNormsBeforeAfterScaling) 
                # if saveRescalingFactors:
                #     rescalingFactors[epoch] = r
                # if saveForbNormsBeforeAfterScaling:
                #     forbNormsBeforeAfterScaling[epoch] = f

            
            
            
            if balScheme=='sqRelGradNorms':
                toRB = False
                if saveInvarC:
                    retInvarC = model.printInvarianceC(balScheme,False)
                    for l in range(numLayers-1):
                        invarC[l][epoch] = retInvarC[l]
                        # temp =torch.abs(invarC[l][epoch]-invarC[l][epoch-1]) > (10**5)
                        # if temp.any():
                        #     #print('-epoch: ',epoch,' -layer: ',l,' feats: ', temp.nonzero().squeeze())
                        #     #model.printInvarianceC(balScheme,True)
                        #     #toRB = True
                        #     temp=None

                if epoch%(rebalanceInterval)==0:
                    r,f  = model.scaleInvarBalance(numBalanceIters,balScheme,False,saveForbNormsBeforeAfterScaling) 
                    if saveRescalingFactors:
                        rescalingFactors[epoch] = r
                    if saveForbNormsBeforeAfterScaling:
                        forbNormsBeforeAfterScaling[epoch] = f
                    
                #     toRB=True
                # rbCount=0
                # maxRBcount=1
                # while toRB==True and rbCount<maxRBcount:
                #     toRB=False
                #     lambdaRec = model.scaleInvarBalance(numBalanceIters,balScheme,False) 
                #     rbCount += 1
                #     print('Epoch: ',epoch, 'RB-RelGrads')
                    # if saveInvarC:
                    #     retInvarC = model.printInvarianceC(balScheme,False)
                    #     for l in range(numLayers-1):
                    #         invarC[l][epoch] = retInvarC[l]
                    #         temp =torch.abs(invarC[l][epoch]-invarC[l][epoch-1]) > (10**5)
                    #         if temp.any():
                    #             #print('ALERT -- too high C even after rebalancing')
                    #             #print('-epoch: ',epoch,' -layer: ',l,' feats: ', temp.nonzero().squeeze()) 
                    #             #toRB = True
                    #             temp=None
                
                #if saveLayerWiseForbNorms:
                # toRBweightNorms = False
                # for l in range(numLayers):
                #     for p in params[l].keys():
                #         if params[l][p]!=None:
                #             #forbNorms[l][p]['grad'][epoch] = 
                #             if torch.sqrt(torch.pow(params[l][p].data.detach(),2).sum()) > 100:
                #                 toRBweightNorms=True
                
                # maxRBwghtNormCount =10 
                # RBwghtNormCount=0
                # while toRBweightNorms and RBwghtNormCount<maxRBwghtNormCount:
                #     toRBweightNorms=False
                #     RBwghtNormCount+=1
                #     model.scaleInvarBalance(numBalanceIters,'sqParamNorms') 
                #     print('Epoch: ',epoch, 'RB-WghtNorms')                    
                #     for l in range(numLayers):
                #         for p in params[l].keys():
                #             if p=='w':
                #                 if torch.sqrt(torch.pow(params[l][p].data.detach(),2).sum()) > 1000:
                #                     toRBweightNorms=True
                #             if p=='a':
                #                 if torch.sqrt(torch.pow(params[l][p].data.detach(),2).sum()) > 100:
                #                     toRBweightNorms=True

            if gradClipNormVal!=None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradClipNormVal, norm_type=2)

            if saveParamGradStatSumry:
                for l in range(numLayers):
                    for p in params[l].keys():
                        if params[l][p]!=None:
                            for k,v in computeStatSumry(params[l][p].data.detach(),quantiles,device).items():
                                paramStatSumry[l][p]['wght'][k][epoch] = v 
            
            if saveNeuronLevelL2Norms:
                for l in range(numLayers):
                    for p in params[l].keys():
                        if p == 'w':
                            neuronSqL2Norms[l]['wght'][p]['row'][epoch] = torch.pow(params[l][p].data.detach(),2).sum(axis=1)
                            neuronSqL2Norms[l]['wght'][p]['col'][epoch] = torch.pow(params[l][p].data.detach(),2).sum(axis=0)
                        if p == 'a':
                            neuronSqL2Norms[l]['wght'][p][epoch] = torch.pow(params[l][p].data.detach(),2)
                        
            if saveLayerWiseForbNorms:
                for l in range(numLayers):
                    for p in params[l].keys():
                        if params[l][p]!=None:
                            forbNorms[l][p]['wght'][epoch] = torch.sqrt(torch.pow(params[l][p].data.detach(),2).sum())
            
            

                # if rbCount>0:
                #     print('---Epoch: ',epoch, ' RBcount: ',rbCount)
                    # tempGrad = [None] * numLayers
                    # for l in range(numLayers):
                    #     tempGrad[l] = {}
                    #     for p in params[l].keys():
                    #         tempGrad[l][p] = params[l][p].grad.detach().clone()
                    
                    # if saveLayerWiseForbNorms:
                    #     for l in range(numLayers):
                    #         for p in params[l].keys():
                    #             if params[l][p]!=None:
                    #                 forbNorms[l][p]['grad'][epoch] = torch.sqrt(torch.pow(params[l][p].grad.detach(),2).sum())
                

                    
                    # model.train()
                    # optimizer.zero_grad()    
                    # out = model(data.x, data.edge_index)
                    # loss = criterion(out[data.train_mask], data.y[data.train_mask])  
                    # #print(epoch, trainLoss[epoch], loss.detach(), trainLoss[epoch]==loss.detach())
                    # trainLoss[epoch] = loss.detach()
                    # pred = out.argmax(dim=1)  
                    # train_correct = pred[data.train_mask] == data.y[data.train_mask] 
                    # trainAcc[epoch] = int(train_correct.sum()) / int(data.train_mask.sum())  
                    
                    # loss.backward()
                    # for l in range(numLayers):
                    #     for p in params[l].keys():
                    #         print(l,p,' grad Vals: ',torch.equal(tempGrad[l][p],params[l][p].grad.detach()),
                    #               torch.abs(tempGrad[l][p]-params[l][p].grad.detach()).sum())
                    #         print(l,p,' grads ForbNorm: ',forbNorms[l][p]['grad'][epoch],
                    #               torch.sqrt(torch.pow(params[l][p].grad.detach(),2).sum()),
                    #               torch.equal(forbNorms[l][p]['grad'][epoch],\
                    #                         torch.sqrt(torch.pow(params[l][p].grad.detach(),2).sum())))
            
            # for l in range(numLayers):
            #     for p in params[l].keys():
            #         prevWghts[l][p] = params[l][p].data.detach().clone()        
            

            if saveParamGradStatSumry:
                for l in range(numLayers):
                    for p in params[l].keys():
                        if params[l][p]!=None:
                            for k,v in computeStatSumry(params[l][p].grad.detach(),quantiles,device).items():
                                paramStatSumry[l][p]['grad'][k][epoch] = v 
                     
            if saveLayerWiseForbNorms:
                for l in range(numLayers):
                    for p in params[l].keys():
                        if params[l][p]!=None:
                            forbNorms[l][p]['grad'][epoch] = torch.sqrt(torch.pow(params[l][p].grad.detach(),2).sum())
                            forbNorms[l][p]['relGrad'][epoch] = torch.sqrt(torch.pow(params[l][p].grad.detach()/params[l][p].data.detach(),2).sum())
#                             

            if saveNeuronLevelL2Norms:
                for l in range(numLayers):
                    for p in params[l].keys():
                        if p == 'w':
                            #print('DIMS: ',p,params[l][p].data.shape)
                            #print('DIMS sum: ',torch.pow(params[l][p].data.detach(),2).sum(axis=1).shape)
                            #print('row', neuronSqL2Norms[l][p]['row'].shape)
                            neuronSqL2Norms[l]['grad'][p]['row'][epoch] = torch.pow(params[l][p].grad.detach(),2).sum(axis=1)
                            neuronSqL2Norms[l]['grad'][p]['col'][epoch] = torch.pow(params[l][p].grad.detach(),2).sum(axis=0)
                        if p == 'a':
                            neuronSqL2Norms[l]['grad'][p][epoch] = torch.pow(params[l][p].grad.detach(),2)
                    
            optimizer.step() 

            model.eval()
            with torch.no_grad():
                out = model(data.x, data.edge_index)
                valLoss[epoch] = criterion(out[data.val_mask], data.y[data.val_mask]).detach() 
                testLoss[epoch] = criterion(out[data.test_mask], data.y[data.test_mask]).detach() 
                pred = out.argmax(dim=1)  
                if metric=='Accuracy':
                    val_correct = pred[data.val_mask] == data.y[data.val_mask]  
                    valMetric[epoch] = int(val_correct.sum()) / int(data.val_mask.sum())  
                    test_correct = pred[data.test_mask] == data.y[data.test_mask] 
                    testMetric[epoch] =  int(test_correct.sum()) / int(data.test_mask.sum()) 
                # pred = out.argmax(dim=1)  
                # val_correct = pred[data.val_mask] == data.y[data.val_mask]  
                # valAcc[epoch] = int(val_correct.sum()) / int(data.val_mask.sum())  
                # test_correct = pred[data.test_mask] == data.y[data.test_mask] 
                # testAcc[epoch] =  int(test_correct.sum()) / int(data.test_mask.sum()) 
                if metric=='AUROC':
                    valMetric[epoch] = auroc(pred[data.val_mask], data.y[data.val_mask]).item()
                    testMetric[epoch] = auroc(pred[data.test_mask], data.y[data.test_mask]).item()


            if(trainLoss[epoch]<trainLossToConverge or torch.isnan(trainLoss).any()):
                if torch.isnan(trainLoss).any():
                    print('Train Loss is NaN @ Epoch', epoch)
                continueTraining=False
            else:
                if(trainLoss[epoch]<trainLossToConverge):
                    continueTraining=False

            if(epoch%printLossEveryXEpoch==0 or epoch==numEpochs-1):
                print(f'--Epoch: {epoch:03d}, Train Loss: {loss:.4f}, Train Metric: {trainMetric[epoch]:.4f}, Test Metric: {testMetric[epoch]:.4f}')
            epoch+=1

        trainLoss = trainLoss[:epoch].detach().cpu().numpy()
        valLoss = valLoss[:epoch].detach().cpu().numpy()
        testLoss = testLoss[:epoch].detach().cpu().numpy()
        trainMetric = trainMetric[:epoch].detach().cpu().numpy()
        valMetric = valMetric[:epoch].detach().cpu().numpy()
        testMetric = testMetric[:epoch].detach().cpu().numpy()
    
        
        print('Max or Convergence Epoch: ', epoch)
        print('Max Val Metric At Epoch: ', np.argmax(valMetric)+1)
        print('Test metric at Max Val Metric:', testMetric[np.argmax(valMetric)]*100)
    
        if saveParamGradStatSumry:
            for l in range(numLayers):
                    for p in paramStatSumry[l].keys():
                        for x in labels:
                            paramStatSumry[l][p]['wght'][x] = paramStatSumry[l][p]['wght'][x][:epoch].cpu().numpy()
                            paramStatSumry[l][p]['grad'][x] = paramStatSumry[l][p]['grad'][x][:epoch].cpu().numpy()


        if saveNeuronLevelL2Norms:
            for l in range(numLayers):
                for x in ['wght','grad']:
                    for p in params[l].keys():
                        if p == 'w':
                            neuronSqL2Norms[l][x][p]['row'] = neuronSqL2Norms[l][x][p]['row'][0:epoch,:].T.cpu().numpy()
                            neuronSqL2Norms[l][x][p]['col'] = neuronSqL2Norms[l][x][p]['col'][0:epoch,:].T.cpu().numpy()
                        if p == 'a':
                            neuronSqL2Norms[l][x][p] = neuronSqL2Norms[l][x][p][0:epoch,:].T.cpu().numpy()
                      
        if saveLayerWiseForbNorms:
            for l in range(numLayers):
                for p in params[l].keys():
                    forbNorms[l][p]['wght'] = forbNorms[l][p]['wght'][:epoch].cpu().numpy()
                    forbNorms[l][p]['grad'] = forbNorms[l][p]['grad'][:epoch].cpu().numpy()
                    forbNorms[l][p]['relGrad'] = forbNorms[l][p]['relGrad'][:epoch].cpu().numpy()
        
        if saveInvarC:
            for l in range(numLayers-1):
                invarC[l] = invarC[l][:epoch,:].cpu().numpy()

        if saveRescalingFactors:
            for k in rescalingFactors.keys():
                for i in range(len(rescalingFactors[k])):
                    for l in range(len(rescalingFactors[k][i])):
                        rescalingFactors[k][i][l] = rescalingFactors[k][i][l].cpu().numpy()
        
        if saveForbNormsBeforeAfterScaling:
            
            for e in rescalingFactors.keys():
                for l in range(numLayers):
                    for k1 in forbNormsBeforeAfterScaling[e][l].keys():
                        for k2 in forbNormsBeforeAfterScaling[e][l][k1].keys():
                            for k3 in forbNormsBeforeAfterScaling[e][l][k1][k2].keys():
                                forbNormsBeforeAfterScaling[e][l][k1][k2][k3] = forbNormsBeforeAfterScaling[e][l][k1][k2][k3].cpu().numpy()
             

        expDict = {'expID':expID,  
                'trainedEpochs':epoch,
                'trainLoss':trainLoss,
                'valLoss':valLoss,
                'testLoss':testLoss,
                'trainMetric':trainMetric,
                'valMetric':valMetric,
                'testMetric':testMetric,
                'lossCriterion':lossCriterion,
                'metric':metric
        }

        with open(resultPath+'dictExp'+str(expID)+'_run'+str(run)+'.pkl', 'wb') as f:
            pickle.dump(expDict,f)

        if saveInvarC:
            invarCDict = {'expID':expID,  
                        'numLayers':numLayers,
                        'trainedEpochs':epoch,
                        'invarC':invarC,
                        'balScheme':balScheme
            }
            with open(resultPath+'inVarCExp'+str(expID)+'_run'+str(run)+'.pkl', 'wb') as f:
                pickle.dump(invarCDict,f)

        if saveRescalingFactors:
            rescalingFactorsDict = {'expID':expID,  
                        'numLayers':numLayers,
                        'trainedEpochs':epoch,
                        'rebalanceInterval':rebalanceInterval,
                        'balScheme':balScheme,
                        'numBalanceIters':numBalanceIters,
                        'rescalingFactors':rescalingFactors

            }
            with open(resultPath+'rescalingFactorsExp'+str(expID)+'_run'+str(run)+'.pkl', 'wb') as f:
                pickle.dump(rescalingFactorsDict,f)

        if saveForbNormsBeforeAfterScaling:
            forbNormsBeforeAfterScalingDict = {'expID':expID,  
                        'numLayers':numLayers,
                        'trainedEpochs':epoch,
                'forbNormsBeforeAfterScaling':forbNormsBeforeAfterScaling
            }
            with open(resultPath+'forbNormsBeforeAfterScalingExp'+str(expID)+'_run'+str(run)+'.pkl', 'wb') as f:
                pickle.dump(forbNormsBeforeAfterScalingDict,f)

        if saveParamGradStatSumry:
            saveParamStatSumry = {'expID':expID,
                        'numLayers':numLayers,
                        'trainedEpochs':epoch,
                        'quantiles':quantiles.cpu().numpy(),
                        'statSumry':paramStatSumry
                    }
            with open(resultPath+'paramStatSumryExp'+str(expID)+'_run'+str(run)+'.pkl', 'wb') as f:
                pickle.dump(saveParamStatSumry,f)

        if saveNeuronLevelL2Norms:
            saveNeuronLevelSqL2NormsDict = {
                        'expID':expID,
                        'numLayers':numLayers,
                        'trainedEpochs':epoch,
                        'neuronSqL2Norms':neuronSqL2Norms
                    }
            with open(resultPath+'neuronLevelSqL2NormsExp'+str(expID)+'_run'+str(run)+'.pkl', 'wb') as f:
                pickle.dump(saveNeuronLevelSqL2NormsDict,f)

        if saveLayerWiseForbNorms:
            saveForbNorms = {'expID':expID,
                        'numLayers':numLayers,
                        'trainedEpochs':epoch,
                        'forbNorms':forbNorms
                }
            with open(resultPath+'forbNormsExp'+str(expID)+'_run'+str(run)+'.pkl', 'wb') as f:
                pickle.dump(saveForbNorms,f)

def main():
    device =  torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    expSetting = pd.read_csv('InitExpSettingsRB.csv',index_col='expId').fillna('').to_dict()
    for exp in range(1,30+1):
       runExpWithGradientClipping(expID=exp,
              runIDs=[],
              device=device,
              expSetting=expSetting,
              trainLossToConverge=0,
              printLossEveryXEpoch=1000,
              saveParamGradStatSumry=False,
              saveNeuronLevelL2Norms=False,
              saveLayerWiseForbNorms=False,
              saveInvarC=False,
              saveRescalingFactors=False,
              saveForbNormsBeforeAfterScaling=False,
              resultPath="ExpResultsRB/")
       
if __name__ == "__main__":
    main()





