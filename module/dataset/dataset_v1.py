import h5py
import torch
import pandas as pd
from torch_geometric.data import InMemoryDataset as PyGDataset
from bisect import bisect_right
from glob import glob
import numpy as np
import torch.nn

class dataset_v1(PyGDataset):
    def __init__(self, **kwargs):
        super(dataset_v1, self).__init__(None, transform=None, pre_transform=None)
        self.isLoaded = False

        self.fNames = []
        self.sampleInfo = pd.DataFrame(columns=["procName", "fileName", "fileIdx"])


    def len(self):
        return int(self.maxEventsList[-1])

    def get(self, idx):
        if not self.isLoaded: self.initialize()

        fileIdx = bisect_right(self.maxEventsList, idx)-1
        offset = self.maxEventsList[fileIdx]
        idx = int(idx - offset)


#         data = self.dataList[fileIdx][idx]
#         print(len(self.feaList))
        fea_List = self.feaList[fileIdx][idx]
        label_List = self.labelList[fileIdx][idx]
        mask_List = self.maskList[fileIdx][idx]
        labelmask_List = self.labelmaskList[fileIdx][idx]

        return fea_List, label_List, mask_List, labelmask_List
    
    def addSample(self, procName, fNamePattern, logger=None):
        if logger: logger.update(annotation='Add sample %s <= %s' % (procName, fNames))
        # print(procName, fNamePattern)

        for fName in glob(fNamePattern):
#           
            fileIdx = len(self.fNames)
            self.fNames.append(fName)
            info = {'procName':procName, 'nEvent':0, 'fileName':fName, 'fileIdx':fileIdx}
            # self.sampleInfo = self.sampleInfo.append(info, ignore_index=True) ### pandas version <2.0
            self.sampleInfo = pd.concat([self.sampleInfo,pd.DataFrame(info,index=[0])], ignore_index=True)   ### pandas version >=2.0

            
    ### data load and remake
    def initialize(self, output):
        if self.isLoaded: return

        
        procNames = list(self.sampleInfo['procName'].unique())  ## config file 'name'
        
        ### all file empty list
        self.procList, self.dataList = [], []
        self.feaList, self.labelList, self.maskList,self.labelmaskList = [], [], [], []     
        ### file num check
        nFiles = len(self.sampleInfo)
        
        max_pmts = 0        
        for i, fName in enumerate(self.sampleInfo['fileName']):
            
            ### file load and check event num
            f = torch.load(fName)

            nEvents = len(f)

            self.sampleInfo.loc[i, 'nEvent'] = nEvents
            for events in range(nEvents):
                    # print(f[j].x.shape)
                    if f[events].x.shape[0] > max_pmts:
                        max_pmts = f[events].x.shape[0]


        ### Load event contents
        for i, fName in enumerate(self.sampleInfo['fileName']):
            
            ### file load and check event num
            f = torch.load(fName)

            nEvents = len(f)

            self.sampleInfo.loc[i, 'nEvent'] = nEvents
            
            

            feas = []
            labels = []
            masks = []
            label_masks = []
            for events in range(nEvents):
                if f[events].x.shape[0]<max_pmts:
                    fea = torch.zeros(max_pmts,5)
                    label = torch.zeros(max_pmts)
                    mask = torch.zeros(max_pmts,5)
                    label_mask = torch.zeros(max_pmts)
                    
                    fea[:f[events].x.shape[0],:] = f[events].x
                    label[:f[events].y.shape[0]] = f[events].y
                    mask[:f[events].x.shape[0],:] = 1
                    label_mask[f[events].y.shape[0]:] = 1
                    
                    feas.append(fea)
                    labels.append(label)
                    masks.append(mask)
                    label_masks.append(label_mask)
                    

                else:
                    ## N jet = 14
                    feas.append(f[events].x)
                    labels.append(f[events].y)
                    masks.append(torch.ones(max_pmts,5))
                    label_masks.append(torch.zeros(max_pmts))


            self.feaList.append(feas)
            self.labelList.append(labels)
            self.maskList.append(masks)
            self.labelmaskList.append(label_masks)
#                   
    
#             self.dataList.append(f)

        

            procIdx = procNames.index(self.sampleInfo['procName'][i])
            self.procList.append(torch.ones(nEvents, dtype=torch.int32, requires_grad=False)*procIdx)


            
        ### save sampleInfo file in train result path
        SI = self.sampleInfo
        SI.to_csv('result/'+output + '/sampleInfo.csv')
        
        
        self.maxEventsList = np.concatenate(([0.], np.cumsum(self.sampleInfo['nEvent']))) 

        self.isLoaded = True
