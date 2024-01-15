#!/usr/bin/env python
import numpy as np
import torch
import h5py
import torch.nn as nn
import torch.nn.functional as F
import sys, os
import subprocess
import csv, yaml
import math
from tqdm import tqdm
import torch.optim as optim
import argparse
import pandas as pd
from torch.utils.data import DataLoader
sys.path.append("./module")
from model.allModel import *
from saja_loss import *


parser = argparse.ArgumentParser()
parser.add_argument('--config', action='store', type=str, help='Configration file with sample information')
parser.add_argument('-o', '--output', action='store', type=str, required=True, help='Path to output directory')
parser.add_argument('--device', action='store', type=int, default=0, help='device name')
args = parser.parse_args()

config = yaml.load(open(args.config).read(), Loader=yaml.FullLoader)

torch.set_num_threads(os.cpu_count())
if torch.cuda.is_available() and args.device >= 0: torch.cuda.set_device(args.device)
if not os.path.exists('result/' + args.output): os.makedirs('result/' + args.output)

##### Define dataset instance #####
from dataset.dataset_v1 import *
dset = dataset_v1()

for sampleInfo in config['samples']:
    if 'ignore' in sampleInfo and sampleInfo['ignore']: continue
    name = sampleInfo['name']
    dset.addSample(name, sampleInfo['path'])
dset.initialize(args.output)


lengths = [int(x*len(dset)) for x in config['training']['splitFractions']]
lengths.append(len(dset)-sum(lengths))
torch.manual_seed(config['training']['randomSeed'])
trnDset, valDset, testDset = torch.utils.data.random_split(dset, lengths)


kwargs = {'num_workers':min(config['training']['nDataLoaders'],os.cpu_count()), 'pin_memory':False}

trnLoader = DataLoader(trnDset, batch_size=config['training']['batch'], shuffle=True, **kwargs)
valLoader = DataLoader(valDset, batch_size=config['training']['batch'], shuffle=False, **kwargs)
torch.manual_seed(torch.initial_seed())

model_y = config['model']['model']
fea_y = config['model']['fea']
cla_y = config['model']['cla']
depths_y = config['model']['depths']
hidden_y = config['model']['hidden']
heads_y = config['model']['heads']
posfeed_y = config['model']['posfeed']
dropout_y = config['model']['dropout']
batch_y = config['training']['batch']

# #### Define model instance #####
exec('model = '+config['model']['model']+'(fea = fea_y, \
                             cla = cla_y, \
                             depths = depths_y, \
                             hidden = hidden_y, \
                             heads = heads_y, \
                             posfeed = posfeed_y, \
                             dropout = dropout_y, \
                             batch = batch_y, \
                             device= args.device)')
     

torch.save(model, os.path.join('result/' + args.output, 'model.pth'))



device = 'cpu'
if args.device >= 0 and torch.cuda.is_available():
    model = model.cuda()
    device = 'cuda'


optm = torch.optim.Adam(model.parameters(), lr=config['training']['learningRate'])


    
from sklearn.metrics import accuracy_score
from tqdm import tqdm
bestState, bestLoss = {}, 1e9
train = {'loss':[], 'val_loss':[]}
nEpoch = config['training']['epoch']
for epoch in range(nEpoch):
    model.train()
    
    trn_loss, trn_acc = 0., 0.
    nProcessed = 0
    
    true_num_trn = 0
    total_num_trn = 0
    for i, (feas,labels, mask,label_mask) in enumerate(tqdm(trnLoader, desc='epoch %d/%d' % (epoch+1, nEpoch))):
        
        data = feas.to(device)
        labels = labels.type(torch.LongTensor).to(device=device)
        mask = mask.to(device)
        label_mask = label_mask.to(device)
        


        pred = model(data,mask)


        
        loss = saja_loss(pred,labels,label_mask.shape[1]-label_mask.sum(dim=1),label_mask,True)
#         print(torch.argmax(pred.softmax(dim=-1),dim=-1))
#         print(torch.argmax(pred.softmax(dim=-1),dim=-1).shape)
#         print(torch.abs(torch.argmax(pred.softmax(dim=-1),dim=-1).masked_fill(label_mask.bool(),0)-labels))
#         print(labels.shape)
#         print(labels)
        
        true_num_trn += ((torch.abs(torch.argmax(pred.softmax(dim=-1),dim=-1).masked_fill(label_mask.bool(),0)-labels)).sum(dim=-1)==0).sum()
        total_num_trn += len(labels)
        
        
        trn_accu = true_num_trn/total_num_trn
        
        loss.backward()
        optm.step()
        optm.zero_grad()
      

        ibatch = len(labels)
        nProcessed += ibatch
        trn_loss += loss.item()*ibatch

        

    trn_loss /= nProcessed 

    print(trn_loss,'trn_loss')
    print(trn_accu,'trn_acc')

    model.eval()
    val_loss, val_acc = 0., 0.
    nProcessed = 0
    true_num_val = 0
    total_num_val = 0
    for i, (feas,labels, mask,label_mask) in enumerate(tqdm(valLoader)):

        data = feas.to(device)
        labels = labels.type(torch.LongTensor).to(device=device)
        mask = mask.to(device)
        label_mask = label_mask.to(device)
        


        pred = model(data,mask)

        loss = saja_loss(pred,labels,label_mask.shape[1]-label_mask.sum(dim=1),label_mask,True)

        true_num_val += ((torch.abs(torch.argmax(pred.softmax(dim=-1),dim=-1).masked_fill(label_mask.bool(),0)-labels)).sum(dim=-1)==0).sum()
        total_num_val += len(labels)
        
        
        val_accu = true_num_val/total_num_val
        ibatch = len(labels)
        nProcessed += ibatch
        val_loss += loss.item()*ibatch

            
            
    val_loss /= nProcessed
    print(val_loss,'val_loss')
    print(val_accu, 'val_acc')
    
    if bestLoss > val_loss:
        bestState = model.to('cpu').state_dict()
        bestLoss = val_loss
        torch.save(bestState, os.path.join('result/' + args.output, 'weight.pth'))

        model.to(device)
    
    
    train['loss'].append(trn_loss)
    train['val_loss'].append(val_loss)

    with open(os.path.join('result/' + args.output, 'train.csv'), 'w') as f:
        writer = csv.writer(f)
        keys = train.keys()
        writer.writerow(keys)
        for row in zip(*[train[key] for key in keys]):
            writer.writerow(row)
    
    


bestState = model.to('cpu').state_dict()
torch.save(bestState, os.path.join('result/' + args.output, 'weightFinal.pth'))
