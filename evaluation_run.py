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
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
 
sys.path.append("./module")
from model.allModel import *

parser = argparse.ArgumentParser()
parser.add_argument('--config', action='store', type=str, default='config.yaml', help='Configration file with sample information')
parser.add_argument('-o', '--output', action='store', type=str, required=True, help='Path to output file')
parser.add_argument('--device', action='store', type=int, default=0, help='device name')
args = parser.parse_args()

config = yaml.load(open(args.config).read(), Loader=yaml.FullLoader)



torch.set_num_threads(os.cpu_count())
if torch.cuda.is_available() and args.device >= 0: torch.cuda.set_device(args.device)

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

kwargs = {'num_workers':min(config['training']['nDataLoaders'], os.cpu_count()), 'pin_memory':False}


testLoader = DataLoader(testDset, batch_size=config['training']['batch'], shuffle=False, **kwargs)
torch.manual_seed(torch.initial_seed())

##### Define model instance #####



device = 'cpu'
model = torch.load('result/' + args.output+'/model.pth', map_location='cpu')
model.load_state_dict(torch.load('result/' + args.output+'/weight.pth', map_location='cpu'))

if args.device >= 0 and torch.cuda.is_available():
    model = model.cuda()
    device = 'cuda'
    model = torch.load('result/' + args.output+'/model.pth', map_location=device)
    model.load_state_dict(torch.load('result/' + args.output+'/weight.pth', map_location=device))

dd = 'result/' + args.output + '/train.csv'

dff = pd.read_csv(dd)



##### Start evaluation #####
from tqdm import tqdm
label_s,  label_masks = [], []
preds, preds_soft = [], []

model.eval()

val_loss, val_acc = 0., 0.

for i, (feas,labels, mask,label_mask) in enumerate(tqdm(testLoader)):
    
    data = feas.to(device)
    labels = labels.type(torch.LongTensor).to(device=device)
    mask = mask.to(device)
    label_mask = label_mask.to(device)

    
    pred = model(data,mask)


    pred_soft = torch.argmax(pred.softmax(dim=-1),dim=-1)
    
    label_masks.extend([x.item() for x in label_mask.view(-1)])  
    label_s.extend([x.item() for x in labels.view(-1)])
    preds.extend([x.item() for x in pred.view(-1)])
    preds_soft.extend([x.item() for x in pred_soft.view(-1)])
    


df = pd.DataFrame({'prediction':preds})
fPred = 'result/' + args.output + '/' + args.output + '_pred.csv'
df.to_csv(fPred, index=False)


df = pd.DataFrame({'pred_soft':preds_soft, 'label':label_s, 'mask':label_masks})
flabel = 'result/' + args.output + '/' + args.output + '_label_mask.csv'
df.to_csv(flabel, index=False)