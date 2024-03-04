#!/usr/bin/env python
import argparse
import uproot
import numpy as np
from tqdm import tqdm
import torch
from torch_geometric.data import Data
import os


parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', required=True, type=str)
parser.add_argument('-o', '--output', required=True, type=str)
args = parser.parse_args()

file_list = os.listdir(args.input)


for files in tqdm(range(len(file_list))):
    
    f = uproot.open(args.input+file_list[files])
    data_list=[]
    for k in tqdm(range(len(np.array(f['tree']['jet_pt'])))):

        ## feature
        jet_pt = torch.from_numpy(np.array(f['tree']['jet_pt'])[k]).view(-1,1)
        jet_eta = torch.from_numpy(np.array(f['tree']['jet_eta'])[k]).view(-1,1)
        jet_phi = torch.from_numpy(np.array(f['tree']['jet_phi'])[k]).view(-1,1)
        jet_mass = torch.from_numpy(np.array(f['tree']['jet_mass'])[k]).view(-1,1)
        jet_btag = torch.from_numpy(np.array(f['tree']['jet_btag'])[k].astype(np.float32)).view(-1,1)

        ## jet_feature shape [N-Jet,5]
        jet_feature = torch.cat((jet_pt,
                                jet_eta,
                                jet_phi,
                                jet_mass,
                                jet_btag),dim=1)

        
        ## label [other, b_m, q0, q1, b_p, q2, q3]
        ## label [0, 1, 2, 2, 3, 4, 4]
        ## jet_label shape {N-Jet}
        jet_parton_match = np.array(f['tree']['jet_parton_match'])[k]+1

        ## find index
        index_3 = np.where(jet_parton_match==3)
        index_4 = np.where(jet_parton_match==4)
        index_5 = np.where(jet_parton_match==5)
        index_6 = np.where(jet_parton_match==6)

        ## change index
        jet_parton_match[index_3] = 2
        jet_parton_match[index_4] = 3
        jet_parton_match[index_5] = 4
        jet_parton_match[index_6] = 4
        
        
        jet_label = torch.from_numpy(jet_parton_match)
        data = Data(x = jet_feature, y = jet_label)
        data_list.append(data)
    output_file = args.output + 'ttbar_'+file_list[files].split('.')[0].split('_')[1]+'.pt'
    torch.save(data_list,output_file)

