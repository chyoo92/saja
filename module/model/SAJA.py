# import torch_geometric.nn as PyG
# from torch_geometric.transforms import Distance
# from torch_geometric.data import Data as PyGData
import torch.nn as nn
import numpy as np
import torch
from torch.nn import Linear


from model.multihead_encoder import *

class SAJA(nn.Module):
    def __init__(self,**kwargs):
        super(SAJA, self).__init__()

        self.fea = kwargs['fea']
        self.cla = kwargs['cla']
        self.hidden_dim = kwargs['hidden']
        self.n_layers = kwargs['depths']
        self.n_heads = kwargs['heads']
        self.pf_dim = kwargs['posfeed']
        self.dropout_ratio = kwargs['dropout']
        self.device = kwargs['device']
        self.batch = kwargs['batch']
        
        self.dim_model = self.hidden_dim*self.n_heads
        
        self.embedding_1 = nn.Linear(self.fea, self.pf_dim)
        self.embedding_2 = nn.Linear(self.pf_dim, self.dim_model)
        
        self.multihead_attn =  nn.MultiheadAttention(self.dim_model, self.n_heads, batch_first=True)
        
        self.encoderlayer = torch.nn.TransformerEncoderLayer(d_model=self.dim_model, nhead = self.n_heads, dim_feedforward = self.pf_dim, dropout = self.dropout_ratio, activation = "relu",batch_first=True,norm_first=True)
        
        self.encoder = torch.nn.TransformerEncoder(self.encoderlayer, num_layers=self.n_layers)
#         self.encoder = Encoder(self.fea, self.hidden_dim, self.n_layers, self.n_heads, self.pf_dim, self.dropout_ratio, self.device)


        self.embedding_3 = nn.Linear(self.dim_model, self.pf_dim)
        self.embedding_4 = nn.Linear(self.pf_dim, self.cla)


        self.relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(self.dropout_ratio)        
    def forward(self, data,mask):
        
        src, src_mask = data, mask

        src = self.dropout(self.relu(self.embedding_1(src)))
     
        src = self.relu(self.embedding_2(src))

        attn_mask_pre = torch.matmul(src_mask,src_mask.contiguous().permute(0,2,1))
        attn_mask_pre[attn_mask_pre==0]=float('-inf')
        attn_mask_pre[attn_mask_pre!=0]=0
        attn_mask = attn_mask_pre.repeat(self.n_heads,1,1)
        
                
        src_mask[src_mask==0]=float('-inf')
        src_mask[src_mask!=0]=0


#         out, weights = self.multihead_attn(Q, K, V, attn_mask= attn_mask, key_padding_mask = src_mask[:,:,0])

#         out = self.encoderlayer(src,attn_mask,src_mask[:,:,0])
    
        out = self.encoder(src,attn_mask,src_mask[:,:,0])
        
        out = self.dropout(self.relu(self.embedding_3(out)))
        out = self.relu(self.embedding_4(out))

        
#         out = torch.reshape(out, (-1,14*self.dim_model))
#         out = self.mlp(out)
#         out = torch.reshape(out, (-1,14, self.cla))


            
        return out
