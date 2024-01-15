import torch.nn as nn
import torch

class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, hidden_dim, n_heads, dropout_ratio, device):
        super().__init__()

        assert hidden_dim % n_heads == 0
                
        # imbedding dimension
        self.hidden_dim = hidden_dim
        
        # head num
        self.n_heads = n_heads
        
                # each head imbedding dimension
        self.head_dim = hidden_dim // n_heads 
                
        # before scaled dot-product attention FC
        self.fc_q = nn.Linear(hidden_dim, hidden_dim)
        self.fc_k = nn.Linear(hidden_dim, hidden_dim)
        self.fc_v = nn.Linear(hidden_dim, hidden_dim)


        self.fc_o = nn.Linear(hidden_dim, hidden_dim)
        self.relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(dropout_ratio)

        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)
        
    def forward(self, query, key, value):
#     def forward(self, query, key, value,mask=None):

        batch_size = query.shape[0]
        
        # initial dimention
        # Q(K,V) : [batch_size, Q_len(K,V), hidden_dim]
        # Q_len : PMT number
        # print(query.shape)
        Q = self.fc_q(query)
        K = self.fc_k(key)
        V = self.fc_v(value)
        
        # after FC dimension
        # Q(K,V) : [batch_size, Q_len(K,V), hidden_dim]

        # hidden_dim → n_heads X head_dim  reshape
        # n_heads(h) different head training
        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        # print(Q.shape)
        # Q(K,V) : [batch_size, n_heads, Q_len(K,V), head_dim]

        # each head calculate Q and K -> divide scale 
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale

        # energy: [batch_size, n_heads, Q_len(K,V), key_len]
        
        # 마스크(mask)를 사용하는 경우
#         if mask is not None:
#             # 마스크(mask) 값이 0인 부분을 -1e10으로 채우기
#             energy = energy.masked_fill(mask==0, -1e10)

        # attention score
        attention = torch.softmax(energy, dim=-1)

        # attention: [batch_size, n_heads, Q_len(K,V), key_len]

        # attention X Value
        x = torch.matmul(self.dropout(attention), V)
        # print(attention)

        
        #### Concat all head
        # x: [batch_size, n_heads, Q_len(K,V), head_dim]

        x = x.permute(0, 2, 1, 3).contiguous()

        # x: [batch_size, Q_len(K,V), n_heads, head_dim]

        x = x.view(batch_size, -1, self.hidden_dim)

        # x: [batch_size, Q_len(K,V), hidden_dim]
        #### Concat all head
        
        x = self.fc_o(x)

        # x: [batch_size, Q_len(K,V), hidden_dim]

        return x, attention

    
class PositionwiseFeedforwardLayer(nn.Module):
    def __init__(self, hidden_dim, pf_dim, dropout_ratio):
        super().__init__()
        
        # hidden_dim input -> hidden_dim output
        self.fc_1 = nn.Linear(hidden_dim, pf_dim)
        self.fc_2 = nn.Linear(pf_dim, hidden_dim)

        self.dropout = nn.Dropout(dropout_ratio)
        self.relu = nn.LeakyReLU()
    def forward(self, x):

        # x: [batch_size, seq_len, hidden_dim]

        x = self.dropout(self.relu(self.fc_1(x)))
        # x: [batch_size, seq_len, pf_dim]

        x = self.relu(self.fc_2(x))
        
#         x = self.dropout(torch.relu(self.fc_1(x)))## origin
#         x = self.fc_2(x)  ###origin
        # x: [batch_size, seq_len, hidden_dim]

        return x
    
    

class EncoderLayer(nn.Module):
    def __init__(self, hidden_dim, n_heads, pf_dim, dropout_ratio,device):
        super().__init__()

        
        
        self.self_attention = MultiHeadAttentionLayer(hidden_dim, n_heads, dropout_ratio,device)
#         self.self_attn_layer_norm = nn.BatchNorm1d(14)
        self.self_attn_layer_norm = nn.LayerNorm(hidden_dim) ## origin
        
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hidden_dim, pf_dim, dropout_ratio)
#         self.ff_layer_norm = nn.BatchNorm1d(14)
        self.ff_layer_norm = nn.LayerNorm(hidden_dim) ## origin
        
        self.dropout = nn.Dropout(dropout_ratio)
        
    def forward(self, src):
#     def forward(self, src,src_mask): ##origin

        _src = self.self_attn_layer_norm(src)
        _src, _ = self.self_attention(_src, _src, _src)
        src = self.dropout(_src)+src


        _src = self.ff_layer_norm(src)
        _src = self.positionwise_feedforward(_src)
        src = src + self.dropout(_src)

###############################################################################################
# ##########  origin ############################################################################
#         _src, _ = self.self_attention(src, src, src, src_mask)

#         # dropout, residual connection and layer norm
#         src = self.self_attn_layer_norm(src + self.dropout(_src))

#         # src: [batch_size, src_len, hidden_dim]

#         # position-wise feedforward
#         _src = self.positionwise_feedforward(src)

#         # dropout, residual and layer norm
#         src = self.ff_layer_norm(src + self.dropout(_src))


        return src





class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers, n_heads, pf_dim, dropout_ratio, device):
        super().__init__()

        

        self.embedding_1 = nn.Linear(input_dim, hidden_dim)
        self.embedding_2 = nn.Linear(hidden_dim, hidden_dim)

        self.layers = nn.ModuleList([EncoderLayer(hidden_dim, n_heads, pf_dim, dropout_ratio,device) for _ in range(n_layers)])

        self.dropout = nn.Dropout(dropout_ratio)
        self.relu = nn.LeakyReLU()
        self.scale = torch.sqrt(torch.FloatTensor([hidden_dim])).to(device)
        self.device = device
    def forward(self, src):

        # src: [batch_size, src_len]
        # src_mask: [batch_size, src_len]

        batch_size = src.shape[0]
        src_len = src.shape[1]

        # pos = src
        # pos: [batch_size, src_len]
        # 소스 문장의 임베딩과 위치 임베딩을 더한 것을 사용
        src = self.dropout(self.relu(self.embedding_1(src)))
        src = self.relu(self.embedding_2(src))
        
        
        # src: [batch_size, src_len, hidden_dim]

        # 모든 인코더 레이어를 차례대로 거치면서 순전파(forward) 수행
        for layer in self.layers:
            src = layer(src, src_mask) ##origin
#             src = layer(src)

        # src: [batch_size, src_len, hidden_dim]

        return src
