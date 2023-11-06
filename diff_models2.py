from sklearn.utils.validation import check_non_negative
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from S4Model import S4Layer
from spatial_conv import SpatialConvOrderK

from gcrnn import GCGRUCell



def get_torch_trans(heads=8, layers=1, channels=64):
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=channels, nhead=heads, dim_feedforward=64, activation="gelu"
    )
    return nn.TransformerEncoder(encoder_layer, num_layers=layers)


def Conv1d_with_init(in_channels, out_channels, kernel_size):
    layer = nn.Conv1d(in_channels, out_channels, kernel_size)
    nn.init.kaiming_normal_(layer.weight)
    return layer


class DiffusionEmbedding(nn.Module):
    def __init__(self, num_steps, embedding_dim=128, projection_dim=None):
        super().__init__()
        if projection_dim is None:
            projection_dim = embedding_dim
        self.register_buffer(
            "embedding",
            self._build_embedding(num_steps, embedding_dim / 2),
            persistent=False,
        )
        self.projection1 = nn.Linear(embedding_dim, projection_dim)
        self.projection2 = nn.Linear(projection_dim, projection_dim)

    def forward(self, diffusion_step):
        x = self.embedding[diffusion_step]
        x = self.projection1(x)
        x = F.silu(x)
        x = self.projection2(x)
        x = F.silu(x)
        return x

    def _build_embedding(self, num_steps, dim=64):
        steps = torch.arange(num_steps).unsqueeze(1)  # (T,1)
        frequencies = 10.0 ** (torch.arange(dim) / (dim - 1) * 4.0).unsqueeze(0)  # (1,dim)
        table = steps * frequencies  # (T,dim)
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)  # (T,dim*2)
        return table


class diff_CSDI(nn.Module):
    def __init__(self, config, inputdim=2):
        super().__init__()
        self.channels = config["channels"]

        self.diffusion_embedding = DiffusionEmbedding(
            num_steps=config["num_steps"],
            embedding_dim=config["diffusion_embedding_dim"],
        )

        self.input_projection = Conv1d_with_init(inputdim, self.channels, 1)
        self.output_projection1 = Conv1d_with_init(self.channels, self.channels, 1)
        self.output_projection2 = Conv1d_with_init(self.channels, 1, 1)
        self.spa_conv = SpatialConvOrderK(1, 1)
        '''self.S4 = S4Layer(features=self.channels,
                          lmax=36,
                          N=64,
                          dropout=0.0,
                          bidirectional=1,
                          layer_norm=1)'''

        nn.init.zeros_(self.output_projection2.weight)

        self.residual_layers = nn.ModuleList(
            [
                ResidualBlock(
                    side_dim=config["side_dim"],
                    channels=self.channels,
                    diffusion_embedding_dim=config["diffusion_embedding_dim"],
                    nheads=config["nheads"],
                )
                for _ in range(config["layers"])
            ]
        )

        self.time_layer = get_torch_trans(heads=config["nheads"], layers=1, channels=self.channels)
        self.feature_layer = get_torch_trans(heads=config["nheads"], layers=1, channels=self.channels)

    def forward_time(self, y, base_shape):
        B, channel, K, L = base_shape
        if L == 1:
            return y
        y = y.reshape(B, channel, K, L).permute(0, 2, 1, 3).reshape(B * K, channel, L)
        y = self.time_layer(y.permute(2, 0, 1)).permute(1, 2, 0)
        y = y.reshape(B, K, channel, L).permute(0, 2, 1, 3).reshape(B, channel, K * L)
        return y

    def forward_feature(self, y, base_shape):
        B, channel, K, L = base_shape
        if K == 1:
            return y
        y = y.reshape(B, channel, K, L).permute(0, 3, 1, 2).reshape(B * L, channel, K)
        y = self.feature_layer(y.permute(2, 0, 1)).permute(1, 2, 0)
        y = y.reshape(B, L, channel, K).permute(0, 2, 3, 1).reshape(B, channel, K * L)
        return y

    def forward(self, x, cond_info, diffusion_step, observed_data, cond_mask, adj, k=2):
        B, inputdim, K, L = x.shape
        base_shape = B, self.channels, K, L

        x = x.reshape(B, inputdim, K * L)
        x = self.input_projection(x)
        x = F.relu(x)
        #x = x.reshape(B, self.channels, K, L)

        #x_c = self.forward_time(x, base_shape) # (B,channel,K*L)
        #x_c = self.forward_feature(x_c, base_shape) # (B,channel,K*L)
        #x_c = x_c.reshape(B, self.channels, K, L).permute(0,3,2,1)
        #x = x.reshape(B, self.channels, K, L).permute(0,3,2,1) #B L K C

        #observed_mask: B K L
        #cond_mask: B K L
        #cond_mask = cond_mask.permute(0,2,1).unsqueeze(-1).repeat(1,1,1,self.channels)
        #x = x * cond_mask #这里x是否需要乘以cond mask存疑
        #x_c = x_c * (1 - cond_mask)
        #x = x + x_c
        #x = x.permute(0,3,2,1)  #B C K L
        #x = self.forward_time(x, base_shape) # (B,channel,K*L)
        #x = self.forward_feature(x, base_shape) # (B,channel,K*L)
        x = x.reshape(B, self.channels, K, L)

        diffusion_emb = self.diffusion_embedding(diffusion_step)

        support = self.spa_conv.compute_support_orderK(adj, k)
        #observed_data = torch.unsqueeze(observed_data, 1)
        #cond_mask = torch.unsqueeze(cond_mask, 1)
        #spa_x = self.spa_conv(observed_data * cond_mask, support) #B, 1, K, L
        #spa_x = spa_x.reshape(B, 1, K*L)
        spa_x = 0
    
        skip = []
        for layer in self.residual_layers:
            x, skip_connection = layer(x, cond_info, diffusion_emb, cond_mask, spa_x, support)
            skip.append(skip_connection)

        x = torch.sum(torch.stack(skip), dim=0) / math.sqrt(len(self.residual_layers))
        x = x.reshape(B, self.channels, K * L)
        x = self.output_projection1(x)  # (B,channel,K*L)
        #last S4 2023-2-27
        '''x = x.reshape(B, self.channels, K, L).permute(0, 2, 1, 3).reshape(B * K, self.channels, L)
        x = self.S4(x.permute(2, 0, 1)).permute(1, 2, 0)
        x = x.reshape(B, K, self.channels, L).permute(0, 2, 1, 3).reshape(B, self.channels, K * L)'''
        #last S4
        x = F.relu(x)
        x = self.output_projection2(x)  # (B,1,K*L)
        x = x.reshape(B, K, L)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, side_dim, channels, diffusion_embedding_dim, nheads):
        super().__init__()
        self.diffusion_projection = nn.Linear(diffusion_embedding_dim, channels)
        self.cond_projection = Conv1d_with_init(side_dim, 2 * channels, 1)
        self.spa_cond_projection = Conv1d_with_init(1, 2 * channels, 1) ##2023-2-17
        self.spa_conv = SpatialConvOrderK(channels, channels)
        #self.replace_conv = Conv1d_with_init(channels, channels, 1)
        self.afternn_conv = Conv1d_with_init(channels, channels, 1)
        self.mid_projection = Conv1d_with_init(channels, 2 * channels, 1)
        self.output_projection = Conv1d_with_init(channels, 2 * channels, 1)
        self.S41 = S4Layer(features=channels,
                          lmax=200,
                          N=64,
                          dropout=0.0,
                          bidirectional=1,
                          layer_norm=1)

        self.S42 = S4Layer(features=2 * channels,
                          lmax=200,
                          N=64,
                          dropout=0.0,
                          bidirectional=1,
                          layer_norm=1)
        
        self.gcrnn = nn.ModuleList()
        for i in range(1):
            self.gcrnn.append(GCGRUCell(d_in=channels + 1, num_units=channels, support_len=4, order=2))
        self.time_layer = get_torch_trans(heads=nheads, layers=1, channels=channels)
        #self.gatrnn = nn.ModuleList()
        #for i in range(1):
        #    self.gatrnn.append(GATRNN_cell(d_in=channels, num_units=channels))
        self.feature_layer = get_torch_trans(heads=nheads, layers=1, channels=channels)
        
        self.gcn = SpatialConvOrderK(c_in = channels, c_out = channels, support_len=4, order=2)
        #self.middle_attention_layer = get_torch_trans(heads=nheads, layer=1, channels=2*channels)

    def forward_time(self, y, base_shape):
        B, channel, K, L = base_shape
        if L == 1:
            return y
        y = y.reshape(B, channel, K, L).permute(0, 2, 1, 3).reshape(B * K, channel, L)
        y = self.time_layer(y.permute(2, 0, 1)).permute(1, 2, 0)
        y = y.reshape(B, K, channel, L).permute(0, 2, 1, 3).reshape(B, channel, K * L)
        return y

    def forward_feature(self, y, base_shape):
        B, channel, K, L = base_shape
        if K == 1:
            return y
        y = y.reshape(B, channel, K, L).permute(0, 3, 1, 2).reshape(B * L, channel, K)
        y = self.feature_layer(y.permute(2, 0, 1)).permute(1, 2, 0)
        y = y.reshape(B, L, channel, K).permute(0, 2, 3, 1).reshape(B, channel, K * L)
        return y

    def update_state(self, x, h, adj, m_in):
        rnn_in = x
        for layer, cell in enumerate(self.gcrnn):
            rnn_in = h[layer] = cell(rnn_in, h[layer], adj, m_in)
            #if self.dropout is not None and layer < (self.n_layers - 1):
                #rnn_in = self.dropout(rnn_in)
        return rnn_in, h

    def init_hidden_states(self, x):
        return [torch.zeros(size=(x.shape[0], x.shape[1], x.shape[2])).to(x.device) for _ in range(1)]

    def forward(self, x, cond_info, diffusion_emb, cond_mask ,spa_x, support):
        #observed_mask: B K L
        #cond_mask: B K L

        B, channel, K, L = x.shape
        base_shape = x.shape
        x = x.reshape(B, channel, K * L)       

        #first impute


        diffusion_emb = self.diffusion_projection(diffusion_emb).unsqueeze(-1)  # (B,channel,1)
        y = x + diffusion_emb   #(B, channel, K * L)

        '''
        #renormalized
        #cond_mask:B, K, L
        #A_m:K, K
        cond_mask = cond_mask.permute(0,2,1).unsqueeze(-2).repeat(1,1,K,1)  #B L K K
        A_m = A_m.unsqueeze(0).unsqueeze(1).repeat(B, L, 1, 1)  #B L K K
        total_mask = cond_mask * A_m
        total_mask = total_mask.sum(-1) #B L K
        A_m = A_m.sum(-1)
        
        #print(A_m[0], total_mask[0])
        renormal = A_m/(total_mask + 1e-5) #B L K
        renormal = renormal.unsqueeze(-1).repeat(1,1,1,K).permute(0,2,3,1) # B K K L
        #print(renormal[0])'''


        '''y = y.reshape(B, channel, K, L)
        A_m = 
        

        
        renormal = cond_mask.sum(-2).unsqueeze(1).unsqueeze(2)/K   #
        renormal = renormal.repeat(1, channel, K, 1)
        y /= renormal'''
        


        #spatial
        #y = x.reshape(B, channel, K, L)
        #y = self.spa_conv(y, support).reshape(B, channel, K*L)

        

        y = self.forward_time(y, base_shape)
        #y = y.reshape(B, channel, K, L)
        #y = self.gat(y, adj).reshape(B, channel, K*L)
        y = self.forward_feature(y, base_shape)
        #y = self.gcn(y, support, cond_mask)

        #S4 1
        y = y.reshape(B, channel, K, L).permute(0, 2, 1, 3).reshape(B * K, channel, L)
        y = self.S41(y.permute(2, 0, 1)).permute(1, 2, 0)
        y = y.reshape(B, K, channel, L).permute(0, 2, 1, 3).reshape(B, channel, K * L)
        #y = self.replace_conv(y.reshape(B, channel, K*L))

        #GAT
        #y = y.reshape(B, channel, K, L)
        #y = self.GAT(y, adj).reshape(B, channel, K*L)

        #y = self.forward_feature(y, base_shape)
        #spatial
        y = y.reshape(B, channel, K, L)
        #y = self.spa_conv(y, support).reshape(B, channel, K*L)
        
        rnn_hidden = self.init_hidden_states(y)
        rnn_out = []
        #GCRNN
        for step in range(L):
            m_s = cond_mask[...,step].unsqueeze(1)
            y_s = y[..., step]
            m_in = cond_mask[...,step].unsqueeze(1).repeat(1,K,1).unsqueeze(-1)
            y_in = torch.cat([y_s, m_s], dim=1)
            out, rnn_hidden = self.update_state(y_in, rnn_hidden, support, m_in)
            rnn_out.append(out)
            
        y = torch.stack(rnn_out, dim=-1)

        y = self.afternn_conv(y.reshape(B, channel, K*L))

        #y = self.forward_feature(y, base_shape)  # (B,channel,K*L)
        y = self.mid_projection(y)  # (B,2*channel,K*L)

        _, cond_dim, _, _ = cond_info.shape
        cond_info = cond_info.reshape(B, cond_dim, K * L)
        cond_info = self.cond_projection(cond_info)  # (B,2*channel,K*L)

        #spa_x = spa_x.reshape(B, 1, K, L).reshape(B, 1, K*L)
        #spa_x = self.spa_cond_projection(spa_x) # (B,2*channel,K*L)

        #y = torch.cat([y.reshape(B, 2*channel, K, L), cond_info.reshape(B, 2*channel, K, L), spa_x.reshape(B, 2*channel, K, L)], dim=2)
        #base_shape2 = y.shape
        #y = self.forward_feature(y, base_shape2) # (B, 2 * channel, 3K, L)
        #y, cond_info, spa_x = torch.chunk(y, 3, dim=2)

        y = y + cond_info# + spa_x
        
        y = y.reshape(B, 2 * channel, K, L).permute(0, 2, 1, 3).reshape(B * K, 2 * channel, L)
        y = self.S42(y.permute(2, 0, 1)).permute(1, 2, 0)
        y = y.reshape(B, K, 2 * channel, L).permute(0, 2, 1, 3).reshape(B, 2 * channel, K * L)

        gate, filter = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter)  # (B,channel,K*L)
        y = self.output_projection(y)

        residual, skip = torch.chunk(y, 2, dim=1)
        x = x.reshape(base_shape)
        residual = residual.reshape(base_shape)
        skip = skip.reshape(base_shape)
        return (x + residual) / math.sqrt(2.0), skip
