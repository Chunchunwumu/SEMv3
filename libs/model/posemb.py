import math
import torch 
import torch.nn as nn
from mmcv.cnn import ConvModule


class SinusoidalPositionalEmbedding(nn.Module):
    """This module produces sinusoidal positional embeddings of any length.

    Padding symbols are ignored.
    """

    def __init__(self, embedding_dim, init_size=1024):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.weights = SinusoidalPositionalEmbedding.get_embedding(
            init_size, embedding_dim
        )
        self.max_positions = int(1e5)

    def get_embedding(num_embeddings: int, embedding_dim: int):
        """Build sinusoidal embeddings.

        This matches the implementation in tensor2tensor, but differs slightly
        from the description in Section 3.5 of "Attention Is All You Need".
        """
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(
            1
        ) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(
            num_embeddings, -1
        )
        if embedding_dim % 2 == 1:
            # zero pad
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
        return emb

    def forward(
        self,
        positions
    ):
        self.weights = self.weights.to(positions.device)
        return (
            self.weights[positions.reshape(-1)]
            .view(positions.size() + (-1,))
            .detach()
        )


class SinusoidalPositionalConv(nn.Module):

    def __init__(self, in_channels,init_size=1024):
        super().__init__()
        self.pos_embed = SinusoidalPositionalEmbedding(in_channels//2,init_size=init_size)
        self.pos_conv = ConvModule(in_channels, in_channels, kernel_size=1, conv_cfg=None, act_cfg=None)
        self.norm = nn.LayerNorm(in_channels)

    def forward(self, feats):
        device = feats.device
        batch_size, channels, y_dim, x_dim = feats.size()

        xx_pos = torch.arange(x_dim).repeat(batch_size, y_dim, 1).to(device=device) # (B, H, W)
        yy_pos = torch.arange(y_dim).repeat(batch_size, x_dim, 1).transpose(1, 2).to(device=device) # (B, H, W)

        xx_pos_embeddings = self.pos_embed(xx_pos).permute(0, 3, 1, 2).contiguous() # (B, C/2, H, W)
        yy_pos_embeddings = self.pos_embed(yy_pos).permute(0, 3, 1, 2).contiguous() # (B, C/2, H, W)
        pos_embeddings = torch.cat([xx_pos_embeddings, yy_pos_embeddings], dim=1) # (B, C, H, W)
        pos_embeddings = self.pos_conv(pos_embeddings)

        feats = feats.permute(0,2,3,1).reshape(-1, channels).contiguous() # (N, C)
        pos_embeddings = pos_embeddings.permute(0,2,3,1).reshape(-1, channels).contiguous() # (N, C)
        feats = self.norm(feats + pos_embeddings) # (N, C) #layer norm 对通道维进行操作
        feats = feats.view(batch_size, y_dim, x_dim, channels).permute(0, 3, 1, 2).contiguous() # (B, C, H, W)
        return feats


class RoPE(torch.nn.Module):
    def __init__(self, inchannels):
        super(RoPE, self).__init__()
        self.inchannels = inchannels
        self.H_max = int(1024)
        # 先生成cos sin各自的位置
        
        self.rota_matrix = self.get_rota_matrix()

    def get_cosValues(self):
        # 一个位置的对角线上角度值
        i = torch.arange( 0, self.inchannels)//2
        theta_i = 10000**( -2*(i)/self.inchannels )
        # 生成每一个位置的矩阵
        theta_i_h = theta_i.repeat( self.H_max, 1 ).unsqueeze(dim=1)
        theta_i_h = theta_i_h*torch.arange(self.H_max).reshape(self.H_max,1,1)
        cosValues = torch.cos( theta_i_h )
        return cosValues

    def get_sinValues(self):
        # 一个位置的对角线上角度值
        i = torch.arange( 0, self.inchannels)//2 
        theta_i = 10000**( -2*(i)/self.inchannels )
        # 生成每一个位置的矩阵
        theta_i_h = theta_i.repeat( self.H_max, 1 ).unsqueeze(dim=1)
        theta_i_h = theta_i_h*torch.arange(self.H_max).reshape(self.H_max,1,1)

        sinValues = torch.sin( theta_i_h )
        # print(sinValues)
        pn = ((-1)**(torch.arange( 0, self.inchannels))).repeat( self.H_max, 1 ).unsqueeze(dim=1)
        # print(pn)
        sinValues = sinValues*pn
        return sinValues
    
    def get_rota_matrix(self):
        cos_position = torch.arange( 0, self.inchannels ).repeat(self.H_max, 1).unsqueeze(dim=1)
        cos_value = self.get_cosValues()
        # print(cos_value)

        cos_dim = 1
        sin_position_temp = torch.arange( 0, self.inchannels ) + (-1)**(torch.arange( 0, self.inchannels ))
        # print(sin_position_temp)
        sin_position  = sin_position_temp.repeat(self.H_max, 1).unsqueeze(dim=1)
        # print(sin_position)
        sin_value = self.get_sinValues()
        sin_dim = 1
        rota_matrix = torch.zeros((self.H_max, self.inchannels, self.inchannels))

        rota_matrix.scatter_( cos_dim, cos_position, cos_value )
        # print(rota_matrix)
        rota_matrix.scatter_( sin_dim, sin_position, sin_value )
        return rota_matrix

    # def forward(self,feats, line_type):
    #     device = feats.device
    #     if line_type=="row":
    #         rota_matrix = self.rota_matrix[:feats.shape[2]].clone().detach().to(device=device)
    #         feats = feats.permute(( 0,3,2,1 )).unsqueeze(dim=-1)
    #         result =  torch.matmul( rota_matrix, feats ).squeeze(dim=-1)
    #         result = result.permute( (0, 3, 2, 1) )

    #     else:
    #         rota_matrix = self.rota_matrix[:feats.shape[3]].clone().detach().to(device=device)
    #         feats = feats.permute(( 0,2,3,1 )).unsqueeze(dim=-1)
    #         result =  torch.matmul( rota_matrix, feats ).squeeze(dim=-1)
    #         result = result.permute( (0, 3, 1, 2) )
    #     return result
    def forward(self,feats, line_type):
        device = feats.device
        if line_type=="row":
            rota_matrix = self.rota_matrix[:feats.shape[2]].clone().detach().to(device=device)
            return torch.einsum( "hvc,bchw->bvhw", rota_matrix, feats )
        else:
            rota_matrix = self.rota_matrix[:feats.shape[3]].clone().detach().to(device=device)
            return torch.einsum( "wvc,bchw->bvhw", rota_matrix, feats )

def build_posemb_head(cfg):
    posemb_head = SinusoidalPositionalConv(cfg['in_channels'],cfg["init_size"])
    return posemb_head

def build_RoPE(inchannels):
    return RoPE(inchannels)