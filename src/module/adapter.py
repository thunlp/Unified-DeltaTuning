import torch
from torch import nn
from transformers.activations import get_activation


# class Adapter(nn.Module):
#     def __init__(self, dim, r, act):
#         super().__init__()
#         self.adapter_A = nn.Linear(dim, r, bias=False)
#         # 改为nn.parameter形式？
#         self.act = get_activation(act)
#         self.adapter_B = nn.Linear(r, dim, bias=False)

#     # def reset_parameters():
    
#     def forward(self, x, residual, share_intrinsic=None):
#         result = self.adapter_A(x)
#         result = self.act(result)
#         result = self.adapter_B(result)
#         return result + residual

class Adapter(nn.Module):
    def __init__(self, dim, r, act):
        super().__init__()
        self.adapter_A = nn.Parameter(torch.zeros(dim, r))
        # 改为nn.parameter形式？
        self.act = get_activation(act)
        self.adapter_B = nn.Parameter(torch.zeros(r, dim))
        self.reset_parameters()

    def reset_parameters(self, r_mean=0, r_std=0.02):
        self.adapter_A.data.normal_(mean=r_mean, std=r_std)
        self.adapter_B.data.normal_(mean=r_mean, std=r_std)
    
    
    def forward(self, x, residual, flatten_pet=None):
        if flatten_pet!=None:
            assert self.adapter_A.size()[0]*self.adapter_A.size()[1]*2 == int(flatten_pet.size()[0])
        if flatten_pet!=None:
            result = x @ flatten_pet[0:int(flatten_pet.size()[0]/2)].view(self.adapter_A.size())
            result = self.act(result)
            result = result @ flatten_pet[int(flatten_pet.size()[0]/2):].view(self.adapter_B.size())
        else:
            result = x @ self.adapter_A
            result = self.act(result)
            result = result @ self.adapter_B
        return result + residual


class IntrinsicAdapter(nn.Module):
    def __init__(self, intrinsic_dim, dim, r, act, share_intrinsic=None):
        super().__init__()
        self.intrinsic_dim = intrinsic_dim
        self.dim = dim
        self.r = r
        # self.hyper_adapter_A = nn.Parameter(torch.zeros(dim, intrinsic_dim, r))
        self.hyper_adapter_A = nn.Parameter(torch.zeros(intrinsic_dim, dim*r))
        self.act = get_activation(act)
        # self.hyper_adapter_B = nn.Parameter(torch.zeros(r, intrinsic_dim, dim))
        self.hyper_adapter_B = nn.Parameter(torch.zeros(intrinsic_dim, r*dim))
        self.reset_parameters()
        self.share_intrinsic = share_intrinsic
        """
        self.adapter_A = nn.Linear(dim, r, bias=False)
        # 改为nn.parameter形式？
        self.act = get_activation(act)
        self.adapter_B = nn.Linear(r, dim, bias=False)
        """
        

    def reset_parameters(self):
        self.hyper_adapter_A.data.normal_(mean=0.0, std=0.02)
        self.hyper_adapter_B.data.normal_(mean=0.0, std=0.02)
    
    def forward(self, x, residual):
        # adapter_A = (self.share_intrinsic.T @ self.hyper_adapter_A).squeeze()
        adapter_A = (self.share_intrinsic.T @ self.hyper_adapter_A).view(self.dim, self.r)
        result = x @ adapter_A
        result = self.act(result)
        # adapter_B = (self.share_intrinsic.T @ self.hyper_adapter_B).squeeze()
        adapter_B = (self.share_intrinsic.T @ self.hyper_adapter_B).view(self.r, self.dim)
        result = result @ adapter_B
        return result + residual
