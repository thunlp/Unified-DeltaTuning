import os
import torch
import torch.nn.functional as F
from torch import Tensor, nn
import numpy as np
import random
# from modeling_t5_multiHyper import T5PreTrainedModel, T5ForConditionalGeneration
from modeling_t5_multiHyper_flatten_pet import T5PreTrainedModel, T5ForConditionalGeneration

class MyT5_pet_MC(nn.Module):
    def __init__(self, args, config):
        super().__init__()
        self.args = args
        self.config = config
        
        self.flatten_size = 1105920
        self.low_dimension = args.low_dimension
        def flatten_init(ckpt_path, pet_type):
            if os.path.exists(ckpt_path):
                ckpt = torch.load(ckpt_path)
            else:
                assert False, f'{ckpt_path} does not exist!'
            if pet_type in ckpt:
                pet_dict = ckpt[pet_type]
            else:
                pet_dict = ckpt
            pet_name_modules = list(pet_dict.keys())
            flatten = torch.Tensor([]).cuda()
            for pet_name_module in pet_name_modules:
                flatten = torch.cat((flatten, pet_dict[pet_name_module].flatten().cuda()),dim=0)
            return flatten
        
        self.init_A = flatten_init('init_pet/adapter_init_seed_42.pth', 'adapter')
        self.init_L = flatten_init('init_pet/lora_init_seed_42.pth', 'lora')
        self.init_P = flatten_init('init_pet/prefix_init_seed_42.pth', 'prefix')
        self.share_intrinsic = nn.Parameter(torch.zeros(self.low_dimension))
        self.share_intrinsic.data.normal_(mean=config.r_mean, std=config.r_std)
        
        self.encoder_adapter = nn.Linear(self.flatten_size, self.low_dimension,)
        self.encoder_adapter_low = nn.Linear(self.low_dimension, self.low_dimension)
        self.decoder_adapter = nn.Linear(self.low_dimension, self.flatten_size, bias=False)
        self.decoder_adapter_low = nn.Linear(self.low_dimension, self.low_dimension, bias=False)
        self.encoder_lora = nn.Linear(self.flatten_size, self.low_dimension,)
        self.encoder_lora_low = nn.Linear(self.low_dimension, self.low_dimension)
        self.decoder_lora = nn.Linear(self.low_dimension, self.flatten_size, bias=False)
        self.decoder_lora_low = nn.Linear(self.low_dimension, self.low_dimension, bias=False)
        self.encoder_prefix = nn.Linear(self.flatten_size, self.low_dimension,)
        self.encoder_prefix_low = nn.Linear(self.low_dimension, self.low_dimension)
        self.decoder_prefix = nn.Linear(self.low_dimension, self.flatten_size, bias=False)
        self.decoder_prefix_low = nn.Linear(self.low_dimension, self.low_dimension, bias=False)
        self.model_AL = T5ForConditionalGeneration.from_pretrained(args.model, config=self.config) #网络结构同时存在，但是不同时forwad
        self.init_weight()
        # self.model_L = T5ForConditionalGeneration.from_pretrained()
    
    def init_weight(self):
        for k,v in self.state_dict().items():
            if 'encoder_' in k or 'decoder_' in k:
                eval('self.'+k).data.normal_(mean=0.0, std=0.02)
    
    def faltten(self, ckpt_path_list, pet_type):
        def faltten_from_path(ckpt_path_list, pet_type):
            flatten_all = []
            for ckpt_path in ckpt_path_list:
                ckpt = torch.load(ckpt_path)
                if pet_type in ckpt:
                    pet_dict = ckpt[pet_type]
                else:
                    pet_dict = ckpt
                pet_name_modules = list(pet_dict.keys())
                flatten = torch.Tensor([]).cuda()
                for pet_name_module in pet_name_modules:
                    flatten = torch.cat((flatten, pet_dict[pet_name_module].flatten().cuda()),dim=0)
                flatten_all.append(flatten)
            flatten_all = torch.stack(flatten_all,dim=0)
            return flatten_all, len(pet_name_modules)
        
        if pet_type=='adapter':
            flatten_adapter_all, num_module = faltten_from_path(ckpt_path_list, pet_type)
            assert num_module==120, f"Num of {pet_type} modules should be 120!"
            return flatten_adapter_all
        if pet_type=='lora':
            flatten_lora_all, num_module = faltten_from_path(ckpt_path_list, pet_type)
            assert num_module==144, f"Num of {pet_type} modules should be 144!"
            return flatten_lora_all
        if pet_type=='prefix':
            flatten_prefix_all, num_module = faltten_from_path(ckpt_path_list, pet_type)
            assert num_module==6, f"Num of {pet_type} modules should be 6!"
            return flatten_prefix_all
        
    def get_low_dim_P(self, paras, only_adapter=False, only_lora=False, only_prefix=False):
        if only_adapter:
            H_A = self.encoder_adapter(paras)
            H_A_nonlinear = torch.tanh(H_A)
            P_A = self.encoder_adapter_low(H_A_nonlinear)
            return P_A
        if only_lora:
            H_L = self.encoder_lora(paras)
            H_L_nonlinear = torch.tanh(H_L)
            P_L = self.encoder_lora_low(H_L_nonlinear)
            return P_L
        if only_prefix:
            H_P = self.encoder_prefix(paras)
            H_P_nonlinear = torch.tanh(H_P)
            P_P = self.encoder_prefix_low(H_P_nonlinear)
            return P_P
    
    def get_high_dim_H_flattened(self, P, only_adapter=False, only_lora=False, only_prefix=False):
        if only_adapter:
            
            H_A_inverse_delta = self.decoder_adapter(P)
            H_A_inverse = H_A_inverse_delta+self.init_A
            return H_A_inverse
        if only_lora:
           
            H_L_inverse_delta = self.decoder_lora(P)
            H_L_inverse = H_L_inverse_delta+self.init_L
            return H_L_inverse
        if only_prefix:
            
            H_P_inverse_delta = self.decoder_prefix(P)
            H_P_inverse = H_P_inverse_delta+self.init_P
            return H_P_inverse
            
    def forward(self, all_input, only_adapter=False, only_lora=False, only_prefix=False):
        pet_flattened = self.get_high_dim_H_flattened(self.share_intrinsic, only_adapter=only_adapter, only_lora=only_lora, only_prefix=only_prefix)
        output_mc = self.model_AL(**all_input, only_adapter=only_adapter, only_lora=only_lora, only_prefix=only_prefix, flatten_pet=pet_flattened)
        loss = output_mc[0]
        return loss