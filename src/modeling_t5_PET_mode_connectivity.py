import os
import torch
import torch.nn.functional as F
from torch import Tensor, nn
import numpy as np
import random
from modeling_t5_multiHyper_flatten_pet import T5PreTrainedModel, T5ForConditionalGeneration
from transformers import T5Tokenizer

class MyT5_pet_MC(nn.Module):
    def __init__(self, args, config):
        super().__init__()
        self.args = args
        self.config = config
        
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
        self.theta_A = self.faltten(args.load_stage1_adapter_path_list, 'adapter')
        self.theta_A.requires_grad = False
        self.init_L = flatten_init('init_pet/lora_init_seed_42.pth', 'lora')
        self.theta_L = self.faltten(args.load_stage1_lora_path_list, 'lora')
        self.theta_L.requires_grad = False
        self.init_P = flatten_init('init_pet/prefix_init_seed_42.pth', 'prefix')
        self.theta_P = self.faltten(args.load_stage1_prefix_path_list, 'prefix')
        self.theta_P.requires_grad = False
        assert self.theta_A.size()==self.theta_L.size()==self.theta_P.size(), "parameters of adapter and lora and prefix should be same!"
        
        self.flatten_size = self.theta_A.size()[1]
        self.low_dimension = args.low_dimension
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
    
    def init_weight(self):
        for k,v in self.state_dict().items():
            if 'encoder_' in k or 'decoder_' in k:
                eval('self.'+k).data.normal_(mean=0.0, std=0.02)
        
    
    def faltten(self, ckpt_path_list, pet_type):
        def faltten_from_path(ckpt_path_list, pet_type):
            flatten_all = []
            if pet_type=='adapter':
                random_init = self.init_A
            elif pet_type=='lora':
                random_init = self.init_L
            elif pet_type=='prefix':
                random_init = self.init_P
            for ckpt_path in ckpt_path_list:
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
                flatten_all.append(flatten-random_init)
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
            if self.args.decoder_type==1:
                H_A_inverse_delta = self.decoder_adapter(P)
                H_A_inverse = H_A_inverse_delta+self.init_A
            elif self.args.decoder_type==2:
                P_A_low = self.decoder_adapter_low(P)
                P_A_low_nonlinear = torch.tanh(P_A_low)
                H_A_inverse = self.decoder_adapter(P_A_low_nonlinear)
            return H_A_inverse
        if only_lora:
            if self.args.decoder_type==1:
                H_L_inverse_delta = self.decoder_lora(P)
                H_L_inverse = H_L_inverse_delta+self.init_L
            elif self.args.decoder_type==2:
                P_L_low = self.decoder_lora_low(P)
                P_L_low_nonlinear = torch.tanh(P_L_low)
                H_L_inverse = self.decoder_lora(P_L_low_nonlinear)
            return H_L_inverse
        if only_prefix:
            if self.args.decoder_type==1:
                H_P_inverse_delta = self.decoder_prefix(P)
                H_P_inverse = H_P_inverse_delta+self.init_P
            elif self.args.decoder_type==2:
                P_P_low = self.decoder_prefix_low(P)
                P_P_low_nonlinear = torch.tanh(P_P_low)
                H_P_inverse = self.decoder_prefix(P_P_low_nonlinear)
            return H_P_inverse
    
                
    def forward(self, all_input):
        num_theta = self.theta_A.size()[0]
        random_A = int(random.randint(0,10000) % num_theta)
        random_L = int(random.randint(0,10000) % num_theta)
        random_P = int(random.randint(0,10000) % num_theta)
        
        
        P_A = self.get_low_dim_P(self.theta_A[random_A], only_adapter=True)
        P_L = self.get_low_dim_P(self.theta_L[random_L], only_lora=True)
        P_P = self.get_low_dim_P(self.theta_P[random_P], only_prefix=True)
        
        alpha = random.uniform(0,1)
        beta = random.uniform(0,1-alpha)
        
        
        P_alpha = alpha*P_A + beta*P_L + (1-alpha-beta)*P_P

        theta_A_inverse = self.decoder_adapter(P_A)
        loss_A_L2 = torch.dist(self.theta_A[random_A], theta_A_inverse)*self.args.reconstruct_alpha
        adapter_flattened = self.get_high_dim_H_flattened(P_alpha, only_adapter=True)
        output_A_mc = self.model_AL(**all_input, only_adapter=True, flatten_pet=adapter_flattened)
        loss_A_mc = output_A_mc[0]
        
        theta_L_inverse = self.decoder_lora(P_L)
        loss_L_L2 = torch.dist(self.theta_L[random_L], theta_L_inverse)*self.args.reconstruct_alpha
        lora_flattened = self.get_high_dim_H_flattened(P_alpha, only_lora=True)
        output_L_mc = self.model_AL(**all_input, only_lora=True, flatten_pet=lora_flattened)
        loss_L_mc = output_L_mc[0]
        
        theta_P_inverse = self.decoder_prefix(P_P)
        loss_P_L2 = torch.dist(self.theta_P[random_P], theta_P_inverse)*self.args.reconstruct_alpha
        prefix_flattened = self.get_high_dim_H_flattened(P_alpha, only_prefix=True)
        output_P_mc = self.model_AL(**all_input, only_prefix=True, flatten_pet=prefix_flattened)
        loss_P_mc = output_P_mc[0]
        loss = loss_A_L2 + loss_A_mc + loss_L_L2 + loss_L_mc + loss_P_L2 + loss_P_mc

        loss_dist_P = (torch.dist(P_A,P_L)+torch.dist(P_A,P_P)+torch.dist(P_L,P_P))/3
        
        return loss, loss_A_L2, loss_A_mc, loss_L_L2, loss_L_mc, loss_P_L2, loss_P_mc, loss_dist_P
        
    
    def get_valid_loss_and_generation(self, all_input, decoder_input_ids):
        num_theta = self.theta_A.size()[0]
        random_A = int(random.randint(0,10000) % num_theta)
        random_L = int(random.randint(0,10000) % num_theta)
        random_P = int(random.randint(0,10000) % num_theta)
        
        
        P_A = self.get_low_dim_P(self.theta_A[random_A], only_adapter=True)
        P_L = self.get_low_dim_P(self.theta_L[random_L], only_lora=True)
        P_P = self.get_low_dim_P(self.theta_P[random_P], only_prefix=True)
        
        alpha = random.uniform(0,1)
        beta = random.uniform(0,1-alpha)
        
        P_alpha = alpha*P_A + beta*P_L + (1-alpha-beta)*P_P

        theta_A_inverse = self.decoder_adapter(P_A)
        loss_A_L2 = torch.dist(self.theta_A[random_A], theta_A_inverse)*self.args.reconstruct_alpha
        adapter_flattened = self.get_high_dim_H_flattened(P_alpha, only_adapter=True)
        output_A_mc = self.model_AL(**all_input, only_adapter=True, flatten_pet=adapter_flattened)
        loss_A_mc = output_A_mc[0]
        gen_text_adapter = self.generate_text(all_input,decoder_input_ids,only_adapter=True,only_lora=False,only_prefix=False,flatten_pet=adapter_flattened)
        
        theta_L_inverse = self.decoder_lora(P_L)
        loss_L_L2 = torch.dist(self.theta_L[random_L], theta_L_inverse)*self.args.reconstruct_alpha
        lora_flattened = self.get_high_dim_H_flattened(P_alpha, only_lora=True)
        output_L_mc = self.model_AL(**all_input, only_lora=True, flatten_pet=lora_flattened)
        loss_L_mc = output_L_mc[0]
        gen_text_lora = self.generate_text(all_input,decoder_input_ids,only_adapter=False,only_lora=True,only_prefix=False,flatten_pet=lora_flattened)
        
        theta_P_inverse = self.decoder_prefix(P_P)
        loss_P_L2 = torch.dist(self.theta_P[random_P], theta_P_inverse)*self.args.reconstruct_alpha
        prefix_flattened = self.get_high_dim_H_flattened(P_alpha, only_prefix=True)
        output_P_mc = self.model_AL(**all_input, only_prefix=True, flatten_pet=prefix_flattened)
        loss_P_mc = output_P_mc[0]
        gen_text_prefix = self.generate_text(all_input,decoder_input_ids,only_adapter=False,only_lora=False,only_prefix=True,flatten_pet=prefix_flattened)

        loss = loss_A_L2  + loss_L_L2  + loss_P_L2
        loss_dist_P = (torch.dist(P_A,P_L)+torch.dist(P_A,P_P)+torch.dist(P_L,P_P))/3
        
        return loss, loss_A_L2, loss_A_mc, loss_L_L2, loss_L_mc, loss_P_L2, loss_P_mc, loss_dist_P, gen_text_adapter, gen_text_lora, gen_text_prefix
        

    def generate_text(self,all_input,decoder_input_ids,only_adapter,only_lora,only_prefix,flatten_pet=None):
        generated_ids = self.model_AL.generate(
            inputs_embeds=all_input["inputs_embeds"],
            attention_mask=all_input["attention_mask"],
            decoder_input_ids=decoder_input_ids,
            max_length=self.args.max_output_length,
            early_stopping=True,
            only_adapter=only_adapter,
            only_lora=only_lora,
            only_prefix=only_prefix,
            flatten_pet=flatten_pet,
        )
        tokenizer = T5Tokenizer.from_pretrained(self.args.tokenizer_path)
        gen_text = tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        gen_text = list(map(str.strip, gen_text))
        return gen_text