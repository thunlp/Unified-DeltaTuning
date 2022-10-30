import os
import torch
import torch.nn.functional as F
from torch import Tensor, layer_norm, nn
import numpy as np
import random
# from modeling_t5_multiHyper import T5PreTrainedModel, T5ForConditionalGeneration
from modeling_t5_multiHyper_flatten_pet import T5PreTrainedModel, T5ForConditionalGeneration
from pytorch_metric_learning import losses
from transformers import T5Tokenizer
from transformers.activations import ACT2FN

class MyT5_pet_MC(nn.Module):
    def __init__(self, args, config):
        super().__init__()
        self.args = args
        self.config = config

        self.train_tasks_to_index = {}
        for i,task in enumerate(self.args.train_tasks):
            self.train_tasks_to_index[task] = i
        self.unseen_tasks_to_index = {}
        for i,task in enumerate(self.args.unseen_tasks):
            self.unseen_tasks_to_index[task] = i

        load_stage1_adapter_path_list = self.load_one_seed_all_tasks(args.load_PET_dir + '/full_data_adapter', PET_name='adapter')
        load_stage1_lora_path_list = self.load_one_seed_all_tasks(args.load_PET_dir + '/full_data_lora', PET_name='lora')
        load_stage1_prefix_path_list = self.load_one_seed_all_tasks(args.load_PET_dir + '/full_data_prefix', PET_name='prefix')
        
        
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
        self.theta_A = self.faltten(load_stage1_adapter_path_list, 'adapter')
        self.theta_A.requires_grad = False
                
        self.init_L = flatten_init('init_pet/lora_init_seed_42.pth', 'lora')
        self.theta_L = self.faltten(load_stage1_lora_path_list, 'lora')
        self.theta_L.requires_grad = False
                
        self.init_P = flatten_init('init_pet/prefix_init_seed_42.pth', 'prefix')
        self.theta_P = self.faltten(load_stage1_prefix_path_list, 'prefix')
        self.theta_P.requires_grad = False
        
        assert self.theta_A.size()[0]==len(self.train_tasks_to_index)
        assert self.theta_A.size()==self.theta_L.size()==self.theta_P.size(), "parameters of adapter and lora and prefix should be same!"
        
        self.flatten_size = self.theta_A.size()[1]
        self.flatten_size_prefix = self.theta_P.size()[1]

        self.low_dimension = args.low_dimension
        self.encoder_adapter = nn.Linear(self.flatten_size, self.low_dimension,)
        self.encoder_adapter_low = nn.Linear(self.low_dimension, self.low_dimension)
        self.decoder_adapter = nn.Linear(self.low_dimension, self.flatten_size, bias=False)
        self.decoder_adapter_low = nn.Linear(self.low_dimension, self.low_dimension, bias=False)
        self.encoder_lora = nn.Linear(self.flatten_size, self.low_dimension,)
        self.encoder_lora_low = nn.Linear(self.low_dimension, self.low_dimension)
        self.decoder_lora = nn.Linear(self.low_dimension, self.flatten_size, bias=False)
        self.decoder_lora_low = nn.Linear(self.low_dimension, self.low_dimension, bias=False)
        self.encoder_prefix = nn.Linear(self.flatten_size_prefix, self.low_dimension,)
        self.encoder_prefix_low = nn.Linear(self.low_dimension, self.low_dimension)
        self.decoder_prefix = nn.Linear(self.low_dimension, self.flatten_size_prefix, bias=False)
        self.decoder_prefix_low = nn.Linear(self.low_dimension, self.low_dimension, bias=False)
        
        self.model_AL = T5ForConditionalGeneration.from_pretrained(args.model, config=self.config) #网络结构同时存在，但是不同时forwad
        self.init_weight()
    
    def load_one_seed_all_tasks(self,pet_path,task_split='train', PET_name=None):
        all_tasks_best_ckpt_path_list = []
        all_path_each_pet = os.listdir(pet_path)
        if task_split=='train':
            tasks = self.args.train_tasks
        elif task_split=='unseen':
            tasks = self.args.unseen_tasks
        for task in tasks:
            all_seed_path_each_task = []
            for path in all_path_each_pet:
                if task in path:
                    if PET_name=='adapter':
                        if 'adapter_size_12-seed_42' in path:
                            all_seed_path_each_task.append(path)
                    elif PET_name=='lora':
                        if 'lora_size_10-seed_42' in path:
                            all_seed_path_each_task.append(path)
                    elif PET_name=='prefix':
                        if 'r_24-num_120-SGD_noise_seed_42' in path:
                            all_seed_path_each_task.append(path)
                    
            if len(all_seed_path_each_task)==0:
                print(f'No ckpt for {task}')
            else:
                flag = 0
                for task_seed in all_seed_path_each_task:
                    all_lr_bs = os.listdir(pet_path+'/'+task_seed)
                    lr_bs_dir = None
                    for file in all_lr_bs:
                        if 'seed_42' in file:
                            lr_bs_dir = file
                        
                    if lr_bs_dir!=None:
                        ckpt_path = os.path.join(pet_path,task_seed,lr_bs_dir,"checkpoint-best.pt")
                        if not os.path.exists(ckpt_path):
                            continue
                        else:
                            all_tasks_best_ckpt_path_list.append(ckpt_path)
                            flag+=1
                            break
                if flag==0:
                    assert False,"No ckpt for any seed!"                

        return all_tasks_best_ckpt_path_list

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
            if self.args.encoder_act_type == 'tanh':
               H_A_nonlinear =  torch.tanh(H_A)
            elif self.args.encoder_act_type == 'gelu':
               H_A_nonlinear =  F.gelu(H_A)
            elif self.args.encoder_act_type == 'leakyrelu':
               H_A_nonlinear =  F.leaky_relu(H_A)
            
            P_A = self.encoder_lora_low(H_A_nonlinear)
            return P_A
        if only_lora:
            H_L = self.encoder_lora(paras)
            if self.args.encoder_act_type == 'tanh':
               H_L_nonlinear =  torch.tanh(H_L)
            elif self.args.encoder_act_type == 'gelu':
               H_L_nonlinear =  F.gelu(H_L)
            elif self.args.encoder_act_type == 'leakyrelu':
               H_L_nonlinear =  F.leaky_relu(H_L)
            
            P_L = self.encoder_lora_low(H_L_nonlinear)
            return P_L
        if only_prefix:
            H_P = self.encoder_prefix(paras)
            if self.args.encoder_act_type == 'tanh':
               H_P_nonlinear =  torch.tanh(H_P)
            elif self.args.encoder_act_type == 'gelu':
               H_P_nonlinear =  F.gelu(H_P)
            elif self.args.encoder_act_type == 'leakyrelu':
               H_P_nonlinear =  F.leaky_relu(H_P)
            
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
    
                
    def forward(self, all_input, task_names):
        
        loss_batch = torch.Tensor([0]).cuda()
        each_input = {}
        for i in range(len(task_names)):
            each_input['attention_mask'] = all_input['attention_mask'][i].unsqueeze(0)
            each_input['labels'] = all_input['labels'][i].unsqueeze(0)
            each_input['decoder_attention_mask'] = all_input['decoder_attention_mask'][i].unsqueeze(0)
            each_input['inputs_embeds'] = all_input['inputs_embeds'][i].unsqueeze(0)
            task_index = self.train_tasks_to_index[task_names[i]]
            random_A, random_L, random_P = task_index, task_index, task_index
        
            P_A = self.get_low_dim_P(self.theta_A[random_A], only_adapter=True)
            P_L = self.get_low_dim_P(self.theta_L[random_L], only_lora=True)
            P_P = self.get_low_dim_P(self.theta_P[random_P], only_prefix=True)
            
            alpha = random.uniform(0,1)
            beta = random.uniform(0,1-alpha)
            
            P_alpha = alpha*P_A + beta*P_L + (1-alpha-beta)*P_P

            theta_A_inverse = self.decoder_adapter(P_A)
            loss_A_L2 = torch.dist(self.theta_A[random_A], theta_A_inverse)*self.args.reconstruct_alpha
            adapter_flattened = self.get_high_dim_H_flattened(P_alpha, only_adapter=True)
            output_A_mc = self.model_AL(**each_input, only_adapter=True, flatten_pet=adapter_flattened)
            loss_A_mc = output_A_mc[0]
            
            theta_L_inverse = self.decoder_lora(P_L)
            loss_L_L2 = torch.dist(self.theta_L[random_L], theta_L_inverse)*self.args.reconstruct_alpha
            lora_flattened = self.get_high_dim_H_flattened(P_alpha, only_lora=True)
            output_L_mc = self.model_AL(**each_input, only_lora=True, flatten_pet=lora_flattened)
            loss_L_mc = output_L_mc[0]

            theta_P_inverse = self.decoder_prefix(P_P)
            loss_P_L2 = torch.dist(self.theta_P[random_P], theta_P_inverse)*self.args.reconstruct_alpha
            prefix_flattened = self.get_high_dim_H_flattened(P_alpha, only_prefix=True)
            output_P_mc = self.model_AL(**each_input, only_prefix=True, flatten_pet=prefix_flattened)
            loss_P_mc = output_P_mc[0]
            
            
            loss = loss_A_L2 + loss_A_mc + loss_L_L2 + loss_L_mc + loss_P_L2 + loss_P_mc
            loss_batch = loss_batch + loss
        loss_batch = loss_batch / len(task_names)

        
        return loss_batch
            
    
    def get_valid_loss_and_generation(self, all_input, task_names, decoder_input_ids):
        loss_batch = torch.Tensor([0]).cuda()
        each_input = {}
        adapter_gen_text = []
        lora_gen_text = []
        prefix_gen_text = []

        task_list = []
        task_left_right = {}
        for i,task in enumerate(task_names):
            if task not in task_list:
                task_list.append(task)
                task_left_right[task]=[i,i+1]
            else:
                task_left_right[task][-1]+=1

        for task in task_list:
            left = task_left_right[task][0]
            right = task_left_right[task][1]
            each_input['attention_mask'] = all_input['attention_mask'][left:right]
            each_input['labels'] = all_input['labels'][left:right]
            each_input['decoder_attention_mask'] = all_input['decoder_attention_mask'][left:right]
            each_input['inputs_embeds'] = all_input['inputs_embeds'][left:right]
            task_index = self.train_tasks_to_index[task]
            random_A, random_L, random_P = task_index, task_index, task_index
        
            P_A = self.get_low_dim_P(self.theta_A[random_A], only_adapter=True)
            P_L = self.get_low_dim_P(self.theta_L[random_L], only_lora=True)
            P_P = self.get_low_dim_P(self.theta_P[random_P], only_prefix=True)

            alpha = random.uniform(0,1)
            beta = random.uniform(0,1-alpha)
            
            P_alpha = alpha*P_A + beta*P_L + (1-alpha-beta)*P_P

            theta_A_inverse = self.decoder_adapter(P_A)
            loss_A_L2 = torch.dist(self.theta_A[random_A], theta_A_inverse)*self.args.reconstruct_alpha
            adapter_flattened = self.get_high_dim_H_flattened(P_alpha, only_adapter=True)
            output_A_mc = self.model_AL(**each_input, only_adapter=True, flatten_pet=adapter_flattened)
            loss_A_mc = output_A_mc[0]
            adapter_each_task = self.generate_text(each_input, decoder_input_ids[left:right], only_adapter=True, only_lora=False, only_prefix=False, flatten_pet=adapter_flattened)
            adapter_gen_text.extend(adapter_each_task)
            
            theta_L_inverse = self.decoder_lora(P_L)
            loss_L_L2 = torch.dist(self.theta_L[random_L], theta_L_inverse)*self.args.reconstruct_alpha
            lora_flattened = self.get_high_dim_H_flattened(P_alpha, only_lora=True)
            output_L_mc = self.model_AL(**each_input, only_lora=True, flatten_pet=lora_flattened)
            loss_L_mc = output_L_mc[0]
            lora_each_task = self.generate_text(each_input, decoder_input_ids[left:right], only_adapter=False, only_lora=True, only_prefix=False, flatten_pet=lora_flattened)
            lora_gen_text.extend(lora_each_task)
            
            theta_P_inverse = self.decoder_prefix(P_P)
            loss_P_L2 = torch.dist(self.theta_P[random_P], theta_P_inverse)*self.args.reconstruct_alpha
            prefix_flattened = self.get_high_dim_H_flattened(P_alpha, only_prefix=True)
            output_P_mc = self.model_AL(**each_input, only_prefix=True, flatten_pet=prefix_flattened)
            loss_P_mc = output_P_mc[0]
            prefix_each_task = self.generate_text(each_input, decoder_input_ids[left:right], only_adapter=False, only_lora=False, only_prefix=True, flatten_pet=prefix_flattened)
            prefix_gen_text.extend(prefix_each_task)
            
            loss = loss_A_L2 + loss_A_mc + loss_L_L2 + loss_L_mc + loss_P_L2 + loss_P_mc
                
            loss_batch = loss_batch + loss
        loss_batch = loss_batch / len(task_list)
        return loss_batch, adapter_gen_text, lora_gen_text, prefix_gen_text

    def get_unseen_loss_and_generation(self, all_input, task_names, decoder_input_ids, PET_name='adapter'):
        loss_batch = torch.Tensor([0]).cuda()
        each_input = {}
        adapter_gen_text = []
        lora_gen_text = []
        prefix_gen_text = []

        task_list = []
        task_left_right = {}
        for i,task in enumerate(task_names):
            if task not in task_list:
                task_list.append(task)
                task_left_right[task]=[i,i+1]
            else:
                task_left_right[task][-1]+=1

        for task in task_list:
            left = task_left_right[task][0]
            right = task_left_right[task][1]
            each_input['attention_mask'] = all_input['attention_mask'][left:right]
            each_input['labels'] = all_input['labels'][left:right]
            each_input['decoder_attention_mask'] = all_input['decoder_attention_mask'][left:right]
            each_input['inputs_embeds'] = all_input['inputs_embeds'][left:right]
            task_index = self.unseen_tasks_to_index[task]
            random_A, random_L, random_P = task_index, task_index, task_index

            if PET_name=='adapter':
                P_A = self.get_low_dim_P(self.theta_A_unseen[random_A], only_adapter=True)
                loss_A, text_A, loss_AL, text_L, loss_AP, text_P = self.generate_pet_source_and_cross(P_A, all_input, decoder_input_ids)
            elif PET_name=='lora':
                P_L = self.get_low_dim_P(self.theta_L_unseen[random_L], only_lora=True)
                loss_LA, text_A, loss_L, text_L, loss_LP, text_P = self.generate_pet_source_and_cross(P_L, all_input, decoder_input_ids)
            elif PET_name=='prefix':
                P_P = self.get_low_dim_P(self.theta_P_unseen[random_P], only_prefix=True)
                loss_PA, text_A, loss_PL, text_L, loss_P, text_P = self.generate_pet_source_and_cross(P_P, all_input, decoder_input_ids)

        return text_A,text_L,text_P
            
    def generate_pet_source_and_cross(self, P, all_input, decoder_input_ids):
        adapter_flattened = self.get_high_dim_H_flattened(P, only_adapter=True)
        adapter_to_lora_flattened = self.get_high_dim_H_flattened(P, only_lora=True)
        adapter_to_prefix_flattened = self.get_high_dim_H_flattened(P, only_prefix=True)
        
        output_A_mc = self.model_AL(**all_input, only_adapter=True, flatten_pet=adapter_flattened)
        loss_A_mc = output_A_mc[0]
        gen_text_adapter = self.generate_text(all_input,decoder_input_ids,only_adapter=True,only_lora=False,only_prefix=False,flatten_pet=adapter_flattened)
        
        output_AL_mc = self.model_AL(**all_input, only_lora=True, flatten_pet=adapter_to_lora_flattened)
        loss_AL_mc = output_AL_mc[0]
        gen_text_adapter_to_lora = self.generate_text(all_input,decoder_input_ids,only_adapter=False,only_lora=True,only_prefix=False,flatten_pet=adapter_to_lora_flattened)

        output_AP_mc = self.model_AL(**all_input, only_prefix=True, flatten_pet=adapter_to_prefix_flattened)
        loss_AP_mc = output_AP_mc[0]
        gen_text_adapter_to_prefix = self.generate_text(all_input,decoder_input_ids,only_adapter=False,only_lora=False,only_prefix=True,flatten_pet=adapter_to_prefix_flattened)
        
        return loss_A_mc, gen_text_adapter, loss_AL_mc, gen_text_adapter_to_lora, loss_AP_mc, gen_text_adapter_to_prefix

    
    def multitask_gen_text(self,all_input, task_names, decoder_input_ids):
        each_input = {}
        adapter_gen_text = []
        lora_gen_text = []
        prefix_gen_text = []

        task_list = []
        task_left_right = {}
        for i,task in enumerate(task_names):
            if task not in task_list:
                task_list.append(task)
                task_left_right[task]=[i,i+1]
            else:
                task_left_right[task][-1]+=1

        for task in task_list:
            left = task_left_right[task][0]
            right = task_left_right[task][1]
            each_input['attention_mask'] = all_input['attention_mask'][left:right]
            each_input['labels'] = all_input['labels'][left:right]
            each_input['decoder_attention_mask'] = all_input['decoder_attention_mask'][left:right]
            each_input['inputs_embeds'] = all_input['inputs_embeds'][left:right]
            task_index = self.train_tasks_to_index[task]
            random_A, random_L, random_P = task_index, task_index, task_index
        
            P_A = self.get_low_dim_P(self.theta_A[random_A], only_adapter=True)
            P_L = self.get_low_dim_P(self.theta_L[random_L], only_lora=True)
            P_P = self.get_low_dim_P(self.theta_P[random_P], only_prefix=True)
            alpha = random.uniform(0,1)
            beta = random.uniform(0,1-alpha)

            P_alpha = alpha*P_A + beta*P_L + (1-alpha-beta)*P_P

            adapter_flattened = self.get_high_dim_H_flattened(P_alpha, only_adapter=True)
            adapter_each_task = self.generate_text(each_input, decoder_input_ids[left:right], only_adapter=True, only_lora=False, only_prefix=False, flatten_pet=adapter_flattened)
            adapter_gen_text.extend(adapter_each_task)
            lora_flattened = self.get_high_dim_H_flattened(P_alpha, only_lora=True)
            lora_each_task = self.generate_text(each_input, decoder_input_ids[left:right], only_adapter=False, only_lora=True, only_prefix=False, flatten_pet=lora_flattened)
            lora_gen_text.extend(lora_each_task)
            prefix_flattened = self.get_high_dim_H_flattened(P_alpha, only_prefix=True)
            prefix_each_task = self.generate_text(each_input, decoder_input_ids[left:right], only_adapter=False, only_lora=False, only_prefix=True, flatten_pet=prefix_flattened)
            prefix_gen_text.extend(prefix_each_task)
        return adapter_gen_text, lora_gen_text, prefix_gen_text

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