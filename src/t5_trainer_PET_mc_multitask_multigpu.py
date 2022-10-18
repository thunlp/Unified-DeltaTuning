import os
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import logging
import random
import math
import warnings
import argparse
import time
# import loralib as lora

# import lora_onlyB

from transformers import AutoTokenizer, BartTokenizer, BartConfig
from transformers import (
    AdamW,
    Adafactor,
    get_scheduler,
    get_linear_schedule_with_warmup,
    is_torch_available,
)
from dataloader.fulldata_gym_multitask_t5 import NLPFulldataGymMultiTaskData

from transformers import T5ForConditionalGeneration

from intrinsic import intrinsic_dimension, intrinsic_dimension_said
from utils import freeze_embeds, trim_batch, get_tasks_list

from tqdm import tqdm
from collections import OrderedDict
import itertools
from torch.utils.tensorboard import SummaryWriter

# logger = logging.getLogger('trainer')
warnings.filterwarnings("ignore")

def uniform_init(prompt, a=0.0, b=1.0):
    torch.nn.init.uniform_(prompt, a, b)
    # logger.info("init prompt by uniform [{:.3f}, {:.3f}]".format(a, b))

def set_seed(seed: int):
    """
    Helper function for reproducible behavior to set the seed in ``random``, ``numpy``, ``torch`` and/or ``tf`` (if
    installed).

    Args:
        seed (:obj:`int`): The seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    if is_torch_available():
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # ^^ safe to call this function even if cuda is not available

def get_params_for_prompt_optimization(module: torch.nn.Module):
    params = []
    for t in module.named_modules():
        if "prompt" in t[0]:
            params.append({'params': [p for p in list(t[1]._parameters.values()) if p is not None]})

    
    return params

class Trainer:
    def __init__(self, args, logger, model_provider):
        self.args = args
        self.logger = logger
        logger.info("Loading model ...")
                        
        train_tasks = get_tasks_list(args.custom_tasks_splits, "train")
        dev_tasks = get_tasks_list(args.custom_tasks_splits, "dev")
        test_tasks = get_tasks_list(args.custom_tasks_splits, "test")
        unseen_tasks = get_tasks_list(args.custom_tasks_splits, "unseen")
        
        self.args.train_tasks = train_tasks
        self.args.unseen_tasks = unseen_tasks

        self.logger_info("Training on the following tasks: {}".format(train_tasks))
        self.logger_info("Dev on the following tasks: {}".format(dev_tasks))

        self._model, self.config, self.tokenizer = model_provider(args)
        if self.args.tune_method == 'fastfood':
            self.model, self.ID_wrap = intrinsic_dimension(self.model, args.intrinsic_dim, None, set(), args.projection_type, "cuda")
        
        self.train_data = NLPFulldataGymMultiTaskData(logger, args, args.task_dir, tasks=train_tasks, data_split="train", data_type="train", is_training=True, is_test=False)
        self.train_data.load_dataset(self.tokenizer)
        self.train_data.load_dataloader()
        self.dev_data = NLPFulldataGymMultiTaskData(logger, args, args.task_dir, tasks=train_tasks, data_split="dev", data_type="dev", is_training=False, is_test=False)
        self.dev_data.load_dataset(self.tokenizer)
        self.dev_data.load_dataloader()
        self.test_data = NLPFulldataGymMultiTaskData(logger, args, args.task_dir, tasks=test_tasks, data_split="test", data_type="test", is_training=False, is_test=True)
        self.test_data.load_dataset(self.tokenizer)
        self.test_data.load_dataloader()
        
        self.device = self.init_device(args)
        self._model = self._model.to(self.device)
        self.wrape_model = None
        self.gradient_accumulation_steps = args.gradient_accumulation_steps
        self.init_tensorboard(args)
        
        if args.seed is not None:
            set_seed(args.seed)
        if args.tune_method == 'prompt':
            self.prompt = torch.rand((args.prompt_num, self.config.d_model), requires_grad=True, device=self.device)
            self.prepare_data = self.prepare_prompt_data
            uniform_init(prompt=self.prompt, a=-math.sqrt(1 / self.config.d_model), b=math.sqrt(1 / self.config.d_model))
        else:
            self.prepare_data = self.prepare_model_data
        
        if args.tune_method == 'lora_stage2' and not args.load_random_B:
            self.load_lora_B(args.load_lora_B_path)
        elif args.tune_method == 'bias_stage2':
            self.load_bias(args.load_bias_path)
    @property
    def model(self):
        if self.wrape_model is not None:
            return self.wrape_model
        else:
            if self.args.local_rank > -1:
                self.wrape_model = DDP(self._model, device_ids=[self.args.local_rank],
                                       output_device=self.args.local_rank, find_unused_parameters=True)
            else:
                self.wrape_model = self._model
        return self.wrape_model
        
    def init_device(self, args):
        if (not torch.cuda.is_available()):
            print('no gpu can be used!')
            assert torch.cuda.is_available()
        else:
            if args.local_rank == -1:
                return torch.device('cuda:0')
            return torch.device('cuda:%d' % (args.local_rank))
    
    def init_tensorboard(self, args):
        self.tensorboard = None
        # if args.tensorboard_dir is not None:
        
        args.tensorboard_dir = args.output_dir + '/tensorboard'
        self.tensorboard = SummaryWriter(log_dir=args.tensorboard_dir)
    def get_optimzied_group(self):
        if self.args.tune_method == 'model':
            no_decay = ["bias", "layer_norm.weight"]
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0,
                },
            ]
            to_update = self.model.parameters()
        elif self.args.tune_method == 'fastfood':
            for n, p in self.model.named_parameters():
                if p.requires_grad == True:
                    print(n)
            optimizer_grouped_parameters = [{'params': [p for n, p in self.model.named_parameters() if p.requires_grad == True], 'weight_decay': 0.0}]
            to_update = self.model.parameters()
        elif self.args.tune_method == 'prompt':
            for n, p in self.model.named_parameters():
                p.requires_grad = False
            optimizer_grouped_parameters = [
                {
                    "params": [self.prompt],
                    "weight_decay": self.args.weight_decay,
                }
            ]
            to_update = [self.prompt]
        elif self.args.tune_method == 'lora':
            no_decay = ["bias", "layer_norm.weight"]
            for n, p in self.model.named_parameters():
                # if 'lora'
                p.requires_grad = False
                if "lora" in n:
                    p.requires_grad = True
                    print(n)
        
            optimizer_grouped_parameters = []
            for n,p in self.model.named_parameters():
                if "lora" in n:
                    # optimizer_grouped_parameters.append({'params': [p for p in list(t[1]._parameters.values()) if p is not None]})
                    optimizer_grouped_parameters.append({'params': [p]})
            
            to_update = self.model.parameters()
            sum = 0
            for group in optimizer_grouped_parameters:
                for param in group['params']:
                    sum+=param.numel()
            print(sum)
        elif self.args.tune_method == 'adapter':
            no_decay = ["bias", "layer_norm.weight"]
            for n, p in self.model.named_parameters():
                # if 'lora'
                p.requires_grad = False
                if "adapter" in n:
                    p.requires_grad = True
                    print(n)
            optimizer_grouped_parameters = []
            for n,p in self.model.named_parameters():
                if "adapter" in n:
                    # optimizer_grouped_parameters.append({'params': [p for p in list(t[1]._parameters.values()) if p is not None]})
                    optimizer_grouped_parameters.append({'params': [p]})
            to_update = self.model.parameters()
            sum = 0
            for group in optimizer_grouped_parameters:
                for param in group['params']:
                    sum+=param.numel()
            print(sum)
        elif self.args.tune_method == 'prefix':
            no_decay = ["bias", "layer_norm.weight"]
            for n, p in self.model.named_parameters():
                # if 'lora'
                p.requires_grad = False
                if "prefix" in n:
                    p.requires_grad = True
                    print(n)
            optimizer_grouped_parameters = []
            for n,p in self.model.named_parameters():
                if "prefix" in n:
                    optimizer_grouped_parameters.append({'params': [p]})
            to_update = self.model.parameters()
        elif self.args.tune_method == 'hyper_PET':
            no_decay = ["bias", "layer_norm.weight"]
            for n, p in self.model.named_parameters():
                # if 'lora'
                p.requires_grad = False
                if "intrinsic" in n or 'hyper' in n:
                    p.requires_grad = True
                    print(n)
            optimizer_grouped_parameters = []
            for n,p in self.model.named_parameters():
                if "intrinsic" in n or 'hyper' in n:
                    optimizer_grouped_parameters.append({'params': [p]})
            to_update = self.model.parameters()
        elif self.args.tune_method == 'PET_mc':
            no_decay = ["bias", "layer_norm.weight"]
            for n, p in self.model.named_parameters():
                p.requires_grad = False
                if 'adapter' in n or 'lora' in n or 'prefix' in n or 'encoder_' in n or 'decoder_' in n:
                    p.requires_grad = True
                    print(n)
            optimizer_grouped_parameters = []
            for n,p in self.model.named_parameters():
                if 'adapter' in n or 'lora' in n or 'prefix' in n or 'encoder_' in n or 'decoder_' in n:
                    optimizer_grouped_parameters.append({'params': [p]})
            to_update = self.model.parameters()
        
        elif self.args.tune_method == 'bias_stage2' or  self.args.tune_method =='lora_stage2':
            no_decay = ["bias", "layer_norm.weight"]
            for n, p in self.model.named_parameters():
                p.requires_grad = False
                if "lora_R" in n:
                    p.requires_grad = True
                    print(n)
            optimizer_grouped_parameters = []
            for n,p in self.model.named_parameters():
                if "lora_R" in n:
                    optimizer_grouped_parameters.append({'params': [p]})
            to_update = self.model.parameters()
        elif self.args.tune_method == 'bias':
            no_decay = ["bias", "layer_norm.weight"]
            for n, p in self.model.named_parameters():
                p.requires_grad = False
                if "lora" in n:
                    p.requires_grad = True
                    print(n)
            optimizer_grouped_parameters = []
            for n,p in self.model.named_parameters():
                if "lora" in n:
                    optimizer_grouped_parameters.append({'params': [p]})
            to_update = self.model.parameters()
        return optimizer_grouped_parameters, to_update

    def train(self):
        if not os.path.exists(self.args.output_dir):
            os.mkdir(self.args.output_dir)
        if self.args.tune_method == 'model' or self.args.tune_method == 'fastfood' or self.args.tune_method == 'lora' or self.args.tune_method == 'lora_stage2' or self.args.tune_method == 'bias' or self.args.tune_method == 'bias_stage2' or self.args.tune_method == 'adapter' or self.args.tune_method == 'prefix' or self.args.tune_method == 'hyper_PET' or self.args.tune_method == 'PET_mc':
            self.model.train()
        elif self.args.tune_method == 'prompt':
            self.model.eval()
        train_dataloader = self.train_data.dataloader
        if self.args.train_iters is None:
            self.args.train_iters = (
                                    len(train_dataloader) // self.gradient_accumulation_steps
                                    * float(self.args.train_epochs)
                                )
            
        if self.args.train_epochs is None:
            self.args.train_epochs = (self.args.train_iters * self.gradient_accumulation_steps) \
                                     // len(train_dataloader) + 1
        
        all_steps = self.args.train_iters
        if os.path.exists(f"{self.args.output_dir}/checkpoint-last.pt"):
            save_last_step = torch.load(f"{self.args.output_dir}/checkpoint-last.pt")['current_state']['num_updates']
            if save_last_step>=all_steps:
                self.logger_info("Done ... lr={}, bsz={} ...!!!".format(self.args.learning_rate, self.args.train_batch_size))
                exit()

        
        optimizer_grouped_parameters, to_update_parameters = self.get_optimzied_group()
        self.logger_info("Using optimizer: {}".format(self.args.optimizer))
        self.optimizer = self.build_optimizer(self.args, optimizer_grouped_parameters)
        warm_up_steps = int(self.args.train_iters) * self.args.warmup_rate

        if self.args.optimizer == 'adafactor':
            self.scheduler = get_scheduler('constant', self.optimizer, warm_up_steps, self.args.train_iters)
        elif self.args.optimizer == 'adamw':
            self.scheduler =  get_linear_schedule_with_warmup(self.optimizer,
                                            num_warmup_steps=warm_up_steps,
                                            num_training_steps=self.args.train_iters)
        
        num_updates = 0
        log_dict = OrderedDict()
        best_metric = 0
        best_metric_cross_1 = 0
        best_metric_cross_2 = 0
        best_metric_dict = None
        best_num_updates = 0
        early_stop = 0

        if os.path.exists(self.args.output_dir):
            epoch, last_num_updates, best_metric = self.load_from_last(self.args.output_dir)
            best_num_updates = last_num_updates
        
        self.logger_info(f"Train {len(train_dataloader) // self.gradient_accumulation_steps} steps every epoch")
        for epoch in range(self.args.train_epochs):
            for p in self.model.parameters():
                p.grad = None
            self.reset_logging(log_dict)
            
            for local_step, batch in enumerate(train_dataloader):
                if num_updates<=last_num_updates:
                    if local_step % self.gradient_accumulation_steps == 0:
                        num_updates += 1
                        
                        continue
                loss = self.train_step(batch)
                
                self.add_logging(log_dict, 'loss', loss.item() * self.gradient_accumulation_steps)
                            
                if local_step % self.gradient_accumulation_steps == 0:
                    # update model parameter 
                    # to_update_parameters
                    updated, old_scale = self.optimizer_step(self.model.parameters())
                    if updated:
                        num_updates += 1
                    else:
                        self.logger_info("Inf or NaN detected in grad. Change scale from {:.1f} to {:.1f}"\
                                    .format(old_scale, self.scaler.get_scale()))
                    if num_updates % self.args.log_interval == 0:
                        # to log
                        train_loss_mean = self.log_step(log_dict, tensorboard_suffix='train', epoch=epoch, num_updates=num_updates,
                                      lr=self.scheduler.get_last_lr()[0])
                    self.reset_logging(log_dict)
                    if self.args.valid_interval is not None and \
                            num_updates % self.args.valid_interval == 0:
                        
                        current_metrics, task2score = self.valid(epoch, num_updates)
                        for task_items in task2score.items():
                            self.logger_info(task_items)
                        best_update, average_score = self.early_stop(current_metrics, best_metric, epoch, num_updates)
                        
                        if not best_update or train_loss_mean < 1e-7:
                            early_stop += 1
                            self.logger_info(f"Early stop + 1 = {early_stop}. " \
                                        f"Best averate score = {best_metric} at {best_num_updates}.")
                        else:
                            early_stop = 0
                            best_metric = average_score
                            best_num_updates = num_updates
                        save_path = f"{self.args.output_dir}/checkpoint-last.pt"
                        self.save_checkpoint(save_path, epoch, num_updates, best_metric)
                        if self.args.early_stop > 0 and early_stop >= self.args.early_stop:
                            break
                    if self.args.output_interval is not None and \
                            num_updates % self.args.output_interval == 0:
                        save_path = f"{self.args.output_dir}/checkpoint@{epoch}-{num_updates}.pt"
                        self.save_checkpoint(save_path, epoch, num_updates, best_metric)
                        
                    if num_updates >= self.args.train_iters:
                        break
            
            if self.args.early_stop > 0 and early_stop >= self.args.early_stop:
                self.logger_info(f"Stop training. Best average score = {best_metric:.3f} at {best_num_updates}.")
                break
            if num_updates >= self.args.train_iters:
                break
        if self.args.tune_method != 'model':
            save_path = f"{self.args.output_dir}/checkpoint-last.pt"
            self.save_checkpoint(save_path, epoch, num_updates, best_metric)
            
        return best_metric
    
    def early_stop(self, metrics, best_metric, epoch, num_updates):
        current_metric = 0
        update = True
        for key in metrics:
            current_metric += metrics[key]
        current_metric = current_metric / len(metrics)  # compare average
        if best_metric > current_metric:
            update = False
        else:
            save_path = f"{self.args.output_dir}/checkpoint-best.pt"
            self.save_checkpoint(save_path, epoch, num_updates, current_metric)
            
        return update, current_metric

    def generate_text(self,all_input,decoder_input_ids,only_adapter,only_lora,only_prefix,flatten_pet=None):
        generated_ids = self.model.model_AL.generate(
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
        gen_text = self.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        gen_text = list(map(str.strip, gen_text))
        return gen_text
    
    def valid(self, epoch=0, num_updates=0):
        self.model.eval()
        
        valid_dataloader = self.dev_data.dataloader
        my_index = []
        my_prediction= []
        my_prediction_adapter = []
        my_prediction_lora = []
        my_prediction_prefix = []
        
        valid_log_dict = OrderedDict()
        self.logger_info("Begin validation on {:d} samples ...".format(len(self.dev_data.dataset)))
        metrics = {}
        
        with torch.no_grad():
            for local_step, batch in enumerate(valid_dataloader):

                all_input, task_names = self.prepare_data(batch)                    
                decoder_input_ids = self.get_decoder_input_ids(all_input["inputs_embeds"])
                valid_loss, gen_text_adapter, gen_text_lora, gen_text_prefix = self._model.get_valid_loss_and_generation(all_input, task_names, decoder_input_ids)                
                self.add_logging(valid_log_dict, 'loss', valid_loss.item())
                                                
                my_prediction_adapter.extend(gen_text_adapter)
                my_prediction_lora.extend(gen_text_lora)
                my_prediction_prefix.extend(gen_text_prefix)
                
        if len(my_prediction_adapter) != 0:
            task2score_adapter, metrics_adapter = self.dev_data.evaluate(my_prediction_adapter, verbose=False)
            metric_key = list(metrics_adapter.keys())[0]+"_adapter"
            metrics[metric_key] = list(metrics_adapter.values())[0]
        if len(my_prediction_lora) != 0:
            task2score_lora, metrics_lora = self.dev_data.evaluate(my_prediction_lora, verbose=False)
            metric_key = list(metrics_lora.keys())[0]+"_lora"
            metrics[metric_key] = list(metrics_lora.values())[0]
        if len(my_prediction_prefix) != 0:
            task2score_prefix, metrics_prefix = self.dev_data.evaluate(my_prediction_prefix, verbose=False)
            metric_key = list(metrics_prefix.keys())[0]+"_prefix"
            metrics[metric_key] = list(metrics_prefix.values())[0]
        
        metric_key = list(metrics_prefix.keys())[0]
        metrics[metric_key] = (list(metrics_adapter.values())[0]+list(metrics_lora.values())[0]+list(metrics_prefix.values())[0])/3
        valid_loss = self.log_step(valid_log_dict, suffix="Valid |", tensorboard_suffix='valid',
                      epoch=epoch, num_updates=num_updates, **metrics)
        
        task2score = {}
        for tasks_name in task2score_adapter.keys():
            task2score[tasks_name] = {
                'adapter':round(task2score_adapter[tasks_name],4),
                'lora':round(task2score_lora[tasks_name],4),
                'prefix':round(task2score_prefix[tasks_name],4),
                'z_mean':round((task2score_adapter[tasks_name] + task2score_lora[tasks_name] + task2score_prefix[tasks_name])/3,4)}
        if self.args.tune_method == 'model' or self.args.tune_method == 'lora' or self.args.tune_method == 'lora_stage2' or self.args.tune_method == 'bias' or self.args.tune_method == 'bias_stage2' or self.args.tune_method == 'adapter' or self.args.tune_method == 'prefix' or self.args.tune_method == 'hyper_PET' or self.args.tune_method == 'PET_mc':
            self.model.train()
        # metrics['loss'] = - valid_loss
        return metrics, task2score

    def test(self, epoch=0, num_updates=0, PET_name='adapter'):
        
        load_best_path = f"{self.args.output_dir}/checkpoint-best.pt"
        self.load_checkpoint(load_best_path)
        self.model.eval()
        # self.valid(0,0)
        
        test_dataloader = self.test_data.dataloader
        my_index = []
        my_prediction = []
        my_prediction_adapter = []
        my_prediction_lora = []
        my_prediction_prefix = []
        test_log_dict = OrderedDict()
        self.logger_info("Begin test on {:d} samples ...".format(len(self.test_data.dataset)))
        metrics = {}
        
        with torch.no_grad():
            for local_step, batch in enumerate(test_dataloader):
                
                all_input, task_names = self.prepare_data(batch)                    
                decoder_input_ids = self.get_decoder_input_ids(all_input["inputs_embeds"])
                test_loss, gen_text_adapter, gen_text_lora, gen_text_prefix = self._model.get_valid_loss_and_generation(all_input, task_names, decoder_input_ids)                
                self.add_logging(test_log_dict, 'loss', test_loss.item())
                
                my_prediction_adapter.extend(gen_text_adapter)
                my_prediction_lora.extend(gen_text_lora)
                my_prediction_prefix.extend(gen_text_prefix)
        
        if len(my_prediction_adapter) != 0:
            task2score_adapter, metrics_adapter = self.test_data.evaluate(my_prediction_adapter, verbose=False)
            metric_key = list(metrics_adapter.keys())[0]+"_adapter"
            metrics[metric_key] = list(metrics_adapter.values())[0]
        if len(my_prediction_lora) != 0:
            task2score_lora, metrics_lora = self.test_data.evaluate(my_prediction_lora, verbose=False)
            metric_key = list(metrics_lora.keys())[0]+"_lora"
            metrics[metric_key] = list(metrics_lora.values())[0]
        if len(my_prediction_prefix) != 0:
            task2score_prefix, metrics_prefix = self.test_data.evaluate(my_prediction_prefix, verbose=False)
            metric_key = list(metrics_prefix.keys())[0]+"_prefix"
            metrics[metric_key] = list(metrics_prefix.values())[0]
        
        metric_key = list(metrics_prefix.keys())[0]
        metrics[metric_key] = (list(metrics_adapter.values())[0]+list(metrics_lora.values())[0]+list(metrics_prefix.values())[0])/3
        test_loss = self.log_step(test_log_dict, suffix="Test |", tensorboard_suffix='test',
                      epoch=epoch, num_updates=num_updates, **metrics)
        
        task2score = {}
        for tasks_name in task2score_adapter.keys():
            task2score[tasks_name] = {
                'adapter':task2score_adapter[tasks_name],
                'lora':task2score_lora[tasks_name],
                'prefix':task2score_prefix[tasks_name],
                'z_mean':(task2score_adapter[tasks_name] + task2score_lora[tasks_name] + task2score_prefix[tasks_name])/3}

        if self.args.tune_method == 'model' or self.args.tune_method == 'lora' or self.args.tune_method == 'lora_stage2' or self.args.tune_method == 'bias' or self.args.tune_method == 'bias_stage2' or self.args.tune_method == 'adapter' or self.args.tune_method == 'prefix' or self.args.tune_method == 'hyper_PET' or self.args.tune_method == 'PET_mc':
            self.model.train()
        return metrics, task2score

    def get_decoder_input_ids(self, inputs_embeds):
        decoder_start_token_id = self.config.decoder_start_token_id
        decoder_input_ids = (
                torch.ones((inputs_embeds.shape[0], 1), dtype=torch.long, device=inputs_embeds.device) * decoder_start_token_id
        )
        return decoder_input_ids
    
    def save_checkpoint(self, path, epoch, num_updates, best_performance):
        model_to_save = self.model if not hasattr(self.model, 'module') else self.model.module
        if self.args.local_rank <= 0:
            state_dict = OrderedDict()
            if self.args.tune_method == 'model':
                # don't save model
                state_dict['model'] = model_to_save.state_dict()
            elif self.args.tune_method == 'fastfood':
                model_state_dict = model_to_save.state_dict()
                model_state_dict['projection_params'] = self.ID_wrap.projection_params
                state_dict['fastfood'] = model_state_dict
            elif self.args.tune_method == 'prompt':
                # save prompt
                state_dict['prompt'] = self.prompt
            elif self.args.tune_method == 'lora' or self.args.tune_method == 'bias':
                # save lora
                my_state_dict = model_to_save.state_dict()
                state_dict['lora'] = {k: my_state_dict[k] for k in my_state_dict if 'lora_' in k}
            elif self.args.tune_method == 'adapter':
                # save lora
                my_state_dict = model_to_save.state_dict()
                state_dict['adapter'] = {k: my_state_dict[k] for k in my_state_dict if 'adapter_' in k}
            elif self.args.tune_method == 'prefix':
                # save lora
                my_state_dict = model_to_save.state_dict()
                state_dict['prefix'] = {k: my_state_dict[k] for k in my_state_dict if 'prefix_' in k}
            elif self.args.tune_method == 'hyper_PET':
                # save lora
                my_state_dict = model_to_save.state_dict()
                state_dict['hyper_PET'] = {k: my_state_dict[k] for k in my_state_dict if 'intrinsic' in k or 'hyper' in k}
            elif self.args.tune_method == 'PET_mc':
                # save lora
                my_state_dict = model_to_save.state_dict()
                state_dict['PET_mc'] = {k: my_state_dict[k] for k in my_state_dict if 'encoder_' in k or 'decoder_' in k}
            elif self.args.tune_method == 'lora_stage2' or self.args.tune_method == 'bias_stage2':
                # save lora
                my_state_dict = model_to_save.state_dict()
                state_dict['lora_R'] = {k: my_state_dict[k] for k in my_state_dict if 'lora_R' in k}
            else:
                raise NotImplementedError("Invalid tune method of %s" % self.args.tune_method)
            state_dict['optimizer'] = self.optimizer.state_dict()
            state_dict['scheduler'] = self.scheduler.state_dict()
            state_dict['config'] = self.config
            state_dict['args'] = vars(self.args)
            state_dict['current_state'] = {'epoch': epoch, 'num_updates': num_updates}
            state_dict['best_performance'] = best_performance
            torch.save(state_dict, path)
            self.logger_info(f"epoch: {epoch} num_updates: {num_updates} Save {self.args.tune_method} to {path}.")
        if self.args.local_rank > -1:
            dist.barrier()
    
    def load_checkpoint(self, path):
        state_dict = torch.load(path)
        if state_dict['args']['tune_method'] == 'model':
            # load model
            self.model.load_state_dict(state_dict['model'])
        elif state_dict['args']['tune_method'] == 'fastfood':
            # load model
            input()
            self.model.load_state_dict(state_dict['fastfood'])
        elif state_dict['args']['tune_method'] == 'prompt':
            # load prompt
            self.prompt = state_dict['prompt']
        elif state_dict['args']['tune_method'] == 'lora' or state_dict['args']['tune_method'] == 'bias':
            # load lora
            model_dict = {k: v for (k, v) in self.model.state_dict().items()}
            model_dict.update({k: v.cuda() for (k, v) in state_dict['lora'].items()})
            self.model.load_state_dict(model_dict)
        elif state_dict['args']['tune_method'] == 'adapter':
            # load lora
            model_dict = {k: v for (k, v) in self.model.state_dict().items()}
            model_dict.update({k: v.cuda() for (k, v) in state_dict['adapter'].items()})
            self.model.load_state_dict(model_dict)
        elif state_dict['args']['tune_method'] == 'prefix':
            # load lora
            model_dict = {k: v for (k, v) in self.model.state_dict().items()}
            model_dict.update({k: v.cuda() for (k, v) in state_dict['prefix'].items()})
            self.model.load_state_dict(model_dict)
        elif state_dict['args']['tune_method'] == 'hyper_PET':
            # load lora
            model_dict = {k: v for (k, v) in self.model.state_dict().items()}
            model_dict.update({k: v.cuda() for (k, v) in state_dict['hyper_PET'].items()})
            self.model.load_state_dict(model_dict)
        elif state_dict['args']['tune_method'] == 'PET_mc':
            # load lora
            model_dict = {k: v for (k, v) in self.model.state_dict().items()}
            if not hasattr(self.model, 'module'):
                model_dict.update({k: v.cuda() for (k, v) in state_dict['PET_mc'].items()})
            else:
                model_dict.update({'module.'+k: v.cuda() for (k, v) in state_dict['PET_mc'].items()})
            self.model.load_state_dict(model_dict)
        elif state_dict['args']['tune_method'] == 'lora_stage2' or state_dict['args']['tune_method'] == 'bias_stage2':
            # load lora
            model_dict = {k: v for (k, v) in self.model.state_dict().items()}
            model_dict.update({k: v.cuda() for (k, v) in state_dict['lora_R'].items()})
            self.model.load_state_dict(model_dict)    
        else:
            raise NotImplementedError("Invalid tune method of %s" % self.args.tune_method)
        current_state = state_dict['current_state']
        self.logger_info(f"Load {state_dict['args']['tune_method']} from {path}.")
        return current_state

    def load_from_last(self, bs_lr_dir):
        last_ckpt_path = bs_lr_dir + '/checkpoint-last.pt'
        if not os.path.exists(last_ckpt_path):
            return 0, 0, 0
            # assert False, f'No checkpoint-last.pt in {bs_lr_dir}'
        else:
            last_state_dict = torch.load(last_ckpt_path)
            last_epoch = last_state_dict['current_state']['epoch']
            last_num_updates = last_state_dict['current_state']['num_updates']
            best_average_score = last_state_dict['best_performance']
            self.logger_info(f"Loading from {last_ckpt_path}, previous best_mean_performance={best_average_score}")

            model_dict = {k: v for (k, v) in self.model.state_dict().items()}
            if not hasattr(self.model, 'module'):
                model_dict.update({k: v.cuda() for (k, v) in last_state_dict['PET_mc'].items()})
            else:
                model_dict.update({'module.'+k: v.cuda() for (k, v) in last_state_dict['PET_mc'].items()})
            self.model.load_state_dict(model_dict)
            
        return last_epoch, last_num_updates, best_average_score


    def load_lora_B(self, path):
        state_dict = torch.load(path)
        model_dict = {k: v for (k, v) in self.model.state_dict().items()}
        if self.args.decoder_mlp:
            model_dict.update({k: v.cuda() for (k, v) in state_dict['lora'].items() if 'lora_B' in k or 'lora_C' in k})
        else:
            model_dict.update({k: v.cuda() for (k, v) in state_dict['lora'].items() if 'lora_B' in k})
        self.model.load_state_dict(model_dict)
        
    def load_bias(self, path):
        state_dict = torch.load(path)
        model_dict = {k: v for (k, v) in self.model.state_dict().items()}
        if self.args.decoder_mlp:
            model_dict.update({k: v.cuda() for (k, v) in state_dict['lora'].items() if 'lora_B' in k or 'lora_C' in k})
        else:
            model_dict.update({k: v.cuda() for (k, v) in state_dict['lora'].items() if 'lora_B' in k})
        self.model.load_state_dict(model_dict)
    
    def build_optimizer(self, args, params):
        if args.optimizer == 'adafactor':
            optimizer = Adafactor(params, lr=args.learning_rate, scale_parameter=False, relative_step=False, warmup_init=False)
        elif args.optimizer == 'adamw':
            optimizer = AdamW(params, lr=args.learning_rate)
        return optimizer


    def prepare_model_data(self, batch): # t5的输入input_ids全部转化为input_embeds
        all_input = {
            'input_ids': batch[0].to(self.device),
            'attention_mask': batch[1].to(self.device),
            'labels': batch[2].to(self.device),
            'decoder_attention_mask': batch[3].to(self.device),
        }
        input_ids = all_input.pop('input_ids')
        input_embeds = self._model.model_AL.get_input_embeddings()(input_ids)
        all_input['inputs_embeds'] = input_embeds
        task_names = batch[-1]
        # batch[0], batch[1] = trim_batch(batch[0], self.tokenizer.pad_token_id, batch[1])
        # all_input['labels'], all_input['decoder_attention_mask'] = trim_batch(all_input['labels'], self.tokenizer.pad_token_id, all_input['decoder_attention_mask'])
        return all_input, task_names

    def prepare_prompt_data(self, batch):
        all_input = {
            'input_ids': batch[0].to(self.device),
            'attention_mask': batch[1].to(self.device),
            'labels': batch[2].to(self.device),
            'decoder_attention_mask': batch[3].to(self.device),
        }
        input_ids = all_input.pop('input_ids')
        input_embeds = self.model.model_AL.get_input_embeddings()(input_ids)
        batch_size = input_ids.shape[0]
        prompt = torch.unsqueeze(self.prompt, dim=0).expand((batch_size,) + self.prompt.shape)
        prompt_attention = torch.ones(prompt.shape[:2], dtype=torch.long, device=prompt.device)
        # cat prompt with input ids
        input_embeds = torch.cat((prompt, input_embeds), dim=1)
        # cat prompt attention mask to initial attention mask
        all_input['attention_mask'] = torch.cat((prompt_attention, all_input['attention_mask']), dim=1)
        # print("input_embeds", input_embeds.shape)
        all_input['inputs_embeds'] = input_embeds
        # all_input['labels'], all_input['decoder_attention_mask'] = trim_batch(all_input['labels'], self.tokenizer.pad_token_id, all_input['decoder_attention_mask'])
        return all_input

    def train_step(self, batch):
        all_input, task_names = self.prepare_data(batch)
        loss = self.model(all_input, task_names)
        loss = loss / self.gradient_accumulation_steps
        loss.backward()
        
        return loss
    
    def optimizer_step(self, parameters):
        updated = True
        scale = 0
        
        if self.args.optimizer == 'adamw':
            torch.nn.utils.clip_grad_norm_(parameters, self.args.max_grad_norm)
        
        self.optimizer.step()
        if updated:
            self.scheduler.step()
        self.optimizer.zero_grad()
        for p in self.model.parameters():
            p.grad = None
        return updated, scale
    
    def log_step(self, log_dict, suffix='', tensorboard_suffix=None, **kwargs):
        new_log_dict = OrderedDict()
        for key, value in kwargs.items():
            new_log_dict[key] = value
        for key in log_dict:
            key_tensor = torch.tensor(log_dict[key], device=self.device)
            
            key_value = key_tensor.mean().item()
            new_log_dict[key] = key_value
        message = '' + suffix
        
        for key, value in new_log_dict.items():
            if isinstance(value, float):
                message += ' {:s}: {:.5f}'.format(key, value)
            else:
                message += ' {:s}: {}'.format(key, value)
        self.logger_info(message)
        if self.tensorboard is not None:
            for key, value in new_log_dict.items():
                if key in ['epoch', 'num_updates']:
                    continue
                tag = f'{tensorboard_suffix}/{key}' if tensorboard_suffix is not None else key
                global_step = kwargs.get('num_updates', None)
                self.tensorboard.add_scalar(tag, value, global_step=global_step)
        return new_log_dict.get('loss', None)
    
    def add_logging(self, log_dict, key, value):
        if key not in log_dict:
            log_dict[key] = []
        log_dict[key].append(value)
    
    def reset_logging(self, log_dict):
        for key in log_dict:
            log_dict[key] = []

    def logger_info(self, message):
        if self.args.local_rank <= 0:
            self.logger.info(message)
