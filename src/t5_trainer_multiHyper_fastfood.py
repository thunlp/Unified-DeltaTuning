import os
import numpy as np
import torch
import logging
import random
import math
import warnings
import argparse


from transformers import AutoTokenizer, BartTokenizer, BartConfig
from transformers import (
    AdamW,
    Adafactor,
    get_scheduler,
    get_linear_schedule_with_warmup,
    is_torch_available,
)
from dataloader.fewshot_gym_singletask_t5 import NLPFewshotGymSingleTaskData

from transformers import T5ForConditionalGeneration
from intrinsic_hyper import intrinsic_dimension, intrinsic_dimension_said

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

    # if torch.distributed.get_rank() == 0:
    #     print("print params", params)
    return params

class Trainer:
    def __init__(self, args, logger, model_provider):
        if args.tune_method == 'hyper_PET_stage2':
            args.PET_name = self.get_stage2_method(args)
        args.cross_valid = False
        self.args = args
        self.logger = logger
        logger.info("Loading model ...")
        
        self.model, self.config, self.tokenizer = model_provider(args)
        if args.do_said:
            self.device = self.init_device(args)
            self.model = self.model.to(self.device)
            self.model, self.ID_wrap = intrinsic_dimension(self.model, args.intrinsic_dim, None, set(), args.projection_type, "cuda")
        if self.args.tune_method == 'hyper_PET_stage2' and self.args.zero_shot_test:
            dic = vars(args)
                        
            args_cross_1 = argparse.Namespace(**dic)
            args_cross_2 = argparse.Namespace(**dic)
            args_cross_3 = argparse.Namespace(**dic)

            if args.apply_adapter:
                args_cross_1.apply_adapter = False
                args_cross_1.apply_lora = True
                args_cross_1.lora_alpha = 16
                args_cross_1.lora_r = 10
                args_cross_2.apply_adapter = False
                args_cross_2.apply_prefix = True
                args_cross_2.prefix_num = 24
                args_cross_3.apply_adapter = False
                args_cross_3.do_said = True
                
            if args.apply_lora:
                args_cross_1.apply_lora = False
                args_cross_1.apply_adapter = True
                args_cross_1.adapter_type = "houlsby"
                args_cross_1.adapter_size = 12
                args_cross_2.apply_lora = False
                args_cross_2.apply_prefix = True
                args_cross_2.prefix_num = 24
                args_cross_3.apply_lora = False
                args_cross_3.do_said = True
                
            if args.apply_prefix:
                args_cross_1.apply_prefix = False
                args_cross_1.apply_lora = True
                args_cross_1.lora_alpha = 16
                args_cross_1.lora_r = 10
                args_cross_2.apply_prefix = False
                args_cross_2.apply_adapter = True
                args_cross_2.adapter_type = "houlsby"
                args_cross_2.adapter_size = 12
                args_cross_3.apply_prefix = False
                args_cross_3.do_said = True
            
            if args.do_said:
                args_cross_1.do_said = False
                args_cross_1.apply_adapter = True
                args_cross_1.adapter_type = "houlsby"
                args_cross_1.adapter_size = 12
                args_cross_2.do_said = False
                args_cross_2.apply_lora = True
                args_cross_2.lora_alpha = 16
                args_cross_2.lora_r = 10
                args_cross_3.do_said = False
                args_cross_3.apply_prefix = True
                args_cross_3.prefix_num = 24
                
            
            args_cross_1.PET_name = self.get_stage2_method(args_cross_1)
            args_cross_1.cross_valid = True
            args_cross_2.PET_name = self.get_stage2_method(args_cross_2)
            args_cross_2.cross_valid = True
            args_cross_3.PET_name = self.get_stage2_method(args_cross_3)
            args_cross_3.cross_valid = True
            
            self.args_cross_1 = args_cross_1
            self.model_cross_1, self.config_cross_1, _ = model_provider(args_cross_1)
            self.args_cross_2 = args_cross_2
            self.model_cross_2, self.config_cross_2, _ = model_provider(args_cross_2)
            self.args_cross_3 = args_cross_3
            self.model_cross_3, self.config_cross_3, _ = model_provider(args_cross_3)

            if args_cross_3.do_said:
                self.device = self.init_device(args_cross_3)
                self.model_cross_3 = self.model_cross_3.to(self.device)
                self.model_cross_3, self.ID_wrap_cross_3 = intrinsic_dimension(self.model_cross_3, args_cross_3.intrinsic_dim, None, set(), args_cross_3.projection_type, "cuda")
        
        logger.info("Loading Dataset ...")
        self.train_data = NLPFewshotGymSingleTaskData(logger, args, args.train_file, data_type="train", is_training=True)
        self.train_data.load_dataset(self.tokenizer)
        self.train_data.load_dataloader()
        self.dev_data = NLPFewshotGymSingleTaskData(logger, args, args.dev_file, data_type="dev", is_training=False)
        self.dev_data.load_dataset(self.tokenizer)
        self.dev_data.load_dataloader()
        self.test_data = NLPFewshotGymSingleTaskData(logger, args, args.test_file, data_type="test", is_training=False)
        self.test_data.load_dataset(self.tokenizer)
        self.test_data.load_dataloader()

        self.device = self.init_device(args)
        self.model = self.model.to(self.device)
        if self.args.tune_method == 'hyper_PET_stage2' and self.args.zero_shot_test:
            self.model_cross_1 = self.model_cross_1.to(self.device)
            self.model_cross_2 = self.model_cross_2.to(self.device)
            self.model_cross_3 = self.model_cross_3.to(self.device)
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
        elif args.tune_method == 'hyper_PET_stage2':
            self.load_PET(args.load_PET_path)
        elif args.tune_method == 'fastfood_stage2':
            ckpt = torch.load(args.load_fastfood_path)
            # model, ID_wrap = intrinsic_dimension(self.model, self.args.intrinsic_dim, None, set(), self.args.projection_type, "cuda")
            # self.ID_wrap.projection_params = ckpt['fastfood']['projection_params']
            model_dict = {k: v for (k, v) in self.model.state_dict().items()}
            model_dict.update({k: v.cuda() for (k, v) in ckpt['fastfood'].items() if k not in ['projection_params', 'intrinsic_parameter']})
            self.model.load_state_dict(model_dict)
        elif args.tune_method == 'hyper_PET_stage3':
            self.load_PET_cross(args.path_intrinsic, args.path_B)
        
    def init_device(self, args):
        if (not torch.cuda.is_available()):
            print('no gpu can be used!')
            assert torch.cuda.is_available()
        else:
            return torch.device('cuda:0')
    
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
        elif self.args.tune_method == 'fastfood_stage2':
            for n, p in self.model.named_parameters():
                if p.requires_grad == True and n == 'intrinsic_parameter':
                    print(n)
            optimizer_grouped_parameters = [{'params': [p for n, p in self.model.named_parameters() if p.requires_grad == True and n == 'intrinsic_parameter'], 'weight_decay': 0.0}]
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
        elif self.args.tune_method == 'lora' or (self.args.tune_method == 'hyper_PET_stage3' and self.args.apply_lora):
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
                    optimizer_grouped_parameters.append({'params': [p]})
            to_update = self.model.parameters()
        elif self.args.tune_method == 'adapter' or (self.args.tune_method == 'hyper_PET_stage3' and self.args.apply_adapter):
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
                    optimizer_grouped_parameters.append({'params': [p]})
            to_update = self.model.parameters()
        elif self.args.tune_method == 'prefix' or (self.args.tune_method == 'hyper_PET_stage3' and self.args.apply_prefix):
            no_decay = ["bias", "layer_norm.weight"]
            parameters = []
            for n, p in self.model.named_parameters():
                p.requires_grad = False
                if "prefix" in n:
                    p.requires_grad = True
                    print(n)
                    parameters.append(p)
            optimizer_grouped_parameters = [{'params': parameters}]
            to_update = self.model.parameters()
        elif self.args.tune_method == 'hyper_PET':
            no_decay = ["bias", "layer_norm.weight"]
            sum_p_lora = 0
            sum_p_adapter = 0
            sum_p_prefix = 0
            
            for n, p in self.model.named_parameters():
                # if 'lora'
                if not self.args.do_said:
                    p.requires_grad = False
                if "intrinsic" in n.lower() or 'hyper' in n.lower():
                    p.requires_grad = True
                    # print(n)
                    if 'hyper_lora' in n:
                        sum_p_lora += p.numel()
                    if 'hyper_adapter' in n:
                        sum_p_adapter += p.numel()
                    if 'hyper_prefix_project' in n:
                        sum_p_prefix += p.numel()
            for n, p in self.model.named_parameters():
                if p.requires_grad == True:
                    print(n)
            print("sum_p_lora:", sum_p_lora)
            print("sum_p_adapter:", sum_p_adapter)
            print("sum_p_prefix:", sum_p_prefix)
            
            optimizer_grouped_parameters = [{'params': [p for n, p in self.model.named_parameters() if p.requires_grad == True], 'weight_decay': 0.0}]

            to_update = self.model.parameters()
                        
        elif self.args.tune_method == 'hyper_PET_stage2':
            no_decay = ["bias", "layer_norm.weight"]
            for n, p in self.model.named_parameters():
                # if 'lora'
                p.requires_grad = False
                if "intrinsic" in n:
                    p.requires_grad = True
                    print(n)
            optimizer_grouped_parameters = []
            for n,p in self.model.named_parameters():
                if "intrinsic" in n:
                    optimizer_grouped_parameters.append({'params': [p]})
            to_update = self.model.parameters()
        elif self.args.tune_method == 'bias_stage2' or  self.args.tune_method =='lora_stage2':
            no_decay = ["bias", "layer_norm.weight"]
            for n, p in self.model.named_parameters():
                # if 'lora'
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
                # if 'lora'
                p.requires_grad = False
                if "lora" in n:
                    p.requires_grad = True
                    print(n)
            optimizer_grouped_parameters = []
            for n,p in self.model.named_parameters():
                if "lora" in n:
                    optimizer_grouped_parameters.append({'params': [p]})
            to_update = self.model.parameters()
        
        else:
            raise NotImplementedError("Invalid tune method of %s" % self.args.tune_method)
        return optimizer_grouped_parameters, to_update

    def train(self):
        if not os.path.exists(self.args.output_dir):
            os.mkdir(self.args.output_dir)
        if self.args.tune_method == 'model' or self.args.tune_method == 'fastfood' or self.args.tune_method == 'fastfood_stage2' or self.args.tune_method == 'lora' or self.args.tune_method == 'lora_stage2' or self.args.tune_method == 'bias' or self.args.tune_method == 'bias_stage2' or self.args.tune_method == 'adapter' or self.args.tune_method == 'prefix' or self.args.tune_method == 'hyper_PET' or self.args.tune_method == 'hyper_PET_stage2':
            self.model.train()
        elif self.args.tune_method == 'prompt':
            self.model.eval()
        else:
            raise NotImplementedError("Invalid tune method of %s" % self.args.tune_method)
        train_dataloader = self.train_data.dataloader
        if self.args.train_iters is None:
            self.args.train_iters = (
                                    len(train_dataloader) // self.gradient_accumulation_steps
                                    * float(self.args.train_epochs)
                                )
        if self.args.train_epochs is None:
            self.args.train_epochs = (self.args.train_iters * self.gradient_accumulation_steps) \
                                     // len(train_dataloader) + 1
        
        optimizer_grouped_parameters, to_update_parameters = self.get_optimzied_group()
        self.logger.info("Using optimizer: {}".format(self.args.optimizer))
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
        best_metric_cross_3 = 0
        best_metric_dict = None
        best_num_updates = 0
        early_stop = 0
        
        
        self.logger.info(f"Train {len(train_dataloader) // self.gradient_accumulation_steps} steps every epoch")
        for epoch in range(self.args.train_epochs):
            self.optimizer.zero_grad()
            self.reset_logging(log_dict)

            for local_step, batch in enumerate(train_dataloader):
                if self.args.tune_method == 'model':
                    loss = self.train_step(batch)
                elif self.args.tune_method == 'hyper_PET' or self.args.tune_method == 'hyper_PET_stage2':
                    loss_adapter, loss_lora, loss_prefix, loss_said, loss = self.hyper_train_step(batch)
                    
                elif self.args.tune_method == 'fastfood' or self.args.tune_method == 'fastfood_stage2':
                    loss_said = self.train_step(batch)
                    loss = loss_said
                else:
                    assert False
                
                if self.args.apply_adapter:
                    self.add_logging(log_dict, 'loss_adapter', loss_adapter.item() * self.gradient_accumulation_steps)
                if self.args.apply_lora:
                    self.add_logging(log_dict, 'loss_lora', loss_lora.item() * self.gradient_accumulation_steps)
                if self.args.apply_prefix:
                    self.add_logging(log_dict, 'loss_prefix', loss_prefix.item() * self.gradient_accumulation_steps)
                if self.args.do_said:
                    self.add_logging(log_dict, 'loss_said', loss_said.item() * self.gradient_accumulation_steps)
                self.add_logging(log_dict, 'loss', loss.item() * self.gradient_accumulation_steps)
                if local_step % self.gradient_accumulation_steps == 0:
                    # update model parameter 
                    # to_update_parameters
                    updated, old_scale = self.optimizer_step(self.model.parameters())
                    if updated:
                        num_updates += 1
                    else:
                        self.logger.info("Inf or NaN detected in grad. Change scale from {:.1f} to {:.1f}"\
                                    .format(old_scale, self.scaler.get_scale()))
                    
                    if num_updates % self.args.log_interval == 0:
                        # to log
                        train_loss_mean = self.log_step(log_dict, tensorboard_suffix='train', epoch=epoch, num_updates=num_updates,
                                      lr=self.scheduler.get_last_lr()[0])
                    self.reset_logging(log_dict)
                    if self.args.valid_interval is not None and \
                            num_updates % self.args.valid_interval == 0:
                        current_metrics = self.valid(epoch, num_updates, self.args, self.model)
                        best_update, average_score = self.early_stop(current_metrics, best_metric, epoch, num_updates, self.args, self.model)
                        
                        if self.args.tune_method == 'hyper_PET_stage2' and self.args.zero_shot_test:
                            self.model_cross_1.share_intrinsic.data = self.model.share_intrinsic.detach().cpu().data.cuda()
                            self.model_cross_2.share_intrinsic.data = self.model.share_intrinsic.detach().cpu().data.cuda()
                            self.model_cross_3.share_intrinsic.data = self.model.share_intrinsic.detach().cpu().data.cuda()
                                                        
                            cross_metrics_1 = self.valid(epoch, num_updates, self.args_cross_1, self.model_cross_1)
                            cross_metrics_2 = self.valid(epoch, num_updates, self.args_cross_2, self.model_cross_2)
                            cross_metrics_3 = self.valid(epoch, num_updates, self.args_cross_3, self.model_cross_3)
                            best_update_cross_1, average_score_cross_1 = self.early_stop(cross_metrics_1, best_metric_cross_1, epoch, num_updates, self.args_cross_1, self.model_cross_1)
                            best_update_cross_2, average_score_cross_2 = self.early_stop(cross_metrics_2, best_metric_cross_2, epoch, num_updates, self.args_cross_2, self.model_cross_2)
                            best_update_cross_3, average_score_cross_3 = self.early_stop(cross_metrics_3, best_metric_cross_3, epoch, num_updates, self.args_cross_3, self.model_cross_3)
                            if best_update_cross_1:
                                best_metric_cross_1 = average_score_cross_1
                            if best_update_cross_2:
                                best_metric_cross_2 = average_score_cross_2
                            if best_update_cross_3:
                                best_metric_cross_3 = average_score_cross_3
                                
                        if not best_update or train_loss_mean < 1e-7:
                            early_stop += 1
                            self.logger.info(f"Early stop + 1 = {early_stop}. " \
                                        f"Best averate score = {best_metric} at {best_num_updates}.")
                        else:
                            early_stop = 0
                            best_metric = average_score
                            best_num_updates = num_updates
                        if self.args.early_stop > 0 and early_stop >= self.args.early_stop:
                            break
                    if self.args.output_interval is not None and \
                            num_updates % self.args.output_interval == 0:
                        save_path = f"{self.args.output_dir}/checkpoint@{epoch}-{num_updates}.pt"
                        self.save_checkpoint(save_path, epoch, num_updates)
                        
                    if num_updates >= self.args.train_iters:
                        break
            
            if self.args.early_stop > 0 and early_stop >= self.args.early_stop:
                self.logger.info(f"Stop training. Best averate score = {best_metric:.3f} at {best_num_updates}.")
                break
            if num_updates >= self.args.train_iters:
                break
        if self.args.tune_method != 'model':
            save_path = f"{self.args.output_dir}/checkpoint-last.pt"
            self.save_checkpoint(save_path, epoch, num_updates)
            if self.args.tune_method == 'hyper_PET_stage2' and self.args.zero_shot_test:
                save_postfix_1 = self.args_cross_1.PET_name
                save_path_cross_1 = f"{self.args.output_dir}/checkpoint-last-cross_{save_postfix_1}.pt"
                self.save_checkpoint_cross(save_path_cross_1, epoch, num_updates, self.args_cross_1, self.model_cross_1)
                
                save_postfix_2 = self.args_cross_2.PET_name
                save_path_cross_2 = f"{self.args.output_dir}/checkpoint-last-cross_{save_postfix_2}.pt"
                self.save_checkpoint_cross(save_path_cross_2, epoch, num_updates, self.args_cross_2, self.model_cross_2)
                
                save_postfix_3 = self.args_cross_3.PET_name
                save_path_cross_3 = f"{self.args.output_dir}/checkpoint-last-cross_{save_postfix_3}.pt"
                self.save_checkpoint_cross(save_path_cross_3, epoch, num_updates, self.args_cross_3, self.model_cross_3)
        return best_metric, best_metric_cross_1, best_metric_cross_2, best_metric_cross_3
    
    
    def get_stage2_method(self, args):
        assert args.tune_method == 'hyper_PET_stage2', "Function get_stage2_method must be used in hyper_PET_stage2!"
        method_name = []
        if args.apply_lora:
            method_name.append("lora")
        if args.apply_adapter:
            method_name.append("adapter")
        if args.apply_prefix:
            method_name.append("prefix")
        if args.do_said:
            method_name.append("said")
        return method_name[0] if len(method_name)==1 else "_and_".join(method_name)
    
    def early_stop(self, metrics, best_metric, epoch, num_updates, args, model):
        current_metric = 0
        update = True
        for key in metrics:
            current_metric += metrics[key]
        current_metric = current_metric / len(metrics)  # compare average
        if best_metric > current_metric:
            update = False
        else:
            if not args.cross_valid:
                save_path = f"{args.output_dir}/checkpoint-best.pt"
                self.save_checkpoint(save_path, epoch, num_updates)
            else:
                save_postfix = args.PET_name
                save_path = f"{args.output_dir}/checkpoint-best-cross_{save_postfix}.pt"
                self.save_checkpoint_cross(save_path, epoch, num_updates, args, model)
            
        return update, current_metric

    def valid(self, epoch=0, num_updates=0, args=None, model=None):
        if args.cross_valid:
            self.logger.info("Cross validation on {:d} samples ...".format(len(self.dev_data.dataset)))
        else:
            self.logger.info("Begin validation on {:d} samples ...".format(len(self.dev_data.dataset)))
        model.eval()
        
        valid_dataloader = self.dev_data.dataloader
        my_index = []
        my_prediction = []
        my_prediction_adapter= []
        my_prediction_lora = []
        my_prediction_prefix = []
        my_prediction_said = []
        valid_log_dict = OrderedDict()
        metrics = {}
        
        with torch.no_grad():
            for local_step, batch in enumerate(valid_dataloader):
                
                if not (args.apply_adapter or args.apply_lora or args.apply_prefix or args.do_said):
                    valid_loss, gen_text = self.hyper_valid_step(batch, only_adapter=False, only_lora=False, only_prefix=False, only_said=False, args=args, model=model)
                    loss_name = 'loss_cross' if args.cross_valid else 'loss'
                    self.add_logging(valid_log_dict, loss_name, valid_loss)
                    my_prediction.extend(gen_text)
                if args.do_said:
                    valid_said_loss, gen_said_text = self.hyper_valid_step(batch, only_adapter=False, only_lora=False, only_prefix=False, only_said=True, args=args, model=model)
                    loss_name = 'loss_said_cross' if args.cross_valid else 'loss_said'
                    self.add_logging(valid_log_dict, loss_name, valid_said_loss)
                    my_prediction_said.extend(gen_said_text)
                if args.apply_adapter:
                    valid_adapter_loss, gen_adapter_text = self.hyper_valid_step(batch, only_adapter=True, only_lora=False, only_prefix=False, only_said=False, args=args, model=model)
                    loss_name = 'loss_adapter_cross' if args.cross_valid else 'loss_adapter'
                    self.add_logging(valid_log_dict, loss_name, valid_adapter_loss)
                    my_prediction_adapter.extend(gen_adapter_text)
                if args.apply_lora:
                    valid_lora_loss, gen_lora_text = self.hyper_valid_step(batch, only_adapter=False, only_lora=True, only_prefix=False, only_said=False, args=args, model=model)
                    loss_name = 'loss_lora_cross' if args.cross_valid else 'loss_lora'
                    self.add_logging(valid_log_dict, loss_name, valid_lora_loss)
                    my_prediction_lora.extend(gen_lora_text)
                if args.apply_prefix:
                    valid_prefix_loss, gen_prefix_text = self.hyper_valid_step(batch, only_adapter=False, only_lora=False, only_prefix=True, only_said=False, args=args, model=model)
                    loss_name = 'loss_prefix_cross' if args.cross_valid else 'loss_prefix'
                    self.add_logging(valid_log_dict, loss_name, valid_prefix_loss)
                    my_prediction_prefix.extend(gen_prefix_text)
        
        if len(my_prediction) != 0:
            metrics = self.dev_data.evaluate(my_prediction, verbose=False)
            metric_key = list(metrics.keys())[0]+"_cross" if args.cross_valid else list(metrics.keys())[0]
            metrics[metric_key] = list(metrics.values())[0]        
        
        if len(my_prediction_said) != 0:
            metrics_said = self.dev_data.evaluate(my_prediction_said, verbose=False)
            metric_key = list(metrics_said.keys())[0]+"_said_cross" if args.cross_valid else list(metrics_said.keys())[0]+"_said"
            metrics[metric_key] = list(metrics_said.values())[0]
        
        if len(my_prediction_adapter) != 0:
            metrics_adapter = self.dev_data.evaluate(my_prediction_adapter, verbose=False)
            metric_key = list(metrics_adapter.keys())[0]+"_adapter_cross" if args.cross_valid else list(metrics_adapter.keys())[0]+"_adapter"
            metrics[metric_key] = list(metrics_adapter.values())[0]
        
        if len(my_prediction_lora) != 0:
            metrics_lora = self.dev_data.evaluate(my_prediction_lora, verbose=False)
            metric_key = list(metrics_lora.keys())[0]+"_lora_cross" if args.cross_valid else list(metrics_lora.keys())[0]+"_lora"
            metrics[metric_key] = list(metrics_lora.values())[0]
        
        if len(my_prediction_prefix) != 0:
            metrics_prefix = self.dev_data.evaluate(my_prediction_prefix, verbose=False)
            metric_key = list(metrics_prefix.keys())[0]+"_prefix_cross" if args.cross_valid else list(metrics_prefix.keys())[0]+"_prefix"
            metrics[metric_key] = list(metrics_prefix.values())[0]
                    
        valid_loss = self.log_step(valid_log_dict, suffix="Valid |", tensorboard_suffix='valid',
                      epoch=epoch, num_updates=num_updates, **metrics)
        
        if self.args.tune_method == 'model' or self.args.tune_method == 'lora' or self.args.tune_method == 'lora_stage2' or self.args.tune_method == 'bias' or self.args.tune_method == 'bias_stage2' or self.args.tune_method == 'adapter' or self.args.tune_method == 'adapter' or self.args.tune_method == 'hyper_PET' or self.args.tune_method == 'hyper_PET_stage3':
            model.train()
        if self.args.tune_method == 'hyper_PET_stage2' and not args.cross_valid:
            model.train()
        return metrics
    
    def test(self, epoch=0, num_updates=0, model=None, args=None, test_main=False):
        
        if args.cross_valid:
            load_postfix = args.PET_name
            load_best_path = f"{args.output_dir}/checkpoint-best-cross_{load_postfix}.pt"
            if os.path.exists(load_best_path):
                # ?
                self.load_checkpoint_cross(load_best_path, model)
            if test_main:
                
                load_best_path_main = f"{args.output_dir}/checkpoint-best.pt"
                if os.path.exists(load_best_path_main):
                    self.logger.info("Using main intrinsic to initialize in cross testing")
                    model.share_intrinsic.data = torch.load(load_best_path_main)['hyper_PET_stage2']['share_intrinsic'].data.cuda()
        else:
            load_best_path = f"{args.output_dir}/checkpoint-best.pt"
            if os.path.exists(load_best_path):
                if not (self.args.tune_method == 'fastfood' or self.args.tune_method == 'fastfood_stage2'):
                    self.load_checkpoint(load_best_path)
                    model = self.model
                else:
                    ckpt = torch.load(load_best_path)
                    
                    model_dict = {k: v for (k, v) in self.model.state_dict().items()}
                    model_dict.update({k: v.cuda() for (k, v) in ckpt['fastfood'].items() if k not in ['projection_params']})
                    model.load_state_dict(model_dict)
                    for par in model.parameters():
                        par.requires_grad = False

        model.eval()
        
        test_dataloader = self.test_data.dataloader
        my_index = []
        my_prediction = []
        my_prediction_adapter = []
        my_prediction_lora = []
        my_prediction_prefix = []
        my_prediction_said = []
        test_log_dict = OrderedDict()
        self.logger.info("Begin test on {:d} samples ...".format(len(self.test_data.dataset)))
        metrics = {}
        
        with torch.no_grad():
            for local_step, batch in enumerate(test_dataloader):
                
                if not (args.apply_adapter or args.apply_lora or args.apply_prefix or args.do_said):
                    test_loss, gen_text = self.hyper_valid_step(batch, only_adapter=False, only_lora=False, only_prefix=False, only_said=False, args=args, model=model)
                    loss_name = 'loss_cross' if args.cross_valid else 'loss'
                    self.add_logging(test_log_dict, loss_name, test_loss)
                    my_prediction.extend(gen_text)
                if args.do_said:
                    test_said_loss, gen_said_text = self.hyper_valid_step(batch, only_adapter=False, only_lora=False, only_prefix=False, only_said=True, args=args, model=model)
                    loss_name = 'loss_said_cross' if args.cross_valid else 'loss_said'
                    self.add_logging(test_log_dict, loss_name, test_said_loss)
                    my_prediction_said.extend(gen_said_text)
                if args.apply_adapter:
                    test_adapter_loss, gen_adapter_text = self.hyper_valid_step(batch, only_adapter=True, only_lora=False, only_prefix=False, only_said=False, args=args, model=model)
                    loss_name = 'loss_adapter_cross' if args.cross_valid else 'loss_adapter'
                    self.add_logging(test_log_dict, loss_name, test_adapter_loss)
                    my_prediction_adapter.extend(gen_adapter_text)
                if args.apply_lora:
                    test_lora_loss, gen_lora_text = self.hyper_valid_step(batch, only_adapter=False, only_lora=True, only_prefix=False, only_said=False, args=args, model=model)
                    loss_name = 'loss_lora_cross' if args.cross_valid else 'loss_lora'
                    self.add_logging(test_log_dict, loss_name, test_lora_loss)
                    my_prediction_lora.extend(gen_lora_text)
                if args.apply_prefix:
                    test_prefix_loss, gen_prefix_text = self.hyper_valid_step(batch, only_adapter=False, only_lora=False, only_prefix=True, only_said=False, args=args, model=model)
                    loss_name = 'loss_prefix_cross' if args.cross_valid else 'loss_prefix'
                    self.add_logging(test_log_dict, loss_name, test_prefix_loss)
                    my_prediction_prefix.extend(gen_prefix_text)
        
        if len(my_prediction) != 0:
            metrics = self.test_data.evaluate(my_prediction, verbose=False)
            metric_key = list(metrics.keys())[0]+"_cross" if args.cross_valid else list(metrics.keys())[0]
            metrics[metric_key] = list(metrics.values())[0]
        
        if len(my_prediction_said) != 0:
            metrics_said = self.test_data.evaluate(my_prediction_said, verbose=False)
            metric_key = list(metrics_said.keys())[0]+"_said_cross" if args.cross_valid else list(metrics_said.keys())[0]+"_said"
            metrics[metric_key] = list(metrics_said.values())[0]
                
        if len(my_prediction_adapter) != 0:
            metrics_adapter = self.test_data.evaluate(my_prediction_adapter, verbose=False)
            metric_key = list(metrics_adapter.keys())[0]+"_adapter_cross" if args.cross_valid else list(metrics_adapter.keys())[0]+"_adapter"
            metrics[metric_key] = list(metrics_adapter.values())[0]
                    
        if len(my_prediction_lora) != 0:
            metrics_lora = self.test_data.evaluate(my_prediction_lora, verbose=False)
            metric_key = list(metrics_lora.keys())[0]+"_lora_cross" if args.cross_valid else list(metrics_lora.keys())[0]+"_lora"
            metrics[metric_key] = list(metrics_lora.values())[0]
                    
        if len(my_prediction_prefix) != 0:
            metrics_prefix = self.test_data.evaluate(my_prediction_prefix, verbose=False)
            metric_key = list(metrics_prefix.keys())[0]+"_prefix_cross" if args.cross_valid else list(metrics_prefix.keys())[0]+"_prefix"
            metrics[metric_key] = list(metrics_prefix.values())[0]
            
        ave_metric = 0
        if args.tune_method == 'model':
            pass
        elif args.tune_method == 'hyper_PET' or args.tune_method == 'hyper_PET_stage2':
            method_cnt = args.apply_adapter + args.apply_lora + args.apply_prefix + args.do_said
            if args.do_said:
                ave_metric += list(metrics_said.values())[0]
                metric_method = list(metrics_said.keys())[0]
            if args.apply_adapter:
                ave_metric += list(metrics_adapter.values())[0]
                metric_method = list(metrics_adapter.keys())[0]
            if args.apply_lora:
                ave_metric += list(metrics_lora.values())[0]
                metric_method = list(metrics_lora.keys())[0]
            if args.apply_prefix:
                ave_metric += list(metrics_prefix.values())[0]
                metric_method = list(metrics_prefix.keys())[0]
            metrics[metric_method] = ave_metric / method_cnt
        
        test_loss = self.log_step(test_log_dict, suffix="Test |", tensorboard_suffix='test',
                      epoch=epoch, num_updates=num_updates, **metrics)
        
        return metrics

    def get_decoder_input_ids(self, inputs_embeds):
        decoder_start_token_id = self.config.decoder_start_token_id
        decoder_input_ids = (
                torch.ones((inputs_embeds.shape[0], 1), dtype=torch.long, device=inputs_embeds.device) * decoder_start_token_id
        )
        return decoder_input_ids
    
    def save_checkpoint(self, path, epoch, num_updates):
        state_dict = OrderedDict()
        if self.args.tune_method == 'model':
            # don't save model
            state_dict['model'] = self.model.state_dict()
        elif self.args.tune_method == 'fastfood' or self.args.tune_method == 'fastfood_stage2':
            my_state_dict = self.model.state_dict()
            # my_state_dict['projection_params'] = self.ID_wrap.projection_params
            state_dict['fastfood'] = my_state_dict
        elif self.args.tune_method == 'prompt':
            # save prompt
            state_dict['prompt'] = self.prompt
        elif self.args.tune_method == 'lora' or self.args.tune_method == 'bias' or (self.args.tune_method == 'hyper_PET_stage3' and self.args.apply_lora):
            # save lora
            my_state_dict = self.model.state_dict()
            state_dict['lora'] = {k: my_state_dict[k] for k in my_state_dict if 'lora_' in k}
        elif self.args.tune_method == 'adapter' or (self.args.tune_method == 'hyper_PET_stage3' and self.args.apply_adapter):
            # save lora
            my_state_dict = self.model.state_dict()
            state_dict['adapter'] = {k: my_state_dict[k] for k in my_state_dict if 'adapter_' in k}
        elif self.args.tune_method == 'prefix' or (self.args.tune_method == 'hyper_PET_stage3' and self.args.apply_prefix):
            # save prefix
            my_state_dict = self.model.state_dict()
            state_dict['prefix'] = {k: my_state_dict[k] for k in my_state_dict if 'prefix_' in k}
        elif self.args.tune_method == 'hyper_PET':
            # save lora
            my_state_dict = self.model.state_dict()
            if not self.args.do_said:
                state_dict['hyper_PET'] = {k: my_state_dict[k] for k in my_state_dict if 'intrinsic' in k or 'hyper' in k}
            else:
                state_dict_mix = {k: my_state_dict[k] for k in my_state_dict if 'intrinsic' in k or 'hyper' in k or 'trained_said' in k}
                state_dict['hyper_PET'] = state_dict_mix
                state_dict['projection_params'] = self.ID_wrap.projection_params
        elif self.args.tune_method == 'hyper_PET_stage2':
            # save lora
            my_state_dict = self.model.state_dict()
            state_dict['hyper_PET_stage2'] = {k: my_state_dict[k] for k in my_state_dict if 'intrinsic' in k or 'hyper' in k or 'trained_said' in k}
        elif self.args.tune_method == 'lora_stage2' or self.args.tune_method == 'bias_stage2':
            # save lora
            my_state_dict = self.model.state_dict()
            state_dict['lora_R'] = {k: my_state_dict[k] for k in my_state_dict if 'lora_R' in k}
        else:
            raise NotImplementedError("Invalid tune method of %s" % self.args.tune_method)
        state_dict['optimizer'] = self.optimizer.state_dict()
        state_dict['scheduler'] = self.scheduler.state_dict()
        state_dict['config'] = self.config
        state_dict['args'] = vars(self.args)
        state_dict['current_state'] = {'epoch': epoch, 'num_updates': num_updates}
        torch.save(state_dict, path)
        self.logger.info(f"epoch: {epoch} num_updates: {num_updates} Save {self.args.tune_method} to {path}.")
    
    def save_checkpoint_cross(self, path, epoch, num_updates, args_cross, model_cross):
        state_dict = OrderedDict()
        if self.args.tune_method == 'hyper_PET_stage2':
            # save cross share intrinsic
            my_state_dict = model_cross.state_dict()
            state_dict['hyper_PET_stage2'] = {k: my_state_dict[k] for k in my_state_dict if 'intrinsic' in k or 'hyper' in k}
        else:
            raise NotImplementedError("Don't save_checkpoint_cross in the tuning method of %s" % self.args.tune_method) 
        
        state_dict['args'] = vars(args_cross)
        state_dict['current_state'] = {'epoch': epoch, 'num_updates': num_updates}
        torch.save(state_dict, path)
        save_B_name = args_cross.PET_name
        save_R_name = self.args.PET_name
        self.logger.info(f"epoch: {epoch} num_updates: {num_updates} Save cross B:{save_B_name} and R:{save_R_name} to {path}.")
        
    def load_checkpoint(self, path):
        state_dict = torch.load(path)
        if state_dict['args']['tune_method'] == 'model':
            # load model
            self.model.load_state_dict(state_dict['model'])
        elif state_dict['args']['tune_method'] == 'fastfood' or state_dict['args']['tune_method'] == 'fastfood_stage2':
            # load model
            # assert False
            self.model.load_state_dict(state_dict['fastfood'])
            model, ID_wrap = intrinsic_dimension(model, self.args.intrinsic_dim, None, set(), self.args.projection_type, "cuda")
            self.ID_wrap.projection_params = state_dict['fastfood']['projection_params']
            model_dict = {k: v for (k, v) in model.state_dict().items()}
            model_dict.update({k: v.cuda() for (k, v) in ckpt.items() if k not in ['projection_params', 'intrinsic_parameter']})
            model.load_state_dict(model_dict)
        elif state_dict['args']['tune_method'] == 'prompt':
            # load prompt
            self.prompt = state_dict['prompt']
        elif state_dict['args']['tune_method'] == 'lora' or state_dict['args']['tune_method'] == 'bias' or (self.args.tune_method == 'hyper_PET_stage3' and self.args.apply_lora):
            # load lora
            model_dict = {k: v for (k, v) in self.model.state_dict().items()}
            model_dict.update({k: v.cuda() for (k, v) in state_dict['lora'].items()})
            self.model.load_state_dict(model_dict)
        elif state_dict['args']['tune_method'] == 'adapter' or (self.args.tune_method == 'hyper_PET_stage3' and self.args.apply_adapter):
            # load adapter
            model_dict = {k: v for (k, v) in self.model.state_dict().items()}
            model_dict.update({k: v.cuda() for (k, v) in state_dict['adapter'].items()})
            self.model.load_state_dict(model_dict)
        elif state_dict['args']['tune_method'] == 'prefix' or (self.args.tune_method == 'hyper_PET_stage3' and self.args.apply_prefix):
            # load prefix
            model_dict = {k: v for (k, v) in self.model.state_dict().items()}
            model_dict.update({k: v.cuda() for (k, v) in state_dict['prefix'].items()})
            self.model.load_state_dict(model_dict)
        elif state_dict['args']['tune_method'] == 'hyper_PET':
            # load lora
            model_dict = {k: v for (k, v) in self.model.state_dict().items()}
            model_dict.update({k: v.cuda() for (k, v) in state_dict['hyper_PET'].items()})
            self.model.load_state_dict(model_dict)
            if self.args.do_said:
                # pass
                self.ID_wrap.projection_params = state_dict['projection_params']            
                
        elif state_dict['args']['tune_method'] == 'hyper_PET_stage2':
            # load lora
            model_dict = {k: v for (k, v) in self.model.state_dict().items()}
            model_dict.update({k: v.cuda() for (k, v) in state_dict['hyper_PET_stage2'].items()})
            self.model.load_state_dict(model_dict)
        elif state_dict['args']['tune_method'] == 'lora_stage2' or state_dict['args']['tune_method'] == 'bias_stage2':
            # load lora
            model_dict = {k: v for (k, v) in self.model.state_dict().items()}
            model_dict.update({k: v.cuda() for (k, v) in state_dict['lora_R'].items()})
            self.model.load_state_dict(model_dict)    
        else:
            raise NotImplementedError("Invalid tune method of %s" % self.args.tune_method)
        current_state = state_dict['current_state']
        self.logger.info(f"Load {state_dict['args']['tune_method']} from {path}.")
        return current_state

    def load_checkpoint_cross(self, path, model):
        state_dict = torch.load(path)
        if state_dict['args']['tune_method'] == 'hyper_PET_stage2':
            # load lora
            model_dict = {k: v for (k, v) in model.state_dict().items()}
            model_dict.update({k: v.cuda() for (k, v) in state_dict['hyper_PET_stage2'].items()})
            model.load_state_dict(model_dict)
            self.logger.info(f"Load {state_dict['args']['tune_method']} from {path}.")
    
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
    
    def load_PET(self, path):
        state_dict = torch.load(path)        
        PET_dict_no_module = {}
        
        for (k, v) in state_dict['hyper_PET'].items():
            if 'module' not in k:
                PET_dict_no_module[k] = v
            else:
                PET_dict_no_module[k[7:]] = v
        def load_cross_model_state(args, model):
            model_dict = {k: v for (k, v) in model.state_dict().items()}
            if args.apply_adapter:
                model_dict.update({k: v.cuda() for (k, v) in PET_dict_no_module.items() if 'hyper_adapter' in k})
            if args.apply_lora:
                model_dict.update({k: v.cuda() for (k, v) in PET_dict_no_module.items() if 'hyper_lora' in k})
            if args.apply_prefix:
                model_dict.update({k: v.cuda() for (k, v) in PET_dict_no_module.items() if 'hyper_prefix_project' in k})
            if args.do_said:
                model_dict.update({k: v.cuda() for (k, v) in PET_dict_no_module.items() if 'trained_said' in k})
                
            method_name = self.get_stage2_method(args)
            model.load_state_dict(model_dict)
                
        self.logger.info(f"Loading {self.args.PET_name} from stage1")
        load_cross_model_state(self.args, self.model)
        if self.args.do_said:
            self.ID_wrap.projection_params = state_dict['projection_params']
        
        if self.args.load_R_stage1:
            self.logger.info("Loading share_intrinsic from stage1 task")
            self.model.share_intrinsic.data = state_dict['hyper_PET']['share_intrinsic'].cuda()
        elif self.args.load_R_mean_std_stage1:
            r_mean = state_dict['hyper_PET']['share_intrinsic'].mean()
            r_std = state_dict['hyper_PET']['share_intrinsic'].std()
            self.logger.info(f"Initializing share_intrinsic using mean: {r_mean} std: {r_std} of stage1_R")
            self.model.share_intrinsic.data.normal_(mean=r_mean, std=r_std)
            self.logger.info(self.model.share_intrinsic.data)
        elif self.args.fix_R:
            if self.args.intrinsic_dim == 1:
                self.model.share_intrinsic.data = torch.Tensor([[0.01]]).cuda()
            elif self.args.intrinsic_dim == 16:
                self.model.share_intrinsic.data = torch.Tensor([[-0.0229],[-0.0133],[-0.0080],[-0.0268],[ 0.0010],[-0.0101],[ 0.0221],[ 0.0085],[-0.0080],[ 0.0314],[-0.0024],[ 0.0012],[ 0.0041],[ 0.0229],[ 0.0047],[-0.0467]]).cuda()
            self.logger.info(f"Initialized share_intrinsic of model using fixed R")
            self.logger.info(self.model.share_intrinsic.data)    
        else:
            self.logger.info(f"Initialized share_intrinsic randomly using mean: {self.args.r_mean} std: {self.args.r_std} ")
            self.logger.info(self.model.share_intrinsic.data)
        
        if self.args.tune_method == 'hyper_PET_stage2' and self.args.zero_shot_test:
            self.logger.info(f"Loading {self.args_cross_1.PET_name} from stage1")
            load_cross_model_state(self.args_cross_1, self.model_cross_1)
            self.logger.info(f"Loading {self.args_cross_2.PET_name} from stage1")
            load_cross_model_state(self.args_cross_2, self.model_cross_2)
            self.logger.info(f"Loading {self.args_cross_3.PET_name} from stage1")
            load_cross_model_state(self.args_cross_3, self.model_cross_3)
            if self.args_cross_3.do_said:
                self.ID_wrap_cross_3.projection_params = state_dict['projection_params']
            
            
    def load_PET_cross(self, path_cross):
        intrinsic_from_another = torch.load(path_cross)['hyper_PET_stage2']['share_intrinsic']
        state_dict = torch.load(path_cross)['hyper_PET_stage2']
        model_dict = {k: v for (k, v) in self.model.state_dict().items()}
        model_dict_to_update = {}        
        if self.args.tune_method == 'hyper_PET_stage3' and self.args.apply_adapter:
            for (k, v) in model_dict.items():
                k_split = k.split('.')
                k_in_hyper = ''
                if 'adapter_A' in k:
                    k_in_hyper = '.'.join(['hyper_adapter_A' if i=='adapter_A' else i for i in k_split])
                if 'adapter_B' in k:
                    k_in_hyper = '.'.join(['hyper_adapter_B' if i=='adapter_B' else i for i in k_split])
                if k_in_hyper in state_dict.keys():
                    model_dict_to_update[k] = (intrinsic_from_another.T @ state_dict[k_in_hyper]).view(v.size()).cuda()    
        if self.args.tune_method == 'hyper_PET_stage3' and self.args.apply_lora:
            for (k, v) in model_dict.items():
                k_split = k.split('.')
                k_in_hyper = ''
                if 'lora_A' in k:
                    k_in_hyper = '.'.join(['hyper_lora_A' if i=='lora_A' else i for i in k_split])
                if 'lora_B' in k:
                    k_in_hyper = '.'.join(['hyper_lora_B' if i=='lora_B' else i for i in k_split])
                if k_in_hyper in state_dict.keys():
                    model_dict_to_update[k] = (intrinsic_from_another.T @ state_dict[k_in_hyper]).view(v.size()).cuda()
        model_dict.update(model_dict_to_update)    
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
        
        return all_input

    def prepare_fastfood_data(self, batch): # t5的输入input_ids全部转化为input_embeds
        all_input = {
            'input_ids': batch[0].to(self.device),
            'attention_mask': batch[1].to(self.device),
            'labels': batch[2].to(self.device),
            'decoder_attention_mask': batch[3].to(self.device),
        }
        
        return all_input
    
    def prepare_prompt_data(self, batch):
        all_input = {
            'input_ids': batch[0].to(self.device),
            'attention_mask': batch[1].to(self.device),
            'labels': batch[2].to(self.device),
            'decoder_attention_mask': batch[3].to(self.device),
        }
        input_ids = all_input.pop('input_ids')
        input_embeds = self.model.get_input_embeddings()(input_ids)
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
        all_input = self.prepare_data(batch)
        output = self.model(**all_input)
        loss = output[0] / self.gradient_accumulation_steps
        loss.backward()
        return loss
    
    def hyper_train_step(self, batch):
        all_input = self.prepare_data(batch)
        loss = torch.zeros((), device=self.device)
        loss_adapter = torch.zeros((), device=self.device)
        loss_lora = torch.zeros((), device=self.device)
        loss_prefix = torch.zeros((), device=self.device)
        loss_said = torch.zeros((), device=self.device)
        if self.args.apply_adapter:
            
            output_adapter = self.model(**all_input, only_adapter=True, only_lora=False, only_prefix=False, only_said=False)
            loss_adapter = output_adapter[0] / self.gradient_accumulation_steps
            loss_adapter.backward()
            loss = loss + loss_adapter.item()
        if self.args.apply_lora:
            output_lora = self.model(**all_input, only_adapter=False, only_lora=True, only_prefix=False, only_said=False)
            loss_lora = output_lora[0] / self.gradient_accumulation_steps
            loss_lora.backward()
            loss = loss + loss_lora.item()
        if self.args.apply_prefix:
            output_prefix = self.model(**all_input, only_adapter=False, only_lora=False, only_prefix=True, only_said=False)
            loss_prefix = output_prefix[0] / self.gradient_accumulation_steps
            loss_prefix.backward()
            loss = loss + loss_prefix.item()
        if self.args.do_said:
            output_said = self.model(**all_input, only_adapter=False, only_lora=False, only_prefix=False, only_said=True)
            loss_said = output_said[0] / self.gradient_accumulation_steps
            loss_said.backward()
            loss = loss + loss_said.item()
        if not (self.args.apply_adapter or self.args.apply_lora or self.args.apply_prefix or self.args.do_said):
            output = self.model(**all_input)
            loss = output[0] / self.gradient_accumulation_steps
            loss.backward()
        
        return loss_adapter, loss_lora, loss_prefix, loss_said, loss
    
    def hyper_valid_step(self, batch, only_adapter=False, only_lora=False, only_prefix=False, only_said=False, args=None, model=None):
        
        all_input = self.prepare_data(batch)
        output = model(**all_input, only_adapter=only_adapter, only_lora=only_lora, only_prefix=only_prefix, only_said=only_said)
        valid_loss = output[0]
        
        decoder_input_ids = self.get_decoder_input_ids(all_input["input_ids"])
        generated_ids = model.generate(
            input_ids=all_input["input_ids"],
            attention_mask=all_input["attention_mask"],
            decoder_input_ids=decoder_input_ids,
            max_length=args.max_output_length,
            early_stopping=True,
            only_adapter=only_adapter,
            only_lora=only_lora,
            only_prefix=only_prefix,
            only_said=only_said,
        )
        gen_text = self.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        gen_text = list(map(str.strip, gen_text))
        return valid_loss.item(), gen_text
    
    def optimizer_step(self, parameters):
        updated = True
        scale = 0
        if self.args.optimizer == 'adamw':
            torch.nn.utils.clip_grad_norm_(parameters, self.args.max_grad_norm)
        
        self.optimizer.step()
        if updated:
            self.scheduler.step()
        self.optimizer.zero_grad()
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
        self.logger.info(message)
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
