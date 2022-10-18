# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import argparse
import logging
import shutil

import random
import numpy as np
import torch

import pandas as pd

from modeling_t5_PET_mode_connectivity_stage2 import (
    # T5Model,
    MyT5_pet_MC,
    # T5Tokenizer,
    # T5Config
)
from configuration_t5 import T5Config
from transformers import T5Tokenizer

# from run_singletask_t5 import run
from t5_trainer_PET_mc_stage2 import Trainer

def model_provider(args):
    # only the master process download model
    
    config = T5Config.from_pretrained(
        args.model,
        apply_lora=True,
        lora_alpha=args.lora_alpha,
        lora_r=args.lora_r,
        apply_adapter=True,
        adapter_type=args.adapter_type,
        adapter_size=args.adapter_size,
        apply_prefix=True,
        prefix_num=args.prefix_num,
        prefix_r=args.prefix_r,
        apply_lora_BR=args.apply_lora_BR,
        apply_bias=args.apply_bias,
        apply_bias_stage2=args.apply_bias_stage2,
        decoder_mlp=args.decoder_mlp,
        share_lora_R=args.share_lora_R,
        )
    tokenizer = T5Tokenizer.from_pretrained(args.tokenizer_path)
    model = MyT5_pet_MC(args, config=config)
    
    return model, config, tokenizer


def main():
    parser = argparse.ArgumentParser()

    ## Basic parameters
    parser.add_argument("--task_dir", default="data", required=True)
    parser.add_argument("--train_file", default="data", required=False)
    parser.add_argument("--dev_file", default="data", required=False)
    parser.add_argument("--test_file", default="data", required=False)
    parser.add_argument("--dataset", default="nlp_forest_single", required=False)
    parser.add_argument("--model", default="facebook/t5-base", required=False)
    parser.add_argument("--tokenizer_path", default="facebook/t5-base", required=False)
    
    parser.add_argument("--output_dir", default=None, type=str, required=True)
    parser.add_argument("--do_train", action='store_true')
    parser.add_argument("--do_predict", action='store_true')
    parser.add_argument("--predict_checkpoint", type=str, default="best-model.pt")

    ## Model parameters
    parser.add_argument("--checkpoint", type=str)
    parser.add_argument("--do_lowercase", action='store_true', default=False)
    parser.add_argument("--freeze_embeds", action='store_true', default=False)

    # Preprocessing/decoding-related parameters
    parser.add_argument('--max_input_length', type=int, default=512)
    parser.add_argument('--max_output_length', type=int, default=64)
    parser.add_argument('--num_beams', type=int, default=4)
    parser.add_argument("--append_another_bos", action='store_true', default=False)

    # Training-related parameters
    parser.add_argument("--train_batch_size", default=64, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--predict_batch_size", default=64, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--learning_rate", default=3e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--warmup_proportion", default=0.01, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--weight_decay", default=0.01, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--optimizer", type=str, default='adafactor')
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=0.1, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int,
                        help="Max gradient norm.")
    parser.add_argument("--train_epochs", default=100000, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_steps", default=500, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--warmup_rate", default=0.06)
    parser.add_argument("--lr_decay_style", default="constant")
    parser.add_argument("--train_iters", default=100000, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument('--wait_step', type=int, default=10000000000)

    # Other parameters
    parser.add_argument("--quiet", action='store_true',
                        help="If true, all of the warnings related to data processing will be printed. "
                             "A number of warnings are expected for a normal SQuAD evaluation.")
    parser.add_argument('--valid_interval', type=int, default=2000,
                        help="Evaluate & save model")
    parser.add_argument("--output_interval", type=int, default=2000)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--early_stop", type=int, default=-1)
    parser.add_argument('--prefix', type=str, default='',
                        help="Prefix for saving predictions")
    parser.add_argument('--debug', action='store_true',
                        help="Use a subset of data for debugging")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    
    # to tune
    parser.add_argument("--learning_rate_list", nargs="*", type=float, default=[])
    parser.add_argument("--bsz_list", nargs="*", type=int, default=[])

    # to prompt tuning
    parser.add_argument("--prompt_num", type=int, default=100)
    # parser.add_argument("--do_prompt", action='store_true', help="prompt tuning or not")
    parser.add_argument("--tune_method", type=str, help="model or prompt or adapter or lora or lora_stage2 or bias or bias_stage2 or hyper_PET or PET_mc or PET_mc_stage2")
    parser.add_argument("--do_inherit_prompt", action='store_true', help="inherit prompt or not")
    parser.add_argument("--inherit_prompt_path", type=str)
    parser.add_argument("--one_prefix", action='store_true')

    # LoRA
    parser.add_argument("--apply_adapter", action='store_true')
    parser.add_argument("--adapter_type", type=str, default='houlsby')
    parser.add_argument("--adapter_size", type=int, default=12)
    parser.add_argument("--apply_lora", action='store_true')
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_r", type=int, default=10)
    
    # LoRA stage2
    parser.add_argument("--apply_lora_BR", action='store_true')
    parser.add_argument("--load_lora_B_path", type=str)
    parser.add_argument("--load_random_B", action='store_true')
    parser.add_argument("--share_lora_R", action='store_true')
    
    # bias
    parser.add_argument("--apply_bias", action='store_true')
    parser.add_argument("--decoder_mlp",action='store_true')
    
    # bias stage2
    parser.add_argument("--apply_bias_stage2", action='store_true')
    parser.add_argument("--load_bias_path", type=str)
    
    parser.add_argument("--share_intrinsic", action='store_true')
    parser.add_argument("--intrinsic_dim", type=int, default=8)
    
    # prefix
    parser.add_argument("--apply_prefix", action='store_true')
    parser.add_argument("--prefix_r", type=int, default=24)
    parser.add_argument("--prefix_num", type=int, default=120)
    
    parser.add_argument("--choose_valid", action='store_true')
    parser.add_argument("--choose_valid_lines", type=int, default=1000)
    parser.add_argument("--choose_test", action='store_true')
    parser.add_argument("--choose_test_lines", type=int, default=1000)
    
    
    # stage2 compress dimension
    parser.add_argument("--load_stage1_adapter_path_list", nargs="*", type=str, default=[])
    parser.add_argument("--load_stage1_lora_path_list", nargs="*", type=str, default=[])
    parser.add_argument("--load_stage1_prefix_path_list", nargs="*", type=str, default=[])
    parser.add_argument("--low_dimension", type=int, default=100)
    
    parser.add_argument("--load_PET_enc_dec_path", type=str, default=None)
    
    
    
    args = parser.parse_args()
    if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
        print("Output directory () already exists and is not empty.")
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
    output_dir = args.output_dir

    ##### Start writing logs

    log_filename = "{}log.txt".format("" if args.do_train else "eval_")

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO,
                    handlers=[logging.FileHandler(os.path.join(args.output_dir, log_filename)),
                              logging.StreamHandler()])
    logger = logging.getLogger(__name__)
    logger.info(args)
    logger.info(args.output_dir)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.n_gpu = torch.cuda.device_count()

    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_predict:
        raise ValueError("At least one of `do_train` or `do_predict` must be True.")

    if args.do_train:
        if not args.train_file:
            raise ValueError("If `do_train` is True, then `train_dir` must be specified.")
        if not args.dev_file:
            raise ValueError("If `do_train` is True, then `predict_dir` must be specified.")

    if args.do_predict:
        if not args.test_file:
            raise ValueError("If `do_predict` is True, then `predict_dir` must be specified.")

    logger.info("Using {} gpus".format(args.n_gpu))

    files = sorted(os.listdir(args.task_dir))
    prefixes = []
    for filename in files:
        if not filename.endswith(".tsv"):
            continue
        prefix = "_".join(filename.split("_")[:-1])
        if prefix not in prefixes:
            prefixes.append(prefix)

    logger.info("Fine-tuning the following samples: {}".format(prefixes))

    df = pd.DataFrame(columns=["prefix", "metric", "lr", "bsz", "dev_performance", "test_performance_adapter", "test_performance_lora", "test_performance_prefix"])

    for prefix in prefixes:
        args.train_file = os.path.join(args.task_dir, prefix + "_train.tsv")
        args.dev_file = os.path.join(args.task_dir, prefix + "_dev.tsv")
        args.test_file = os.path.join(args.task_dir, prefix + "_test.tsv")

       
        best_dev_performance = -1.0
        best_model_prompt_weight = torch.Tensor()
        best_config = None
        for bsz in args.bsz_list:
            for lr in args.learning_rate_list:
                
                args.learning_rate = lr
                if bsz > 16:
                    args.train_batch_size = 16
                    args.gradient_accumulation_steps = int(bsz // 16)
                else:
                    args.train_batch_size = bsz
                    args.gradient_accumulation_steps = 1
                
                args.output_dir = output_dir + '/lr_' +str(lr)+'_bsz_'+str(bsz)+'_seed_'+str(args.seed)
                
                logger.info("Running ... prefix={}, lr={}, bsz={} ...".format(prefix, lr, bsz))
                trainer = Trainer(args, logger, model_provider)
                dev_performance = None
                test_performance = None
                if args.do_train:
                    dev_performance = trainer.train()
                if args.do_predict:
                    metrics, metric_of_task = trainer.test()
                    for metrics_key in metrics.keys():
                        if 'adapter' in metrics_key:
                            test_performance_adapter = metrics[metrics_key]
                        if 'lora' in metrics_key:
                            test_performance_lora = metrics[metrics_key]
                        if 'prefix' in metrics_key:
                            test_performance_prefix = metrics[metrics_key]
                
                logger.info("prefix={}, lr={}, bsz={}, dev_performance={}, test_performance={}".format(prefix, lr, bsz, dev_performance, test_performance))
                df.loc[len(df.index)] = [prefix, metric_of_task, lr, bsz, dev_performance, test_performance_adapter, test_performance_lora, test_performance_prefix]
                df.to_csv(os.path.join(output_dir, "result.csv"),sep=',',index=False,header=True)
        
        if args.one_prefix:
            break

if __name__=='__main__':
    main()
